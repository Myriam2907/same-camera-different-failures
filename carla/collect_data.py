import os
import sys
import csv
import queue
import shutil
import argparse

import carla
import numpy as np


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.projection import (
    build_camera_matrix,
    world_to_image,
    interpolate_x_at_row,
)


# ============================================================
# CARLA CONFIGURATION
# ============================================================

HOST = "127.0.0.1"
PORT = 2000

FIXED_DELTA = 0.05

IMAGE_W = 1280
IMAGE_H = 720
FOV = 90.0

LOOKAHEAD_ROW = int(round(0.70 * IMAGE_H))


# ============================================================
# FIXED STOP-SIGN EXPERIMENT
# ============================================================

TARGET_STOP_HINT = carla.Location(
    x=26.10,
    y=75.15,
    z=0.15,
)

CAPTURE_UPSTREAM_M = 50.0
CAPTURE_DOWNSTREAM_M = 10.0

ROUTE_STEP_M = 0.5

# Keep extra future route points for lane GT projection.
INTERNAL_DOWNSTREAM_M = 70.0
LOOK_FORWARD_M = 60.0

VEHICLE_Z_OFFSET_M = 0.25


# ============================================================
# FROZEN ROUTE
# ============================================================

ROUTE_DIR = os.path.join(
    "data",
    "routes",
)

ROUTE_FILE = os.path.join(
    ROUTE_DIR,
    "stop3_route.csv",
)

EXPECTED_ROUTE_POINTS = int(
    round(
        (
            CAPTURE_UPSTREAM_M
            + INTERNAL_DOWNSTREAM_M
        )
        / ROUTE_STEP_M
    )
) + 1

EXPECTED_CAPTURE_SAMPLES = int(
    round(
        (
            CAPTURE_UPSTREAM_M
            + CAPTURE_DOWNSTREAM_M
        )
        / ROUTE_STEP_M
    )
) + 1


# ============================================================
# POSE VERIFICATION
# ============================================================

POSE_LOCATION_TOL_M = 0.01
POSE_ANGLE_TOL_DEG = 0.01


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Record the frozen STOP-sign experiment route."
        )
    )

    parser.add_argument(
        "--weather",
        choices=["clear", "fog"],
        required=True,
    )

    parser.add_argument(
        "--severity",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--reset-route",
        action="store_true",
        help=(
            "Rebuild and overwrite the canonical route."
        ),
    )

    return parser.parse_args()


# ============================================================
# GENERAL HELPERS
# ============================================================

def angle_difference(a, b):

    return abs(
        (a - b + 180.0) % 360.0 - 180.0
    )


def yaw_difference(a, b):

    return angle_difference(a, b)


def choose_route_waypoint(
    current_wp,
    candidates,
):

    if not candidates:
        return None

    same_lane = [
        wp
        for wp in candidates
        if (
            wp.road_id == current_wp.road_id
            and wp.lane_id == current_wp.lane_id
        )
    ]

    pool = (
        same_lane
        if same_lane
        else candidates
    )

    return min(
        pool,
        key=lambda wp: (
            yaw_difference(
                current_wp.transform.rotation.yaw,
                wp.transform.rotation.yaw,
            ),
            wp.road_id,
            wp.lane_id,
        ),
    )


def get_matching_image(
    image_queue,
    target_frame,
):

    while True:

        image = image_queue.get(
            timeout=10.0
        )

        if image.frame < target_frame:
            continue

        if image.frame == target_frame:
            return image

        raise RuntimeError(
            f"Camera skipped frame "
            f"{target_frame}; "
            f"received {image.frame}"
        )


# ============================================================
# WEATHER
# ============================================================

def make_weather(
    weather_name,
    severity,
):

    if not 0 <= severity <= 100:
        raise ValueError(
            "Severity must be between 0 and 100."
        )

    if weather_name == "clear":

        if severity != 0:
            raise ValueError(
                "Clear weather must use severity 0."
            )

        weather = carla.WeatherParameters.ClearNoon

        weather.fog_density = 0.0
        weather.fog_distance = 0.0
        weather.fog_falloff = 0.2

        return weather

    if weather_name == "fog":

        weather = carla.WeatherParameters.ClearNoon

        weather.fog_density = float(severity)
        weather.fog_distance = 0.0
        weather.fog_falloff = 0.2

        return weather

    raise ValueError(
        f"Unknown weather: {weather_name}"
    )


# ============================================================
# STOP SIGN HELPERS
# ============================================================

def find_target_stop(world):

    stop_actors = [
        actor
        for actor in world.get_actors()
        if "traffic.stop" in actor.type_id
    ]

    if not stop_actors:
        raise RuntimeError(
            "No traffic.stop actors found."
        )

    target_stop = min(
        stop_actors,
        key=lambda actor: (
            actor.get_location().distance(
                TARGET_STOP_HINT
            )
        ),
    )

    distance = (
        target_stop
        .get_location()
        .distance(TARGET_STOP_HINT)
    )

    if distance > 5.0:
        raise RuntimeError(
            "Could not confidently locate "
            "the selected STOP sign."
        )

    return target_stop


def get_stop_trigger_world_location(
    stop_actor,
):

    local = (
        stop_actor.trigger_volume.location
    )

    local_copy = carla.Location(
        x=local.x,
        y=local.y,
        z=local.z,
    )

    return (
        stop_actor
        .get_transform()
        .transform(local_copy)
    )


def find_visible_sign_center(
    world,
    stop_actor,
):

    boxes = world.get_level_bbs(
        carla.CityObjectLabel.TrafficSigns
    )

    if not boxes:
        raise RuntimeError(
            "No TrafficSigns bounding boxes found."
        )

    stop_location = (
        stop_actor.get_location()
    )

    best_box = min(
        boxes,
        key=lambda bb: (
            bb.location.distance(
                stop_location
            )
        ),
    )

    distance = (
        best_box.location.distance(
            stop_location
        )
    )

    if distance > 5.0:
        raise RuntimeError(
            "Could not match visible STOP-sign "
            "bounding box."
        )

    return best_box.location


# ============================================================
# BUILD ROUTE FROM CARLA MAP
# ============================================================

def build_route_from_map(
    carla_map,
    stop_actor,
):

    trigger_world = (
        get_stop_trigger_world_location(
            stop_actor
        )
    )

    anchor_wp = carla_map.get_waypoint(
        trigger_world,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if anchor_wp is None:
        raise RuntimeError(
            "Could not locate driving lane "
            "for STOP trigger."
        )

    # --------------------------------------------------------
    # UPSTREAM
    # --------------------------------------------------------

    upstream_steps = int(
        round(
            CAPTURE_UPSTREAM_M
            / ROUTE_STEP_M
        )
    )

    upstream = []

    current_wp = anchor_wp

    for step_index in range(
        1,
        upstream_steps + 1,
    ):

        candidates = current_wp.previous(
            ROUTE_STEP_M
        )

        previous_wp = choose_route_waypoint(
            current_wp,
            candidates,
        )

        if previous_wp is None:
            raise RuntimeError(
                "Route ended before reaching "
                f"{CAPTURE_UPSTREAM_M} m upstream."
            )

        route_s = (
            -step_index
            * ROUTE_STEP_M
        )

        upstream.append(
            (
                route_s,
                previous_wp,
            )
        )

        current_wp = previous_wp

    upstream.reverse()

    # --------------------------------------------------------
    # ANCHOR
    # --------------------------------------------------------

    route_wp = list(upstream)

    route_wp.append(
        (
            0.0,
            anchor_wp,
        )
    )

    # --------------------------------------------------------
    # DOWNSTREAM
    # --------------------------------------------------------

    downstream_steps = int(
        round(
            INTERNAL_DOWNSTREAM_M
            / ROUTE_STEP_M
        )
    )

    current_wp = anchor_wp

    for step_index in range(
        1,
        downstream_steps + 1,
    ):

        candidates = current_wp.next(
            ROUTE_STEP_M
        )

        next_wp = choose_route_waypoint(
            current_wp,
            candidates,
        )

        if next_wp is None:
            raise RuntimeError(
                "Route ended before reaching "
                f"{INTERNAL_DOWNSTREAM_M} m downstream."
            )

        route_s = (
            step_index
            * ROUTE_STEP_M
        )

        route_wp.append(
            (
                route_s,
                next_wp,
            )
        )

        current_wp = next_wp

    # --------------------------------------------------------
    # CONVERT INTO PLAIN FROZEN DATA
    # --------------------------------------------------------

    route = []

    for index, (
        route_s,
        waypoint,
    ) in enumerate(route_wp):

        transform = waypoint.transform

        route.append({
            "route_index": index,
            "route_s_m": float(route_s),

            "lane_x":
                float(transform.location.x),

            "lane_y":
                float(transform.location.y),

            "lane_z":
                float(transform.location.z),

            "pitch":
                float(transform.rotation.pitch),

            "yaw":
                float(transform.rotation.yaw),

            "roll":
                float(transform.rotation.roll),

            "is_junction":
                int(bool(waypoint.is_junction)),

            "road_id":
                int(waypoint.road_id),

            "lane_id":
                int(waypoint.lane_id),
        })

    return route


# ============================================================
# ROUTE FILE
# ============================================================

def save_route(
    route,
    path,
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    fieldnames = [
        "route_index",
        "route_s_m",
        "lane_x",
        "lane_y",
        "lane_z",
        "pitch",
        "yaw",
        "roll",
        "is_junction",
        "road_id",
        "lane_id",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in route:

            writer.writerow({
                "route_index":
                    row["route_index"],

                "route_s_m":
                    f'{row["route_s_m"]:.6f}',

                "lane_x":
                    f'{row["lane_x"]:.9f}',

                "lane_y":
                    f'{row["lane_y"]:.9f}',

                "lane_z":
                    f'{row["lane_z"]:.9f}',

                "pitch":
                    f'{row["pitch"]:.9f}',

                "yaw":
                    f'{row["yaw"]:.9f}',

                "roll":
                    f'{row["roll"]:.9f}',

                "is_junction":
                    row["is_junction"],

                "road_id":
                    row["road_id"],

                "lane_id":
                    row["lane_id"],
            })


def load_route(path):

    if not os.path.exists(path):
        raise RuntimeError(
            f"Frozen route does not exist: {path}"
        )

    route = []

    with open(
        path,
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            route.append({
                "route_index":
                    int(row["route_index"]),

                "route_s_m":
                    float(row["route_s_m"]),

                "lane_x":
                    float(row["lane_x"]),

                "lane_y":
                    float(row["lane_y"]),

                "lane_z":
                    float(row["lane_z"]),

                "pitch":
                    float(row["pitch"]),

                "yaw":
                    float(row["yaw"]),

                "roll":
                    float(row["roll"]),

                "is_junction":
                    int(row["is_junction"]),

                "road_id":
                    int(row["road_id"]),

                "lane_id":
                    int(row["lane_id"]),
            })

    validate_route(route)

    return route


def validate_route(route):

    if len(route) != EXPECTED_ROUTE_POINTS:

        raise RuntimeError(
            f"Frozen route has {len(route)} points, "
            f"expected {EXPECTED_ROUTE_POINTS}."
        )

    indices = np.array([
        row["route_index"]
        for row in route
    ])

    expected_indices = np.arange(
        len(route)
    )

    if not np.array_equal(
        indices,
        expected_indices,
    ):
        raise RuntimeError(
            "Frozen route indices are invalid."
        )

    route_s = np.array([
        row["route_s_m"]
        for row in route
    ])

    expected_s = np.arange(
        -CAPTURE_UPSTREAM_M,
        INTERNAL_DOWNSTREAM_M
        + ROUTE_STEP_M / 2.0,
        ROUTE_STEP_M,
    )

    if not np.allclose(
        route_s,
        expected_s,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Frozen route spacing is invalid."
        )


def prepare_route(
    carla_map,
    stop_actor,
    reset_route,
):

    if reset_route:

        print(
            "Rebuilding frozen route..."
        )

        route = build_route_from_map(
            carla_map,
            stop_actor,
        )

        validate_route(route)

        save_route(
            route,
            ROUTE_FILE,
        )

        print(
            "Frozen route saved:",
            ROUTE_FILE,
        )

        return route

    if os.path.exists(
        ROUTE_FILE
    ):

        print(
            "Loading frozen route:",
            ROUTE_FILE,
        )

        return load_route(
            ROUTE_FILE
        )

    print(
        "Frozen route not found."
    )

    print(
        "Building it automatically..."
    )

    route = build_route_from_map(
        carla_map,
        stop_actor,
    )

    validate_route(route)

    save_route(
        route,
        ROUTE_FILE,
    )

    print(
        "Frozen route saved:",
        ROUTE_FILE,
    )

    return route


# ============================================================
# ROUTE POINT -> VEHICLE TRANSFORM
# ============================================================

def route_point_to_vehicle_transform(
    route_point,
):

    return carla.Transform(
        carla.Location(
            x=route_point["lane_x"],
            y=route_point["lane_y"],
            z=(
                route_point["lane_z"]
                + VEHICLE_Z_OFFSET_M
            ),
        ),
        carla.Rotation(
            pitch=route_point["pitch"],
            yaw=route_point["yaw"],
            roll=route_point["roll"],
        ),
    )


# ============================================================
# VERIFY ACTUAL VEHICLE POSE
# ============================================================

def verify_actual_pose(
    planned_transform,
    actual_transform,
):

    p = planned_transform.location
    a = actual_transform.location

    location_error = float(
        np.sqrt(
            (a.x - p.x) ** 2
            + (a.y - p.y) ** 2
            + (a.z - p.z) ** 2
        )
    )

    planned_rot = (
        planned_transform.rotation
    )

    actual_rot = (
        actual_transform.rotation
    )

    pitch_error = angle_difference(
        actual_rot.pitch,
        planned_rot.pitch,
    )

    yaw_error = angle_difference(
        actual_rot.yaw,
        planned_rot.yaw,
    )

    roll_error = angle_difference(
        actual_rot.roll,
        planned_rot.roll,
    )

    max_angle_error = max(
        pitch_error,
        yaw_error,
        roll_error,
    )

    if (
        location_error
        > POSE_LOCATION_TOL_M
    ):
        raise RuntimeError(
            "Vehicle pose location mismatch: "
            f"{location_error:.6f} m"
        )

    if (
        max_angle_error
        > POSE_ANGLE_TOL_DEG
    ):
        raise RuntimeError(
            "Vehicle pose rotation mismatch: "
            f"{max_angle_error:.6f} deg"
        )

    return (
        location_error,
        max_angle_error,
    )


# ============================================================
# LANE GROUND TRUTH
# ============================================================

def compute_lane_gt_x(
    route,
    route_index,
    image,
    K,
):

    current_s = (
        route[route_index][
            "route_s_m"
        ]
    )

    world_to_camera = np.array(
        image.transform.get_inverse_matrix()
    )

    projected = []

    for index in range(
        route_index,
        len(route),
    ):

        point = route[index]

        forward_distance = (
            point["route_s_m"]
            - current_s
        )

        if (
            forward_distance
            > LOOK_FORWARD_M
        ):
            break

        lane_location = carla.Location(
            x=point["lane_x"],
            y=point["lane_y"],
            z=point["lane_z"],
        )

        result = world_to_image(
            lane_location,
            world_to_camera,
            K,
        )

        if result is None:
            continue

        u, v, depth = result

        if not (
            np.isfinite(u)
            and np.isfinite(v)
            and np.isfinite(depth)
        ):
            continue

        projected.append(
            (u, v)
        )

    if len(projected) < 2:
        return None

    lane_gt_x = interpolate_x_at_row(
        projected,
        LOOKAHEAD_ROW,
    )

    if lane_gt_x is None:
        return None

    if not np.isfinite(
        lane_gt_x
    ):
        return None

    if not (
        0.0
        <= lane_gt_x
        < IMAGE_W
    ):
        return None

    return float(
        lane_gt_x
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    weather_name = args.weather
    severity = args.severity

    weather = make_weather(
        weather_name,
        severity,
    )

    experiment_name = (
        f"stop3_{weather_name}_{severity}"
    )

    output_root = os.path.join(
        "data",
        "recordings",
        experiment_name,
    )

    frame_dir = os.path.join(
        output_root,
        "frames",
    )

    csv_path = os.path.join(
        output_root,
        "frames.csv",
    )

    # --------------------------------------------------------
    # Replace old recording for this weather
    # --------------------------------------------------------

    if os.path.exists(
        output_root
    ):
        shutil.rmtree(
            output_root
        )

    os.makedirs(
        frame_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Connect to CARLA
    # --------------------------------------------------------

    client = carla.Client(
        HOST,
        PORT,
    )

    client.set_timeout(
        10.0
    )

    world = client.get_world()
    carla_map = world.get_map()

    print(
        "Connected to:",
        carla_map.name,
    )

    original_settings = (
        world.get_settings()
    )

    original_weather = (
        world.get_weather()
    )

    vehicle = None
    camera = None

    try:

        # ====================================================
        # SYNCHRONOUS MODE
        # ====================================================

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = (
            FIXED_DELTA
        )

        world.apply_settings(
            settings
        )

        print(
            "Synchronous mode: ON"
        )

        # ====================================================
        # STOP SIGN
        # ====================================================

        stop_actor = find_target_stop(
            world
        )

        stop_location = (
            stop_actor.get_location()
        )

        print(
            "STOP actor:",
            stop_actor.id,
        )

        print(
            "STOP actor location:",
            f"({stop_location.x:.2f}, "
            f"{stop_location.y:.2f}, "
            f"{stop_location.z:.2f})",
        )

        sign_center = (
            find_visible_sign_center(
                world,
                stop_actor,
            )
        )

        print(
            "Visible sign center:",
            f"({sign_center.x:.2f}, "
            f"{sign_center.y:.2f}, "
            f"{sign_center.z:.2f})",
        )

        # ====================================================
        # FROZEN ROUTE
        # ====================================================

        route = prepare_route(
            carla_map,
            stop_actor,
            args.reset_route,
        )

        capture_indices = [
            index
            for index, point
            in enumerate(route)
            if (
                -CAPTURE_UPSTREAM_M
                <= point["route_s_m"]
                <= CAPTURE_DOWNSTREAM_M
            )
        ]

        if (
            len(capture_indices)
            != EXPECTED_CAPTURE_SAMPLES
        ):
            raise RuntimeError(
                f"Expected "
                f"{EXPECTED_CAPTURE_SAMPLES} samples, "
                f"got {len(capture_indices)}."
            )

        print(
            "Frozen route points:",
            len(route),
        )

        print(
            "Captured samples:",
            len(capture_indices),
        )

        print(
            "Capture range:",
            f"-{CAPTURE_UPSTREAM_M:.1f} m "
            f"to +{CAPTURE_DOWNSTREAM_M:.1f} m",
        )

        print(
            "Route spacing:",
            ROUTE_STEP_M,
            "m",
        )

        # ====================================================
        # WEATHER
        # ====================================================

        world.set_weather(
            weather
        )

        print(
            "Weather:",
            weather_name,
        )

        print(
            "Severity:",
            severity,
        )

        world.tick()

        # ====================================================
        # SPAWN VEHICLE
        # ====================================================

        blueprint_library = (
            world.get_blueprint_library()
        )

        vehicle_bp = (
            blueprint_library.filter(
                "vehicle.tesla.model3"
            )[0]
        )

        for spawn in (
            carla_map.get_spawn_points()
        ):

            vehicle = (
                world.try_spawn_actor(
                    vehicle_bp,
                    spawn,
                )
            )

            if vehicle is not None:
                break

        if vehicle is None:
            raise RuntimeError(
                "Could not spawn ego vehicle."
            )

        vehicle.set_simulate_physics(
            False
        )

        first_point = route[
            capture_indices[0]
        ]

        first_transform = (
            route_point_to_vehicle_transform(
                first_point
            )
        )

        vehicle.set_transform(
            first_transform
        )

        print(
            "Vehicle spawned"
        )

        print(
            "Physics: OFF "
            "(frozen-pose experiment)"
        )

        # ====================================================
        # CAMERA
        # ====================================================

        camera_bp = (
            blueprint_library.find(
                "sensor.camera.rgb"
            )
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(IMAGE_W),
        )

        camera_bp.set_attribute(
            "image_size_y",
            str(IMAGE_H),
        )

        camera_bp.set_attribute(
            "fov",
            str(FOV),
        )

        camera_bp.set_attribute(
            "sensor_tick",
            "0.0",
        )

        if camera_bp.has_attribute(
            "motion_blur_intensity"
        ):
            camera_bp.set_attribute(
                "motion_blur_intensity",
                "0.0",
            )

        camera_transform = carla.Transform(
            carla.Location(
                x=1.5,
                y=0.0,
                z=1.7,
            ),
            carla.Rotation(
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
            ),
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle,
            attachment_type=(
                carla.AttachmentType.Rigid
            ),
        )

        image_queue = queue.Queue()

        camera.listen(
            image_queue.put
        )

        print(
            "RGB camera attached"
        )

        print(
            f"Camera: "
            f"{IMAGE_W}x{IMAGE_H}, "
            f"FOV={FOV}"
        )

        print(
            "Lane GT row:",
            LOOKAHEAD_ROW,
        )

        # ====================================================
        # CAMERA MATRIX
        # ====================================================

        K = build_camera_matrix(
            IMAGE_W,
            IMAGE_H,
            FOV,
        )

        # ====================================================
        # WARM UP
        # ====================================================

        print(
            "Warming up..."
        )

        for _ in range(5):

            target_frame = (
                world.tick()
            )

            get_matching_image(
                image_queue,
                target_frame,
            )

        print(
            "Warm-up complete"
        )

        # ====================================================
        # RECORD
        # ====================================================

        junction_count = 0
        nonjunction_missing_gt = 0

        max_location_error = 0.0
        max_angle_error = 0.0

        with open(
            csv_path,
            "w",
            newline="",
        ) as csv_file:

            writer = csv.writer(
                csv_file
            )

            writer.writerow([
                "sample_id",
                "frame",
                "weather",
                "severity",
                "route_s_m",
                "ego_x",
                "ego_y",
                "ego_z",
                "ego_pitch",
                "ego_yaw",
                "ego_roll",
                "distance_to_sign_m",
                "is_junction",
                "lane_gt_x",
            ])

            total_samples = len(
                capture_indices
            )

            for (
                sample_id,
                route_index,
            ) in enumerate(
                capture_indices,
                start=1,
            ):

                point = route[
                    route_index
                ]

                # ============================================
                # SET EXACT FROZEN POSE
                # ============================================

                planned_transform = (
                    route_point_to_vehicle_transform(
                        point
                    )
                )

                vehicle.set_transform(
                    planned_transform
                )

                # ============================================
                # IMPORTANT FIX:
                # SETTLING TICK
                #
                # CARLA can keep the previous actor/camera
                # pose for the first simulation tick after
                # set_transform().
                #
                # Therefore:
                #   tick 1 = settle, discard
                #   tick 2 = actual capture
                # ============================================

                settle_frame = (
                    world.tick()
                )

                get_matching_image(
                    image_queue,
                    settle_frame,
                )

                # ============================================
                # CAPTURE TICK
                # ============================================

                target_frame = (
                    world.tick()
                )

                image = get_matching_image(
                    image_queue,
                    target_frame,
                )

                # ============================================
                # VERIFY VEHICLE POSE
                # ============================================

                actual_transform = (
                    vehicle.get_transform()
                )

                (
                    location_error,
                    angle_error,
                ) = verify_actual_pose(
                    planned_transform,
                    actual_transform,
                )

                max_location_error = max(
                    max_location_error,
                    location_error,
                )

                max_angle_error = max(
                    max_angle_error,
                    angle_error,
                )

                actual_location = (
                    actual_transform.location
                )

                actual_rotation = (
                    actual_transform.rotation
                )

                # ============================================
                # DISTANCE TO PHYSICAL STOP SIGN
                # ============================================

                camera_location = (
                    image.transform.location
                )

                distance_to_sign = (
                    camera_location.distance(
                        sign_center
                    )
                )

                # ============================================
                # LANE GT
                # ============================================

                is_junction = bool(
                    point["is_junction"]
                )

                if is_junction:

                    junction_count += 1

                    lane_gt_x = None

                else:

                    lane_gt_x = (
                        compute_lane_gt_x(
                            route,
                            route_index,
                            image,
                            K,
                        )
                    )

                    if lane_gt_x is None:
                        nonjunction_missing_gt += 1

                if lane_gt_x is None:

                    lane_gt_value = ""

                else:

                    lane_gt_value = (
                        f"{lane_gt_x:.6f}"
                    )

                # ============================================
                # SAVE IMAGE
                # ============================================

                image_path = os.path.join(
                    frame_dir,
                    f"{sample_id:04d}.png",
                )

                image.save_to_disk(
                    image_path
                )

                # ============================================
                # SAVE CSV
                # ============================================

                writer.writerow([
                    sample_id,
                    target_frame,
                    weather_name,
                    severity,
                    f'{point["route_s_m"]:.3f}',

                    f"{actual_location.x:.9f}",
                    f"{actual_location.y:.9f}",
                    f"{actual_location.z:.9f}",

                    f"{actual_rotation.pitch:.9f}",
                    f"{actual_rotation.yaw:.9f}",
                    f"{actual_rotation.roll:.9f}",

                    f"{distance_to_sign:.6f}",

                    int(is_junction),

                    lane_gt_value,
                ])

                # ============================================
                # PROGRESS
                # ============================================

                if (
                    sample_id == 1
                    or sample_id % 20 == 0
                    or sample_id == total_samples
                ):

                    print(
                        f"Saved "
                        f"{sample_id}/{total_samples} "
                        f"| route_s="
                        f'{point["route_s_m"]:+.1f} m '
                        f"| sign_dist="
                        f"{distance_to_sign:.2f} m "
                        f"| junction="
                        f"{int(is_junction)} "
                        f"| lane_gt_x="
                        f"{lane_gt_x}"
                    )

        # ====================================================
        # SUMMARY
        # ====================================================

        print()

        print(
            "RECORDING COMPLETE"
        )

        print(
            "Experiment:",
            experiment_name,
        )

        print(
            "Frozen route:",
            ROUTE_FILE,
        )

        print(
            "Frames:",
            frame_dir,
        )

        print(
            "CSV:",
            csv_path,
        )

        print(
            "Captured samples:",
            len(capture_indices),
        )

        print(
            "Junction frames "
            "(lane GT intentionally blank):",
            junction_count,
        )

        print(
            "Non-junction missing lane GT:",
            nonjunction_missing_gt,
        )

        print(
            "Max vehicle location pose error:",
            f"{max_location_error:.9f}",
            "m",
        )

        print(
            "Max vehicle angle pose error:",
            f"{max_angle_error:.9f}",
            "deg",
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        print()

        print(
            "Cleaning up..."
        )

        if camera is not None:
            camera.stop()
            camera.destroy()

        if vehicle is not None:
            vehicle.destroy()

        world.set_weather(
            original_weather
        )

        world.apply_settings(
            original_settings
        )

        print(
            "CARLA world settings restored"
        )


if __name__ == "__main__":
    main()