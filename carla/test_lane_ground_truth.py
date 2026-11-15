import os
import sys
import queue

import carla
import cv2
import numpy as np


# ------------------------------------------------------------
# Import project helper
# ------------------------------------------------------------

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
# CONFIGURATION
# ============================================================

HOST = "127.0.0.1"
PORT = 2000
TM_PORT = 8000

FIXED_DELTA = 0.05

IMAGE_W = 1280
IMAGE_H = 720
FOV = 90.0

# 70% down from top of image.
# round() ensures 0.70 * 720 becomes exactly row 504.
LOOKAHEAD_ROW = int(round(0.70 * IMAGE_H))

WAYPOINT_STEP_M = 2.0
LOOK_FORWARD_M = 60.0

NUM_IMAGES = 20

OUTPUT_DIR = "data/lane_gt_test"


# ============================================================
# HELPERS
# ============================================================

def yaw_difference(a, b):
    """
    Return the smallest absolute yaw difference in degrees.
    """
    return abs(
        (a - b + 180.0) % 360.0 - 180.0
    )


def choose_next_waypoint(current_wp, candidates):
    """
    Choose the next lane-center waypoint.

    Prefer staying on the same road and lane.
    If CARLA returns multiple possible branches,
    choose the one with the smallest heading change.
    """

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

    if same_lane:
        pool = same_lane
    else:
        pool = candidates

    return min(
        pool,
        key=lambda wp: yaw_difference(
            current_wp.transform.rotation.yaw,
            wp.transform.rotation.yaw,
        ),
    )


def get_future_lane_waypoints(
    carla_map,
    ego_location,
):
    """
    Project the ego location onto the driving lane and then
    follow the lane-center waypoints forward.
    """

    current_wp = carla_map.get_waypoint(
        ego_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if current_wp is None:
        return []

    waypoints = [current_wp]

    travelled = 0.0

    while travelled < LOOK_FORWARD_M:

        candidates = current_wp.next(
            WAYPOINT_STEP_M
        )

        next_wp = choose_next_waypoint(
            current_wp,
            candidates,
        )

        if next_wp is None:
            break

        waypoints.append(next_wp)

        current_wp = next_wp
        travelled += WAYPOINT_STEP_M

    return waypoints


def get_matching_image(
    image_queue,
    target_frame,
):
    """
    Return the camera image corresponding exactly
    to the requested CARLA simulation frame.
    """

    while True:

        image = image_queue.get(
            timeout=10.0
        )

        if image.frame < target_frame:
            continue

        if image.frame == target_frame:
            return image

        raise RuntimeError(
            f"Camera skipped CARLA frame "
            f"{target_frame}; received {image.frame}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Connect to CARLA
    # --------------------------------------------------------

    client = carla.Client(
        HOST,
        PORT,
    )

    client.set_timeout(10.0)

    world = client.get_world()
    carla_map = world.get_map()

    print(
        "Connected to:",
        carla_map.name,
    )

    print(
        f"Camera: {IMAGE_W}x{IMAGE_H}, "
        f"FOV={FOV}"
    )

    print(
        "Lookahead row:",
        LOOKAHEAD_ROW,
    )

    original_settings = world.get_settings()
    original_weather = world.get_weather()

    traffic_manager = (
        client.get_trafficmanager(
            TM_PORT
        )
    )

    vehicle = None
    camera = None

    try:

        # ====================================================
        # 1. SYNCHRONOUS MODE
        # ====================================================

        settings = world.get_settings()

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = (
            FIXED_DELTA
        )

        world.apply_settings(
            settings
        )

        traffic_manager.set_synchronous_mode(
            True
        )

        world.set_weather(
            carla.WeatherParameters.ClearNoon
        )

        # ====================================================
        # 2. SPAWN EGO VEHICLE
        # ====================================================

        blueprint_library = (
            world.get_blueprint_library()
        )

        vehicle_bp = (
            blueprint_library.filter(
                "vehicle.tesla.model3"
            )[0]
        )

        spawn_points = (
            carla_map.get_spawn_points()
        )

        for index, spawn in enumerate(
            spawn_points
        ):

            vehicle = world.try_spawn_actor(
                vehicle_bp,
                spawn,
            )

            if vehicle is not None:

                print(
                    "Vehicle spawn index:",
                    index,
                )

                break

        if vehicle is None:

            raise RuntimeError(
                "Could not spawn ego vehicle"
            )

        vehicle.set_autopilot(
            True,
            TM_PORT,
        )

        print("Autopilot: ON")

        # Do not allow automatic lane changes
        # during the GT verification.
        traffic_manager.auto_lane_change(
            vehicle,
            False,
        )

        # ====================================================
        # 3. FRONT RGB CAMERA
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

        print("RGB camera attached")

        image_queue = queue.Queue()

        camera.listen(
            image_queue.put
        )

        # ====================================================
        # 4. CAMERA INTRINSIC MATRIX
        # ====================================================

        K = build_camera_matrix(
            IMAGE_W,
            IMAGE_H,
            FOV,
        )

        # ====================================================
        # 5. SENSOR WARM-UP
        # ====================================================

        print("Warming up...")

        for _ in range(10):

            frame = world.tick()

            get_matching_image(
                image_queue,
                frame,
            )

        print("Warm-up complete")

        # ====================================================
        # 6. LANE GROUND-TRUTH VISUAL TEST
        # ====================================================

        saved = 0

        while saved < NUM_IMAGES:

            target_frame = world.tick()

            image = get_matching_image(
                image_queue,
                target_frame,
            )

            # Save one example every 10 simulation frames.
            if target_frame % 10 != 0:
                continue

            # ------------------------------------------------
            # Get current ego position
            # ------------------------------------------------

            ego_location = (
                vehicle.get_location()
            )

            # ------------------------------------------------
            # Get future CARLA lane-center waypoints
            # ------------------------------------------------

            future_waypoints = (
                get_future_lane_waypoints(
                    carla_map,
                    ego_location,
                )
            )

            # ------------------------------------------------
            # Camera world -> camera transform
            # ------------------------------------------------

            world_to_camera = np.array(
                image.transform.get_inverse_matrix()
            )

            # ------------------------------------------------
            # Project lane-center waypoints into image
            # ------------------------------------------------

            projected_points = []

            for wp in future_waypoints:

                result = world_to_image(
                    wp.transform.location,
                    world_to_camera,
                    K,
                )

                if result is None:
                    continue

                u, v, depth = result

                # Keep points near the visible image region.
                if (
                    -100
                    <= u
                    <= IMAGE_W + 100
                    and
                    -100
                    <= v
                    <= IMAGE_H + 100
                ):

                    projected_points.append(
                        (u, v)
                    )

            # ------------------------------------------------
            # Compute GT x at the fixed lookahead row
            # ------------------------------------------------

            lane_gt_x = (
                interpolate_x_at_row(
                    projected_points,
                    LOOKAHEAD_ROW,
                )
            )

            # ------------------------------------------------
            # Convert CARLA BGRA -> OpenCV BGR
            # ------------------------------------------------

            img = np.frombuffer(
                image.raw_data,
                dtype=np.uint8,
            )

            img = img.reshape(
                (
                    image.height,
                    image.width,
                    4,
                )
            )

            img = img[:, :, :3].copy()

            # ------------------------------------------------
            # Draw horizontal lookahead row
            # ------------------------------------------------

            cv2.line(
                img,
                (
                    0,
                    LOOKAHEAD_ROW,
                ),
                (
                    IMAGE_W - 1,
                    LOOKAHEAD_ROW,
                ),
                (255, 255, 0),
                2,
            )

            # ------------------------------------------------
            # Draw projected lane-center waypoints
            # ------------------------------------------------

            for u, v in projected_points:

                if (
                    0 <= u < IMAGE_W
                    and
                    0 <= v < IMAGE_H
                ):

                    cv2.circle(
                        img,
                        (
                            int(round(u)),
                            int(round(v)),
                        ),
                        4,
                        (0, 255, 0),
                        -1,
                    )

            # ------------------------------------------------
            # Draw lane GT point
            # ------------------------------------------------

            if lane_gt_x is not None:

                gt_x = int(
                    round(lane_gt_x)
                )

                cv2.circle(
                    img,
                    (
                        gt_x,
                        LOOKAHEAD_ROW,
                    ),
                    12,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    img,
                    f"GT x={lane_gt_x:.1f}",
                    (
                        max(
                            10,
                            gt_x - 70,
                        ),
                        LOOKAHEAD_ROW - 20,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            else:

                cv2.putText(
                    img,
                    "NO GT CROSSING",
                    (
                        30,
                        60,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # ------------------------------------------------
            # Save visualization
            # ------------------------------------------------

            saved += 1

            output_path = os.path.join(
                OUTPUT_DIR,
                f"lane_gt_{saved:02d}.png",
            )

            success = cv2.imwrite(
                output_path,
                img,
            )

            if not success:
                raise RuntimeError(
                    f"Could not save "
                    f"{output_path}"
                )

            print(
                f"[{saved:02d}/{NUM_IMAGES}] "
                f"frame={target_frame} | "
                f"projected_points="
                f"{len(projected_points)} | "
                f"lane_gt_x={lane_gt_x}"
            )

        print()
        print(
            "LANE GT TEST COMPLETE"
        )

        print(
            "Images:",
            OUTPUT_DIR,
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        print("Cleaning up...")

        if camera is not None:

            camera.stop()
            camera.destroy()

        if vehicle is not None:

            vehicle.set_autopilot(
                False,
                TM_PORT,
            )

            vehicle.destroy()

        traffic_manager.set_synchronous_mode(
            False
        )

        world.set_weather(
            original_weather
        )

        world.apply_settings(
            original_settings
        )

        print(
            "CARLA settings restored"
        )


if __name__ == "__main__":
    main()