import carla
import math
import os
import queue
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "127.0.0.1"
PORT = 2000

TARGET_STOP_ID = 3

IMAGE_W = 1280
IMAGE_H = 720
FOV = 90.0

FIXED_DELTA = 0.05

OUTPUT_DIR = Path("data/screenshots")

# 48, 46, 44, ..., 12, 10 = exactly 20 images
DISTANCES_M = list(range(48, 8, -2))


# ============================================================
# HELPERS
# ============================================================

def angle_difference_deg(a, b):
    """Smallest absolute difference between two yaw angles."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def choose_approach_waypoint(candidates, reference_yaw):
    """
    When CARLA returns multiple predecessor waypoints,
    choose the one whose heading is closest to the
    STOP-controlled lane heading.
    """
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda wp: angle_difference_deg(
            wp.transform.rotation.yaw,
            reference_yaw,
        ),
    )


def get_matching_camera_frame(world, image_queue):
    """
    Advance CARLA by one synchronous tick and return the
    RGB image with exactly the same simulation frame number.
    """
    world_frame = world.tick()

    while True:
        image = image_queue.get(timeout=10.0)

        if image.frame < world_frame:
            continue

        if image.frame == world_frame:
            return image

        raise RuntimeError(
            f"Camera skipped world frame {world_frame}; "
            f"received camera frame {image.frame}"
        )


def find_visual_sign_center(world, stop_actor):
    """
    traffic.stop actor location is near ground/trigger geometry.
    For camera geometry we want the visible sign object itself.

    Match the STOP actor to the nearest TrafficSigns level
    bounding box.
    """
    actor_loc = stop_actor.get_location()

    sign_boxes = world.get_level_bbs(
        carla.CityObjectLabel.TrafficSigns
    )

    if not sign_boxes:
        raise RuntimeError("No visible traffic-sign bounding boxes found")

    def distance_to_actor(bb):
        dx = bb.location.x - actor_loc.x
        dy = bb.location.y - actor_loc.y
        dz = bb.location.z - actor_loc.z

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    visual_bb = min(
        sign_boxes,
        key=distance_to_actor,
    )

    distance = distance_to_actor(visual_bb)

    # In Town10HD_Opt our matching STOP sign is ~2.16 m
    # from its traffic.stop actor location.
    if distance > 5.0:
        raise RuntimeError(
            f"Nearest visible traffic sign is unexpectedly far "
            f"from STOP actor: {distance:.2f} m"
        )

    return visual_bb.location, visual_bb


def check_sign_geometry(vehicle_transform, sign_location):
    """
    Verify that the physical sign is still in front of the car
    and geometrically inside the camera's horizontal FOV.
    """
    vehicle_loc = vehicle_transform.location

    dx = sign_location.x - vehicle_loc.x
    dy = sign_location.y - vehicle_loc.y
    dz = sign_location.z - vehicle_loc.z

    actual_distance = math.sqrt(
        dx * dx + dy * dy + dz * dz
    )

    forward = vehicle_transform.get_forward_vector()
    right = vehicle_transform.get_right_vector()

    forward_distance = (
        dx * forward.x
        + dy * forward.y
        + dz * forward.z
    )

    lateral_distance = (
        dx * right.x
        + dy * right.y
        + dz * right.z
    )

    horizontal_angle = math.degrees(
        math.atan2(
            lateral_distance,
            forward_distance,
        )
    )

    return (
        actual_distance,
        forward_distance,
        horizontal_angle,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(DISTANCES_M) != 20:
        raise RuntimeError(
            f"Expected exactly 20 capture distances, "
            f"got {len(DISTANCES_M)}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Remove only our old generated development screenshots.
    # This ensures the folder does not contain the incorrect
    # 44 -> 6 m set mixed with the corrected set.
    # --------------------------------------------------------

    for path in OUTPUT_DIR.glob("stop_[0-9][0-9]_*.png"):
        path.unlink()

    for path in OUTPUT_DIR.glob("test_stop*.png"):
        path.unlink()

    client = carla.Client(
        HOST,
        PORT,
    )

    client.set_timeout(10.0)

    world = client.get_world()
    carla_map = world.get_map()

    print("Connected to:", carla_map.name)

    original_settings = world.get_settings()
    original_weather = world.get_weather()

    vehicle = None
    camera = None

    try:

        # ====================================================
        # SYNCHRONOUS MODE
        # ====================================================

        settings = world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA

        world.apply_settings(settings)

        # Development screenshots should be clear-weather
        # reference images for Luan's initial detector.
        world.set_weather(
            carla.WeatherParameters.ClearNoon
        )

        # ====================================================
        # FIND TARGET STOP ACTOR
        # ====================================================

        target_stop = None

        for stop in world.get_actors().filter(
            "*traffic.stop*"
        ):
            if stop.id == TARGET_STOP_ID:
                target_stop = stop
                break

        if target_stop is None:
            raise RuntimeError(
                f"STOP actor {TARGET_STOP_ID} not found"
            )

        stop_transform = target_stop.get_transform()

        print()
        print("STOP actor:", target_stop.id)
        print(
            "Actor location:",
            f"({stop_transform.location.x:.2f}, "
            f"{stop_transform.location.y:.2f}, "
            f"{stop_transform.location.z:.2f})",
        )

        # ====================================================
        # FIND VISIBLE RED STOP SIGN
        # ====================================================

        visual_sign_location, visual_bb = (
            find_visual_sign_center(
                world,
                target_stop,
            )
        )

        print(
            "Visible sign center:",
            f"({visual_sign_location.x:.2f}, "
            f"{visual_sign_location.y:.2f}, "
            f"{visual_sign_location.z:.2f})",
        )

        # ====================================================
        # FIND THE CONTROLLED DRIVING LANE
        # ====================================================

        trigger_world = stop_transform.transform(
            target_stop.trigger_volume.location
        )

        trigger_wp = carla_map.get_waypoint(
            trigger_world,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if trigger_wp is None:
            raise RuntimeError(
                "Could not find STOP-controlled driving lane"
            )

        reference_yaw = (
            trigger_wp.transform.rotation.yaw
        )

        print(
            "Controlled lane:",
            f"road={trigger_wp.road_id}",
            f"lane={trigger_wp.lane_id}",
            f"yaw={reference_yaw:.2f}",
        )

        # ====================================================
        # FIND FARTHEST APPROACH POSITION
        # ====================================================

        first_candidates = trigger_wp.previous(
            float(DISTANCES_M[0])
        )

        first_wp = choose_approach_waypoint(
            first_candidates,
            reference_yaw,
        )

        if first_wp is None:
            raise RuntimeError(
                "Could not find initial approach waypoint"
            )

        # ====================================================
        # SPAWN EGO VEHICLE
        # ====================================================

        bp_lib = world.get_blueprint_library()

        vehicle_candidates = bp_lib.filter(
            "vehicle.tesla.model3"
        )

        if not vehicle_candidates:
            raise RuntimeError(
                "Tesla Model 3 blueprint not found"
            )

        vehicle_bp = vehicle_candidates[0]

        spawn_transform = carla.Transform(
            carla.Location(
                x=first_wp.transform.location.x,
                y=first_wp.transform.location.y,
                z=first_wp.transform.location.z + 0.4,
            ),
            first_wp.transform.rotation,
        )

        vehicle = world.try_spawn_actor(
            vehicle_bp,
            spawn_transform,
        )

        if vehicle is None:
            raise RuntimeError(
                "Could not spawn ego vehicle"
            )

        # We teleport the vehicle between development
        # positions. Physics is intentionally disabled.
        vehicle.set_simulate_physics(False)

        print("Vehicle spawned:", vehicle.type_id)

        # ====================================================
        # FRONT RGB CAMERA
        # ====================================================

        camera_bp = bp_lib.find(
            "sensor.camera.rgb"
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
            attachment_type=carla.AttachmentType.Rigid,
        )

        image_queue = queue.Queue()

        camera.listen(
            image_queue.put
        )

        print("RGB camera attached")
        print()

        # ====================================================
        # CAPTURE 20 DEVELOPMENT IMAGES
        # ====================================================

        saved = 0

        for index, requested_distance in enumerate(
            DISTANCES_M,
            start=1,
        ):

            candidates = trigger_wp.previous(
                float(requested_distance)
            )

            wp = choose_approach_waypoint(
                candidates,
                reference_yaw,
            )

            if wp is None:
                raise RuntimeError(
                    f"No waypoint found for "
                    f"{requested_distance} m"
                )

            vehicle_transform = carla.Transform(
                carla.Location(
                    x=wp.transform.location.x,
                    y=wp.transform.location.y,
                    z=wp.transform.location.z + 0.4,
                ),
                wp.transform.rotation,
            )

            # ------------------------------------------------
            # Verify the physical sign is still ahead
            # ------------------------------------------------

            (
                actual_sign_distance,
                forward_distance,
                horizontal_angle,
            ) = check_sign_geometry(
                vehicle_transform,
                visual_sign_location,
            )

            if forward_distance <= 0.0:
                raise RuntimeError(
                    f"At requested distance "
                    f"{requested_distance} m, "
                    f"the physical STOP sign is behind "
                    f"the vehicle."
                )

            if abs(horizontal_angle) >= FOV / 2.0:
                raise RuntimeError(
                    f"At requested distance "
                    f"{requested_distance} m, "
                    f"STOP sign is outside camera FOV "
                    f"({horizontal_angle:.2f} deg)."
                )

            vehicle.set_transform(
                vehicle_transform
            )

            # Clear frames produced before this teleport.
            while not image_queue.empty():
                try:
                    image_queue.get_nowait()
                except queue.Empty:
                    break

            # Give camera/renderer two ticks to settle.
            get_matching_camera_frame(
                world,
                image_queue,
            )

            get_matching_camera_frame(
                world,
                image_queue,
            )

            # Capture third synchronized frame.
            image = get_matching_camera_frame(
                world,
                image_queue,
            )

            filename = (
                f"stop_{index:02d}_"
                f"{requested_distance:02d}m.png"
            )

            output_path = (
                OUTPUT_DIR / filename
            )

            image.save_to_disk(
                str(output_path)
            )

            saved += 1

            print(
                f"[{index:02d}/20] "
                f"upstream={requested_distance:02d} m | "
                f"physical_sign_distance="
                f"{actual_sign_distance:.2f} m | "
                f"bearing={horizontal_angle:.2f} deg | "
                f"saved={output_path}"
            )

        print()
        print("CAPTURE COMPLETE")
        print("Images saved:", saved)

        if saved != 20:
            raise RuntimeError(
                f"Expected 20 images, saved {saved}"
            )

    finally:

        print("Cleaning up...")

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

        print("CARLA settings restored")


if __name__ == "__main__":
    main()