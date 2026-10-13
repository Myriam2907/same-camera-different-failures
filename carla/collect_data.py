import carla
import csv
import os
import queue
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "127.0.0.1"
PORT = 2000

TM_PORT = 8000

FIXED_DELTA = 0.05          # 20 simulation FPS
NUM_FRAMES = 300            # 15 simulated seconds

IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

WEATHER_NAME = "clear"
SEVERITY = 0

OUTPUT_ROOT = "data/recordings/test_clear"
FRAME_DIR = os.path.join(OUTPUT_ROOT, "frames")
CSV_PATH = os.path.join(OUTPUT_ROOT, "frames.csv")


# ============================================================
# SENSOR SYNCHRONIZATION
# ============================================================

def get_camera_frame(image_queue, target_frame):
    """
    Wait until the camera image corresponding exactly to
    target_frame arrives.

    GPU cameras can arrive later in real time, but their
    image.frame still identifies the simulation frame.
    """

    while True:
        image = image_queue.get(timeout=10.0)

        if image.frame < target_frame:
            continue

        if image.frame == target_frame:
            return image

        raise RuntimeError(
            f"Camera skipped simulation frame {target_frame}. "
            f"Received camera frame {image.frame}."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_ROOT):
        print("Removing previous test recording:", OUTPUT_ROOT)
        shutil.rmtree(OUTPUT_ROOT)

    os.makedirs(FRAME_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Connect to CARLA
    # --------------------------------------------------------

    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    world = client.get_world()

    print("Connected to:", world.get_map().name)

    original_settings = world.get_settings()
    original_weather = world.get_weather()

    traffic_manager = client.get_trafficmanager(TM_PORT)

    vehicle = None
    camera = None

    try:

        # ====================================================
        # 1. SYNCHRONOUS MODE
        # ====================================================

        settings = world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA

        world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)

        print("Synchronous mode: ON")
        print("Fixed delta:", FIXED_DELTA, "seconds")

        # ====================================================
        # 2. CLEAR WEATHER
        # ====================================================

        world.set_weather(
            carla.WeatherParameters.ClearNoon
        )

        print("Weather: clear")
        print("Severity:", SEVERITY)

        # Tick once so the weather/settings are applied.
        world.tick()

        # ====================================================
        # 3. SPAWN EGO VEHICLE
        # ====================================================

        blueprint_library = world.get_blueprint_library()

        vehicle_blueprints = blueprint_library.filter(
            "vehicle.tesla.model3"
        )

        if not vehicle_blueprints:
            raise RuntimeError(
                "vehicle.tesla.model3 blueprint not found"
            )

        vehicle_bp = vehicle_blueprints[0]

        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            raise RuntimeError(
                "Town has no vehicle spawn points"
            )

        vehicle = None
        chosen_spawn_index = None

        # Try spawn points in order until one works.
        for i, spawn_transform in enumerate(spawn_points):
            vehicle = world.try_spawn_actor(
                vehicle_bp,
                spawn_transform
            )

            if vehicle is not None:
                chosen_spawn_index = i
                break

        if vehicle is None:
            raise RuntimeError(
                "Could not spawn ego vehicle at any spawn point"
            )

        print(
            "Vehicle spawned:",
            vehicle.type_id,
            "| spawn index:",
            chosen_spawn_index
        )

        # ====================================================
        # 4. ENABLE AUTOPILOT
        # ====================================================

        vehicle.set_autopilot(
            True,
            TM_PORT
        )

        print("Autopilot: ON")

        # ====================================================
        # 5. FRONT RGB CAMERA
        # ====================================================

        camera_bp = blueprint_library.find(
            "sensor.camera.rgb"
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(IMAGE_W)
        )

        camera_bp.set_attribute(
            "image_size_y",
            str(IMAGE_H)
        )

        camera_bp.set_attribute(
            "fov",
            str(FOV)
        )

        # Capture every simulation tick.
        camera_bp.set_attribute(
            "sensor_tick",
            "0.0"
        )

        camera_transform = carla.Transform(
            carla.Location(
                x=1.5,
                z=1.7
            ),
            carla.Rotation(
                pitch=0.0,
                yaw=0.0,
                roll=0.0
            )
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )

        image_queue = queue.Queue()

        camera.listen(
            image_queue.put
        )

        print("RGB camera attached")
        print(
            f"Camera: {IMAGE_W}x{IMAGE_H}, "
            f"FOV={FOV}"
        )

        # ====================================================
        # 6. WARM UP
        # ====================================================

        print("Warming up sensors...")

        for _ in range(5):
            target_frame = world.tick()
            get_camera_frame(
                image_queue,
                target_frame
            )

        print("Warm-up complete")

        # ====================================================
        # 7. RECORD FRAMES + CSV
        # ====================================================

        with open(
            CSV_PATH,
            "w",
            newline=""
        ) as csv_file:

            writer = csv.writer(csv_file)

            # EXACT Saturday schema.
            writer.writerow([
                "frame",
                "weather",
                "severity",
                "ego_x",
                "ego_y"
            ])

            for saved_index in range(1, NUM_FRAMES + 1):

                # Advance exactly one simulation frame.
                target_frame = world.tick()

                # Receive the RGB frame belonging to that tick.
                image = get_camera_frame(
                    image_queue,
                    target_frame
                )

                # World is paused here until our next tick,
                # so ego location belongs to the same state.
                ego_location = vehicle.get_location()

                filename = f"{target_frame:06d}.png"

                image_path = os.path.join(
                    FRAME_DIR,
                    filename
                )

                image.save_to_disk(
                    image_path
                )

                writer.writerow([
                    target_frame,
                    WEATHER_NAME,
                    SEVERITY,
                    f"{ego_location.x:.6f}",
                    f"{ego_location.y:.6f}"
                ])

                if (
                    saved_index == 1
                    or saved_index % 50 == 0
                    or saved_index == NUM_FRAMES
                ):
                    print(
                        f"Saved {saved_index}/{NUM_FRAMES} "
                        f"| CARLA frame={target_frame} "
                        f"| ego=({ego_location.x:.2f}, "
                        f"{ego_location.y:.2f})"
                    )

        print()
        print("RECORDING COMPLETE")
        print("Frames:", FRAME_DIR)
        print("CSV:", CSV_PATH)

    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        print()
        print("Cleaning up...")

        if camera is not None:
            camera.stop()
            camera.destroy()

        if vehicle is not None:
            vehicle.set_autopilot(
                False,
                TM_PORT
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

        print("CARLA world settings restored")


if __name__ == "__main__":
    main()