import carla
import queue
import os

HOST = "127.0.0.1"
PORT = 2000
TARGET_STOP_ID = 3

IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

OUTPUT_DIR = "data/screenshots"

client = carla.Client(HOST, PORT)
client.set_timeout(10.0)

world = client.get_world()
carla_map = world.get_map()

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Connected to:", carla_map.name)

# --------------------------------------------------
# Find target STOP sign
# --------------------------------------------------
target_stop = None

for stop in world.get_actors().filter("*traffic.stop*"):
    if stop.id == TARGET_STOP_ID:
        target_stop = stop
        break

if target_stop is None:
    raise RuntimeError("STOP sign not found")

sign_transform = target_stop.get_transform()
sign_loc = sign_transform.location

print("STOP sign:", target_stop.id)

# --------------------------------------------------
# Find the lane controlled by this STOP sign
# --------------------------------------------------
trigger_loc = sign_transform.transform(
    target_stop.trigger_volume.location
)

trigger_wp = carla_map.get_waypoint(
    trigger_loc,
    project_to_road=True,
    lane_type=carla.LaneType.Driving,
)

if trigger_wp is None:
    raise RuntimeError("Controlled lane not found")

print(
    "Controlled lane:",
    trigger_wp.road_id,
    trigger_wp.lane_id,
    "yaw:",
    round(trigger_wp.transform.rotation.yaw, 2),
)

# --------------------------------------------------
# Distances to capture
# 44, 42, ..., 6 m = 20 images
# --------------------------------------------------
distances = list(range(44, 4, -2))

# --------------------------------------------------
# Spawn vehicle at farthest position
# --------------------------------------------------
first_candidates = trigger_wp.previous(float(distances[0]))

if not first_candidates:
    raise RuntimeError("Could not find starting waypoint")

spawn_wp = first_candidates[0]

bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

spawn_transform = spawn_wp.transform
spawn_transform.location.z += 0.4

vehicle = world.try_spawn_actor(
    vehicle_bp,
    spawn_transform,
)

if vehicle is None:
    raise RuntimeError("Could not spawn vehicle")

# Prevent physics from moving the car after teleporting
vehicle.set_simulate_physics(False)

print("Vehicle spawned")

# --------------------------------------------------
# RGB camera
# --------------------------------------------------
camera_bp = bp_lib.find("sensor.camera.rgb")

camera_bp.set_attribute("image_size_x", str(IMAGE_W))
camera_bp.set_attribute("image_size_y", str(IMAGE_H))
camera_bp.set_attribute("fov", str(FOV))

camera_transform = carla.Transform(
    carla.Location(x=1.5, z=1.7)
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle,
)

image_queue = queue.Queue()
camera.listen(image_queue.put)

# --------------------------------------------------
# Capture images
# --------------------------------------------------
saved = 0

for index, distance in enumerate(distances, start=1):

    candidates = trigger_wp.previous(float(distance))

    if not candidates:
        print("No waypoint for", distance, "m")
        continue

    # Prefer same road/lane as the STOP-controlled lane
    same_lane = [
        wp for wp in candidates
        if wp.road_id == trigger_wp.road_id
        and wp.lane_id == trigger_wp.lane_id
    ]

    if same_lane:
        wp = same_lane[0]
    else:
        wp = candidates[0]

    transform = wp.transform
    transform.location.z += 0.4

    vehicle.set_transform(transform)

    # Remove old camera frames
    while not image_queue.empty():
        try:
            image_queue.get_nowait()
        except queue.Empty:
            break

    # Let camera update after teleport
    image = None

    for _ in range(3):
        world.tick()
        image = image_queue.get(timeout=5.0)

    filename = os.path.join(
        OUTPUT_DIR,
        f"stop_{index:02d}_{distance:02d}m.png"
    )

    image.save_to_disk(filename)

    print(
        f"[{index:02d}/20] "
        f"distance={distance:02d} m -> {filename}"
    )

    saved += 1

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
camera.stop()
camera.destroy()
vehicle.destroy()

print()
print("DONE")
print("Images saved:", saved)