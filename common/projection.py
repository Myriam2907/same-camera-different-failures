import math
import numpy as np


def build_camera_matrix(width, height, fov_deg):
    """
    Build the 3x3 pinhole-camera intrinsic matrix.
    """

    focal = width / (
        2.0 * math.tan(
            math.radians(fov_deg) / 2.0
        )
    )

    K = np.identity(3)

    K[0, 0] = focal
    K[1, 1] = focal

    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0

    return K


def world_to_image(location, world_to_camera, K):
    """
    Project a CARLA world point into image coordinates.

    Returns:
        (u, v, depth)

    depth <= 0 means the point is behind the camera.
    """

    point = np.array([
        location.x,
        location.y,
        location.z,
        1.0,
    ])

    camera_point = np.dot(
        world_to_camera,
        point,
    )

    # CARLA / Unreal:
    # x = forward
    # y = right
    # z = up
    #
    # Camera projection coordinates:
    # x = right
    # y = down
    # z = forward
    camera_point = np.array([
        camera_point[1],
        -camera_point[2],
        camera_point[0],
    ])

    depth = camera_point[2]

    if depth <= 0:
        return None

    image_point = np.dot(
        K,
        camera_point,
    )

    u = image_point[0] / image_point[2]
    v = image_point[1] / image_point[2]

    return float(u), float(v), float(depth)


def interpolate_x_at_row(points, target_y):
    """
    Find x where the projected lane-center polyline
    crosses target_y.

    points:
        [(u1, v1), (u2, v2), ...]
        ordered from near -> far.

    Returns:
        x coordinate, or None if no crossing exists.
    """

    for i in range(len(points) - 1):

        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        # Does target_y lie between these two points?
        if (
            min(y1, y2)
            <= target_y
            <= max(y1, y2)
        ):

            if abs(y2 - y1) < 1e-6:
                return (x1 + x2) / 2.0

            alpha = (
                target_y - y1
            ) / (
                y2 - y1
            )

            x = x1 + alpha * (x2 - x1)

            return float(x)

    return None
