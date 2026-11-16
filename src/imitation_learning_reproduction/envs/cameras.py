from __future__ import annotations

import numpy as np


def camera_fovy(model, cam_id: int) -> float:
    """Field of view in degrees, as stored by MuJoCo."""
    return float(model.cam_fovy[cam_id])


def camera_pose(data, cam_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (position, rotation) of the camera in world coordinates.

    ``data.cam_xmat`` is a row-major 3x3 rotation whose columns are the camera's
    x, y, z axes in world coordinates. The camera looks along -z.
    """
    pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64).copy()
    mat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3).copy()
    return pos, mat


def backproject(
    depth: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    fovy_deg: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Turn a depth image into a world-frame point cloud.

    Args:
        depth: (H, W) metres, camera z-depth (distance along the optical axis).
        cam_pos: (3,) world position of the camera.
        cam_mat: (3, 3) camera-to-world rotation (columns are camera axes).
        fovy_deg: vertical field of view in degrees.
        width, height: image dimensions.

    Returns:
        (H, W, 3) world-frame points.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.shape != (height, width):
        raise ValueError(f"depth shape {depth.shape} != {(height, width)}")

    fy = height / (2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0))
    fx = fy
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0

    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    z = depth
    x_cam = (u - cx) / fx * z
    y_cam = -(v - cy) / fy * z
    z_cam = -z

    cam_points = np.stack([x_cam, y_cam, z_cam], axis=-1)
    return cam_points @ cam_mat.T + cam_pos

