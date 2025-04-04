import numpy as np

def triangulate_multi_view(proj_mats, points):
    """Triangulate using all cameras via DLT."""
    A = []
    for P, (x, y) in zip(proj_mats, points):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]  # Solution is last row of V^T
    return X[:3] / X[3]

def compute_reprojection_error(P, X, point_2d):
    """Reproject 3D point and compute pixel error."""
    proj = P @ np.append(X, 1.0)
    proj /= proj[2]
    return np.linalg.norm(proj[:2] - point_2d)

def compute_final_pose(final_pose_array, triangulated_points):
    """
    Combine average rotations and triangulated translations into a final 6D pose.

    Args:
        final_pose_array (np.ndarray): Shape (N, 3, 5), with [Rx, Ry, Rz, cx, cy] per camera.
        triangulated_points (np.ndarray): Shape (N, 3), with [X, Y, Z] translations.

    Returns:
        np.ndarray: Final 6D pose array of shape (N, 6) with [Rx_avg, Ry_avg, Rz_avg, X, Y, Z].
    """
    num_objects = final_pose_array.shape[0]
    final_6d_pose = np.zeros((num_objects, 6), dtype=np.float32)

    for i in range(num_objects):
        # Average rotations (Rx, Ry, Rz)
        avg_rotation = final_pose_array[i, :, :3].mean(axis=0)

        # Translation from triangulated points (X, Y, Z)
        translation = triangulated_points[i]

        # Combine
        final_6d_pose[i, :3] = avg_rotation
        final_6d_pose[i, 3:] = translation

    return final_6d_pose
