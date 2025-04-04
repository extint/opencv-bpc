import os
import json
import numpy as np


def load_camera_params(scene_dir, cam_ids):
    """Load camera intrinsics and extrinsics."""
    params = {}
    
    for cid in cam_ids:
        with open(os.path.join(scene_dir, f"scene_camera_{cid}.json")) as f:
            data = json.load(f)
        params[cid] = {'K': {}, 'R': {}, 't': {}}
        for im_id_str, vals in data.items():
            im_id = int(im_id_str)
            params[cid]['K'][im_id] = np.array(vals['cam_K'], dtype=np.float32).reshape(3, 3)
            params[cid]['R'][im_id] = np.array(vals['cam_R_w2c'], dtype=np.float32).reshape(3, 3)
            params[cid]['t'][im_id] = np.array(vals['cam_t_w2c'], dtype=np.float32).flatten()
    print(f"Loading camera parameters from: {os.path.join(scene_dir, f'scene_camera_{cid}.json')}")
    return params


def compute_fundamental_matrix(K1, R1, t1, K2, R2, t2):
    """Compute the fundamental matrix between two cameras."""
    t1 = t1.flatten()
    t2 = t2.flatten()
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    # Skew-symmetric matrix
    tx = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ], dtype=np.float32)

    E = tx @ R_rel
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    F = (K2_inv.T @ E @ K1_inv)

    # Normalize
    if abs(F[2, 2]) > 1e-8:
        F /= F[2, 2]

    return F
    

