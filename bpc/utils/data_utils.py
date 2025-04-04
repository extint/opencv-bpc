import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import json
import glob
import cv2
import math
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T  # for color jitter, etc.
from bpc.inference.utils.camera_utils import load_camera_params


def compute_2d_center(K, R, t):
    """
    Projects the 3D point 't' (the object translation) into 2D via K.
    Returns (u, v) or None if behind camera.
    """
    if t[2, 0] <= 0:
        return None
    uv = K @ t
    if uv[2, 0] == 0:
        return None
    uv /= uv[2, 0]
    return uv[0, 0], uv[1, 0]


def letterbox_preserving_aspect_ratio(img, target_size=256, fill_color=(255, 255, 255)):
    h, w = img.shape[:2]
    scale = float(target_size) / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), fill_color, dtype=np.uint8)

    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas, scale, dx, dy


def matrix_to_euler_xyz(R):
    """
    Convert a 3x3 rotation matrix to Euler angles (XYZ).
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        # Fallback
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return x, y, z

class BOPSingleObjRGBDDataset(Dataset):
    """
    Reads BOP data for a single obj_id from multiple scenes & cameras.
    Then splits 80/20 for train/val *per scene/cam*.

    Returns (rgbd_t, label_5d, meta).
    rgbd_t => concatenated RGB and Depth tensors.
    label_5d => [Rx, Ry, Rz, cx, cy].
    'augment' => random shift/scale bounding box + optional color jitter if training.
    'split' => "train" or "val" to pick that portion from each scene/cam.
    """

    def __init__(
        self,
        root_dir,
        scene_ids,
        cam_ids,
        target_obj_id,
        target_size=256,
        augment=False,
        split="train",
        max_per_scene=None,
        train_ratio=0.8,
        depth_scale=1000.0,  # depth scale factor (typically 1000 for mm)
        seed=42
    ):
        super().__init__()
        self.root_dir = root_dir
        self.scene_ids = scene_ids
        self.cam_ids = cam_ids
        self.obj_id = target_obj_id
        self.target_size = target_size
        self.augment = augment
        self.split = split.lower()  # "train" or "val"
        self.max_per_scene = max_per_scene
        self.train_ratio = train_ratio
        self.depth_scale = depth_scale
        self.samples = []

        random.seed(seed)  # for reproducibility

        # gather all samples in a temp list
        all_samples = []

        # Loop over each scene
        train_pbr_path = os.path.join(root_dir, "")
        for sid in scene_ids:
            scene_path = os.path.join(train_pbr_path, sid)
            scene_count = 0

            for cam_id in cam_ids:
                info_file = os.path.join(scene_path, f"scene_gt_info_{cam_id}.json")
                pose_file = os.path.join(scene_path, f"scene_gt_{cam_id}.json")
                cam_file = os.path.join(scene_path, f"scene_camera_{cam_id}.json")
                rgb_dir = os.path.join(scene_path, f"rgb_{cam_id}")
                depth_dir = os.path.join(scene_path, f"depth_{cam_id}")

                # Skip if RGB or depth directories are missing
                if not all(os.path.exists(f) for f in [info_file, pose_file, cam_file, rgb_dir, depth_dir]):
                    continue

                with open(info_file, "r") as f1, open(pose_file, "r") as f2, open(cam_file, "r") as f3:
                    info_json = json.load(f1)
                    pose_json = json.load(f2)
                    cam_json = json.load(f3)

                all_im_ids = sorted(info_json.keys(), key=lambda x: int(x))
                for im_id_s in all_im_ids:
                    im_id = int(im_id_s)
                    if im_id_s not in cam_json:
                        continue

                    K = np.array(cam_json[im_id_s]["cam_K"], dtype=np.float32).reshape(3, 3)
                    img_name = f"{im_id:06d}.png"
                    rgb_path = os.path.join(rgb_dir, img_name)
                    depth_path = os.path.join(depth_dir, img_name)
                    
                    # Skip if either RGB or depth image is missing
                    if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                        continue

                    # each image can contain multiple objects, pick the ones that match self.obj_id
                    for inf, pos in zip(info_json[im_id_s], pose_json[im_id_s]):
                        if pos["obj_id"] != self.obj_id:
                            continue
                        x, y, w_, h_ = inf["bbox_visib"]
                        if w_ <= 0 or h_ <= 0:
                            continue

                        R = np.array(pos["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                        t = np.array(pos["cam_t_m2c"], dtype=np.float32).reshape(3, 1)

                        all_samples.append({
                            "scene_id": sid,
                            "cam_id": cam_id,
                            "im_id": im_id,
                            "rgb_path": rgb_path,
                            "depth_path": depth_path,
                            "K": K,
                            "R": R,
                            "t": t,
                            "bbox_visib": [x, y, w_, h_]
                        })

                scene_count += 1
                if self.max_per_scene is not None and scene_count >= self.max_per_scene:
                    break

        # Split 80/20 (train/val) per scene/cam
        from collections import defaultdict
        groups = defaultdict(list)
        for s in all_samples:
            key = (s["scene_id"], s["cam_id"])
            groups[key].append(s)

        # Shuffle each group and split
        for key, group_samples in groups.items():
            random.shuffle(group_samples)
            n_total = len(group_samples)
            n_train = int(round(self.train_ratio * n_total))

            if self.split == "train":
                selected = group_samples[:n_train]
            else:  # "val"
                selected = group_samples[n_train:]

            self.samples.extend(selected)

        print(f"[INFO] BOPSingleObjRGBDDataset(split={self.split}, augment={self.augment}): total={len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        rgb_path = data["rgb_path"]
        depth_path = data["depth_path"]
        
        # Read RGB image
        bgr = cv2.imread(rgb_path)
        if bgr is None:
            raise IOError(f"Cannot read RGB image {rgb_path}")
        
        # Read depth image (usually 16-bit single channel)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"Cannot read depth image {depth_path}")

        K = data["K"]
        R = data["R"]
        t = data["t"]
        x, y, w, h = map(int, data["bbox_visib"])

        H_img, W_img = bgr.shape[:2]

        # BBox augment if self.augment and self.split=="train"
        if self.augment and self.split == "train":
            scale_factor = 1.0 + 0.2 * random.random()  # up to +20%
            new_w = int(round(w * scale_factor))
            new_h = int(round(h * scale_factor))

            max_shift_x = int(0.1 * w)
            max_shift_y = int(0.1 * h)
            shift_x = random.randint(-max_shift_x, max_shift_x)
            shift_y = random.randint(-max_shift_y, max_shift_y)

            x0 = x - shift_x
            y0 = y - shift_y
            x0 = max(0, min(x0, W_img - 1))
            y0 = max(0, min(y0, H_img - 1))
            new_w = min(new_w, W_img - x0)
            new_h = min(new_h, H_img - y0)
        else:
            x0, y0 = x, y
            new_w, new_h = w, h

        # Crop both RGB and depth images
        rgb_crop = bgr[y0:y0+new_h, x0:x0+new_w]
        depth_crop = depth[y0:y0+new_h, x0:x0+new_w]
        
        if rgb_crop.size == 0 or depth_crop.size == 0:
            raise RuntimeError("Empty crop => skip")

        # Letterbox and preserve aspect ratio for both RGB and depth
        rgb_letter, scale, dx, dy = letterbox_preserving_aspect_ratio(
            rgb_crop, target_size=self.target_size
        )
        
        # Use the same scaling and padding for depth to ensure alignment
        if len(depth_crop.shape) == 2:  # Single channel
            depth_crop = depth_crop[:, :, np.newaxis]  # Add channel dimension
        
        depth_letter = np.zeros((self.target_size, self.target_size, 1), dtype=depth_crop.dtype)
        h, w = depth_crop.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        depth_resized = cv2.resize(depth_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        if len(depth_resized.shape) == 2:  # Single channel after resize
            depth_resized = depth_resized[:, :, np.newaxis]  # Add channel dimension
            
        # Place the depth image with the same padding as RGB
        x_offset, y_offset = int(dx), int(dy)
        depth_letter[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = depth_resized

        # Compute 2D center
        uv = compute_2d_center(K, R, t)
        if uv is None:
            raise RuntimeError("Behind camera => skip")
        u_full, v_full = uv
        local_u = (u_full - x0)
        local_v = (v_full - y0)
        letter_x = local_u * scale + dx
        letter_y = local_v * scale + dy

        # Convert R => Euler
        Rx, Ry, Rz = matrix_to_euler_xyz(R)
        label_5d = np.array([Rx, Ry, Rz, letter_x, letter_y], dtype=np.float32)

        # Convert RGB to tensor
        rgb_letter_c = np.ascontiguousarray(rgb_letter, dtype=np.uint8)
        rgb_t = torch.from_numpy(rgb_letter_c).permute(2, 0, 1).float() / 255.0
        
        # Apply color jitter to RGB if self.augment and split="train"
        if self.augment and self.split == "train":
            jitter_transform = T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
            img_pil = TF.to_pil_image(rgb_t)
            img_pil = jitter_transform(img_pil)
            rgb_t = TF.to_tensor(img_pil)

        # Normalize RGB (ImageNet style)
        rgb_t = TF.normalize(rgb_t, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        
        # Convert depth to tensor and normalize
        # Depth is usually in millimeters, so normalize by depth_scale
        depth_letter_c = np.ascontiguousarray(depth_letter, dtype=np.float32)
        depth_t = torch.from_numpy(depth_letter_c).permute(2, 0, 1).float() / self.depth_scale
        
        # Clip depth to a reasonable range (e.g., 0-10 meters)
        depth_t = torch.clamp(depth_t, 0.0, 10.0)

        # Standardize depth (zero mean, unit variance) - optional but often helpful
        # Comment this out if you prefer raw depth values
        if torch.sum(depth_t > 0) > 0:  # Only normalize non-zero values
            valid_mask = depth_t > 0
            valid_depth = depth_t[valid_mask]
            mean_depth = valid_depth.mean()
            std_depth = valid_depth.std()
            if std_depth > 0:
                depth_t = (depth_t - mean_depth) / std_depth
                depth_t = torch.where(valid_mask, depth_t, torch.zeros_like(depth_t))

        # Concatenate RGB and depth to form RGBD tensor
        rgbd_t = torch.cat([rgb_t, depth_t], dim=0)  # Shape: [4, 256, 256]

        lbl_t = torch.from_numpy(label_5d)
        meta = {
            "scene_id": data["scene_id"],
            "cam_id": data["cam_id"],
            "im_id": data["im_id"]
        }
        
        # rgbd_t -> 4,256,256 (RGB + Depth)
        return rgbd_t, lbl_t, meta


def bop_collate_fn(batch):
    imgs, labels, metas = [], [], []
    for (img, lbl, meta) in batch:
        imgs.append(img)
        labels.append(lbl)
        metas.append(meta)
    imgs_t = torch.stack(imgs, dim=0)
    labels_t = torch.stack(labels, dim=0)
    return imgs_t, labels_t, metas



def render_mask(mesh, K, camera_pose, imsize, mesh_poses):
    """
    Render the mesh with the given camera and mesh pose on the full image,
    with a directional light from the camera's POV.
    """
    K = K.copy()
    camera_pose = camera_pose.copy()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.background_color = np.array([0.0, 0.0, 0.0]) # Set background to black

    for mesh_pose in mesh_poses:
        scene.add(mesh, pose=mesh_pose)

    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], zfar=10000)
    camera_pose[1, :] = -camera_pose[1, :]
    camera_pose[2, :] = -camera_pose[2, :]
    camera_pose = np.linalg.inv(camera_pose)
    scene.add(camera, pose=camera_pose)

    # Add directional light from camera POV
    light_direction = np.array([0, 0, -1])  # Direction light points -Z axis in camera space.
    # Transform light direction to world space.
    light_direction_world = camera_pose.copy()
    light_direction_world[:3, :3] = light_direction_world[:3, :3] @ light_direction
    #The position of the light. We can put it at the camera's position.
    light = pyrender.DirectionalLight(
        color=np.array([1.0, 0, 1.0]), intensity=5)
    scene.add(light, pose=light_direction_world) #Light is at the origin in world space.

    renderer = pyrender.OffscreenRenderer(*imsize)
    color, depth = renderer.render(scene)
    return color, depth


def load_gt_poses(scene_dir, scene_id, cam_ids, image_id, obj_id):
    """Load all GT 6D poses and bounding boxes from scene_gt_camX.json and scene_gt_info_camX.json."""
    gt_poses = []
    scene_path = os.path.join(scene_dir, scene_id)

    for cam_id in cam_ids[:1]:
        # TOTHINK why only cam1 is considered instead of all 3 cams
        print(f"Camera {cam_id} was called")
        gt_path = os.path.join(scene_path, f"scene_gt_{cam_id}.json")
        info_path = os.path.join(scene_path, f"scene_gt_info_{cam_id}.json")

        if not os.path.exists(gt_path) or not os.path.exists(info_path):
            print(f"Missing GT files for {cam_id}")
            continue

        with open(gt_path, "r") as f:
            gt_data = json.load(f)

        with open(info_path, "r") as f:
            info_data = json.load(f)

        img_key = str(image_id)
        if img_key not in gt_data or img_key not in info_data:
            print(f"Image {image_id} not found in {cam_id}")
            continue

        objects = gt_data[img_key]
        bboxes = info_data[img_key]

        for obj, bbox in zip(objects, bboxes):
            print(obj["obj_id"])
            if obj["obj_id"] != obj_id:
                print(obj["obj_id"] ," skipped")
                continue  # Skip objects that don't match the specified obj_id

            rotation_matrix = np.array(obj["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
            translation = np.array(obj["cam_t_m2c"], dtype=np.float32)  # [X, Y, Z]
            gt_poses.append(calc_pose_matrix(rotation_matrix, translation))
    print("gt_poses\n",gt_poses)
    return gt_poses

def calc_pose_matrix(R, t):
    pose = np.eye(4) #identity matrix of 4x4
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose

class Capture:
    def __init__(self, images, Ks, RTs, obj_id, gt_poses=None):
        self.images = images
        self.Ks = Ks
        self.RTs = RTs
        print(gt_poses)
        if gt_poses:
            self.gt_poses = np.linalg.inv(RTs[0]) @ gt_poses
        
    @classmethod
    def from_dir(cls, scene_dir, cam_ids, image_id, obj_id):
        cam_params = load_camera_params(scene_dir, cam_ids)
        Ks = [cam_params[x]['K'][image_id] for x in cam_ids]
        Rs = [cam_params[x]['R'][image_id] for x in cam_ids]
        Ts = [cam_params[x]['t'][image_id] for x in cam_ids]
        RTs = [calc_pose_matrix(r, t) for r, t in zip(Rs, Ts)]
        image_paths = [glob.glob(os.path.join(scene_dir, f"rgb_{cam_id}", f"{image_id:06d}.*g"))[0] for cam_id in cam_ids]
        print(image_paths)
        images = [cv2.imread(x) for x in image_paths] # cv2 mentioned
        print("no.of images in Capture: ",len(images))
        gt_poses = load_gt_poses(scene_dir, '', cam_ids, image_id, obj_id)
        return cls(images, Ks, RTs, obj_id, gt_poses)
