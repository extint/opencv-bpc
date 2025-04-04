import os
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image  # For getting image dimensions

def prepare_train_pbr(train_pbr_path, output_path, obj_id):
    """
    Prepare the train_pbr dataset for YOLO, incorporating multiple camera inputs.
    Creates 'images' and 'labels' folders under output_path.
    """
    # Define the different cameras and modalities to process
    cameras = ["cam1", "cam2", "cam3"]
    modalities = ["rgb", "depth", "aolp", "dolp"]

    # Mapping from cameras to ground-truth JSON files
    camera_gt_map = {f"rgb_{cam}": f"scene_gt_{cam}.json" for cam in cameras}
    camera_gt_info_map = {f"rgb_{cam}": f"scene_gt_info_{cam}.json" for cam in cameras}

    # Ensure output directories exist
    images_dir = os.path.join(output_path, "images")
    labels_dir = os.path.join(output_path, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Iterate over each scene (e.g. 000000, 000001, ...)
    scene_folders = sorted([
        d for d in os.listdir(train_pbr_path)
        if os.path.isdir(os.path.join(train_pbr_path, d)) and not d.startswith(".")
    ])

    for scene_folder in tqdm(scene_folders, desc="Processing train_pbr scenes"):
        scene_path = os.path.join(train_pbr_path, scene_folder)

        for cam in cameras:
            rgb_path = os.path.join(scene_path, f"rgb_{cam}")
            depth_path = os.path.join(scene_path, f"depth_{cam}")
            aolp_path = os.path.join(scene_path, f"aolp_{cam}")
            dolp_path = os.path.join(scene_path, f"dolp_{cam}")

            scene_gt_file = os.path.join(scene_path, camera_gt_map[f"rgb_{cam}"])
            scene_gt_info_file = os.path.join(scene_path, camera_gt_info_map[f"rgb_{cam}"])

            # Ensure required files exist
            if not all(os.path.exists(p) for p in [rgb_path, depth_path, aolp_path, dolp_path, scene_gt_file, scene_gt_info_file]):
                print(f"Skipping {scene_folder}, missing files for {cam}.")
                continue

            # Load the ground truth and metadata
            with open(scene_gt_file, "r") as f:
                scene_gt_data = json.load(f)
            with open(scene_gt_info_file, "r") as f:
                scene_gt_info_data = json.load(f)

            # Process each image in the scene
            num_imgs = len(scene_gt_data)
            for img_id in range(num_imgs):
                img_key = str(img_id)
                img_file = os.path.join(rgb_path, f"{img_id:06d}.png")
                depth_file = os.path.join(depth_path, f"{img_id:06d}.png")
                aolp_file = os.path.join(aolp_path, f"{img_id:06d}.png")
                dolp_file = os.path.join(dolp_path, f"{img_id:06d}.png")
                print(depth_file)
                if not all(os.path.exists(p) for p in [img_file, depth_file, aolp_file, dolp_file]):
                    print(f"skipped")
                    continue

                if img_key not in scene_gt_data or img_key not in scene_gt_info_data:
                    continue

                # Extract bounding boxes for the target object
                valid_bboxes = []
                for bbox_info, gt_info in zip(scene_gt_info_data[img_key], scene_gt_data[img_key]):
                    if gt_info["obj_id"] == obj_id and bbox_info["visib_fract"] > 0:
                        valid_bboxes.append(bbox_info["bbox_obj"])  # (x, y, w, h)

                if not valid_bboxes:
                    continue

                # Save multi-modal image as .npy (RGB + Depth + AOLP + DOLP)
                out_img_name = f"{scene_folder}_{cam}_{img_id:06d}.npy"
                out_img_path = os.path.join(images_dir, out_img_name)

                rgb_img = np.array(Image.open(img_file))
                depth_img = np.array(Image.open(depth_file))
                aolp_img = np.array(Image.open(aolp_file))
                dolp_img = np.array(Image.open(dolp_file))

                print(f"RGB Image Shape: {rgb_img.shape}")  # Should be (H, W, 3)
                print(f"Depth Image Shape: {depth_img.shape}")  # Should be (H, W)
                print(f"AoLP Image Shape: {aolp_img.shape}")  # Should be (H, W)
                print(f"DoLP Image Shape: {dolp_img.shape}")  # Should be (H, W)
                multimodal_img = np.stack([rgb_img[..., 0], rgb_img[..., 1], rgb_img[..., 2], depth_img, aolp_img, dolp_img], axis=-1)

                np.save(out_img_path, multimodal_img)

                # Save YOLO format labels
                with Image.open(img_file) as img:
                    img_width, img_height = img.size

                out_label_name = f"{scene_folder}_{cam}_{img_id:06d}.txt"
                out_label_path = os.path.join(labels_dir, out_label_name)
                with open(out_label_path, "w") as lf:
                    for (x, y, w, h) in valid_bboxes:
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        width = w / img_width
                        height = h / img_height
                        lf.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def generate_yaml(output_path, obj_id):
    """
    Generate a YOLO .yaml file for training/validation.
    """
    yolo_configs_dir = os.path.join(os.getcwd(), "yolo", "configs")
    os.makedirs(yolo_configs_dir, exist_ok=True)

    images_dir = os.path.join(output_path, "images")
    train_path = os.path.abspath(images_dir)
    val_path = os.path.abspath(images_dir)

    yaml_path = os.path.join(yolo_configs_dir, f"data_obj_{obj_id}.yaml")

    yaml_content = {
        "train": train_path,
        "val": val_path,
        "nc": 1,
        "names": [f"object_{obj_id}"]
    }

    with open(yaml_path, "w") as f:
        for key, value in yaml_content.items():
            f.write(f"{key}: {value}\n")

    print(f"[INFO] Generated YAML file at: {yaml_path}\n")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare the train_pbr dataset for YOLO training with multiple cameras and modalities.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the train_pbr dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for YOLO dataset.")
    parser.add_argument("--obj_id", type=int, required=True, help="Object ID to filter for.")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path
    obj_id = args.obj_id

    prepare_train_pbr(dataset_path, output_path, obj_id)
    generate_yaml(output_path, obj_id)

    print("[INFO] Dataset preparation complete!")


if __name__ == "__main__":
    main()
