import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
import os
import sys

# # The directory containing 'pose' is one level up:
# FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.dirname(FILE_DIR)

# # Add this parent directory to the path
# sys.path.append(PARENT_DIR)

# Now 'utils' is visible as a direct import
from bpc.utils.data_utils import BOPSingleObjDataset, bop_collate_fn
from bpc.pose.models.simple_pose_net import SimplePoseNet
from bpc.pose.models.losses import EulerAnglePoseLoss
from bpc.pose.trainers.trainer import train_pose_estimation
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pose Estimation Model")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Path to dataset root directory (with train_pbr)")
    parser.add_argument("--target_obj_id", type=int, default=11,
                        help="Target object ID")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                        help="Base directory for checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the last checkpoint")
    return parser.parse_args()


def find_scenes(root_dir):
    """
    Return a sorted list of all numeric scene folder names
    under root_dir/train_pbr, e.g. ["000000", "000001", ...].
    """
    train_pbr_dir = os.path.join(root_dir, "train_pbr")
    if not os.path.exists(train_pbr_dir):
        raise FileNotFoundError(f"{train_pbr_dir} does not exist")

    all_items = os.listdir(train_pbr_dir)
    scene_ids = [item for item in all_items if item.isdigit()]
    scene_ids.sort()
    return scene_ids


def main():
    args = parse_args()

    # Find all scene folders
    scene_ids = find_scenes(args.root_dir)
    print(f"[INFO] Found scene_ids={scene_ids}")

    # Construct a simpler checkpoint path for object ID only (no single scene)
    obj_id = args.target_obj_id
    checkpoint_dir = os.path.join(args.checkpoints_dir, f"obj_{obj_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Prepare dataset: train (no augment), train (augment), val
    train_ds_fixed = BOPSingleObjDataset(
        root_dir=args.root_dir,
        scene_ids=scene_ids,
        cam_ids=["cam1", "cam2", "cam3"],
        target_obj_id=args.target_obj_id,
        target_size=256,
        augment=False,
        split="train"
    )
    train_ds_aug = BOPSingleObjDataset(
        root_dir=args.root_dir,
        scene_ids=scene_ids,
        cam_ids=["cam1", "cam2", "cam3"],
        target_obj_id=args.target_obj_id,
        target_size=256,
        augment=True,
        split="train"
    )
    val_ds = BOPSingleObjDataset(
        root_dir=args.root_dir,
        scene_ids=scene_ids,
        cam_ids=["cam1", "cam2", "cam3"],
        target_obj_id=args.target_obj_id,
        target_size=256,
        augment=False,
        split="val"
    )

    # Print a quick summary so you see the train vs val sizes
    print(f"[INFO] train_ds_fixed:  {len(train_ds_fixed)} samples")
    print(f"[INFO] train_ds_aug:    {len(train_ds_aug)} samples")
    print(f"[INFO] val_ds:          {len(val_ds)} samples")

    # Concat the two train sets
    train_dataset = ConcatDataset([train_ds_fixed, train_ds_aug])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=bop_collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=bop_collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePoseNet(pretrained=True).to(device) # TODO FIX RESUMEING

    # Load checkpoint if resuming
    # checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    # checkpoint_path = 'best_model.pth' # TODO UNDO THIS
    # if args.resume and os.path.exists(checkpoint_path):
    # print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    # checkpoint = torch.load(checkpoint_path)
    # print(checkpoint.keys())

    # model.load_state_dict(checkpoint)

    # Initialize criterion and optimizer
    criterion = EulerAnglePoseLoss(w_rot=1.0, w_center=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Train the model
    train_pose_estimation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        out_dir=checkpoint_dir,
        device=device,
        resume=args.resume
    )


if __name__ == "__main__":
    main()

"""
python pose/train.py \
  --root_dir datasets/ipd_bop_data_jan25_1 \
  --target_obj_id 11 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3 \
  --num_workers 16 \
  --checkpoints_dir /home/exouser/Desktop/idp_codebase/pose/checkpoints
"""