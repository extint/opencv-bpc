import os
from ultralytics import YOLO
import torch
import argparse

def train_yolo11(task, data_path, obj_id, epochs, imgsz, batch):
    """
    Train YOLO11 for a specific task ("detection" or "segmentation")
    using Ultralytics YOLO with multi-modal (5-channel) input.

    Args:
        task (str): "detection" or "segmentation"
        data_path (str): Path to the YOLO .yaml file (e.g. data_obj_11.yaml).
        obj_id (int): The BOP object ID (e.g. 11).
        epochs (int): Number of training epochs.
        imgsz (int): Image size used for training.
        batch (int): Batch size.

    Returns:
        final_model_path (str): Path where the trained model is saved.
    """

    # Automatically select device: MPS (Apple), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Pick the pre-trained model file based on task
    if task == "detection":
        pretrained_weights = "yolo11n.pt"
        task_suffix = "detection"
    elif task == "segmentation":
        pretrained_weights = "yolo11n-seg.pt"
        task_suffix = "segmentation"
    else:
        print("Invalid task. Must be 'detection' or 'segmentation'.")
        return None

    # Validate dataset YAML file existence
    if not os.path.exists(data_path):
        print(f"Error: Dataset YAML file not found at {data_path}")
        return None

    # Load the YOLO model
    print(f"Loading model {pretrained_weights} for {task_suffix} ...")
    model = YOLO(pretrained_weights)

    # Train the model
    print(f"Training YOLO11 for {task_suffix} on object {obj_id} using {device}...")
    model.train(
        data=data_path,   # Dataset YAML path
        epochs=epochs,     # Training epochs
        imgsz=imgsz,       # Image size
        batch=batch,       # Batch size
        device=device,     # Auto-detect device
        workers=2,         # Number of workers
        save=True,         # Save results
    )

    # ----------------------------------------------------------------------------
    # Save trained model in: yolo/models/<detection or segmentation>/obj_<obj_id>/
    # ----------------------------------------------------------------------------
    save_dir = os.path.join("yolo", "models", task_suffix, f"obj_{obj_id}")
    os.makedirs(save_dir, exist_ok=True)

    model_name = f"yolo11-{task_suffix}-obj_{obj_id}.pt"
    final_model_path = os.path.join(save_dir, model_name)

    # Save final model
    model.save(final_model_path)

    print(f"Model saved as: {final_model_path}")
    return final_model_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 on a specific dataset and object.")
    parser.add_argument("--obj_id", type=int, required=True, help="Object ID for training (e.g., 11).")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the dataset YAML file (e.g. yolo/configs/data_obj_11.yaml).")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--task", type=str, choices=["detection", "segmentation"], default="detection",
                        help="Task type (detection or segmentation).")

    args = parser.parse_args()

    train_yolo11(
        task=args.task,
        data_path=args.data_path,
        obj_id=args.obj_id,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )


if __name__ == "__main__":
    main()
