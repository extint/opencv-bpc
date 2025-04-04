import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_pose_estimation(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=10,
    out_dir="checkpoints",
    device=torch.device("cuda"),
    resume=False,
):
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))

    best_val_rot = float("inf")
    start_epoch = 1
    best_model_path = os.path.join(out_dir, "best_model.pth")
    checkpoint_path = os.path.join(out_dir, "last_checkpoint.pth")

    # If resume is requested, check if checkpoint exists
    if resume and os.path.exists(checkpoint_path):
        print(f"[INFO] Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_rot = checkpoint["best_val_rot"]

    for epoch in range(start_epoch, epochs + 1):
        # TRAIN
        model.train()
        train_sums = {"total_loss": 0, "rot_loss": 0, "center_loss": 0, 
                      "rotDeg": 0, "centerPx": 0, "acc": 0}
        train_steps = 0

        tbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs}", ncols=120)
        for step, (imgs, lbls, metas) in enumerate(tbar):
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            # Just a placeholder for symmetrical objects if needed
            sym_list = [[torch.eye(3).numpy()] for _ in metas]

            optimizer.zero_grad()
            preds = model(imgs)
            # loss, metrics = criterion(lbls, preds, sym_list)
            keypoint_preds, pose_preds = preds  # Unpack the tuple
            loss, metrics = criterion(lbls, (keypoint_preds, pose_preds), sym_list)

            loss.backward()
            optimizer.step()

            train_steps += 1
            train_sums["total_loss"] += metrics["total_loss"].item()
            train_sums["rot_loss"] += metrics["rot_loss"].item()
            train_sums["center_loss"] += metrics["center_loss"].item()
            train_sums["rotDeg"] += metrics["rot_deg_mean"].item()
            train_sums["centerPx"] += metrics["center_px_mean"].item()
            train_sums["acc"] += metrics["acc_5deg_5px"].item()

            tbar.set_postfix({
                "tot": f"{metrics['total_loss'].item():.3f}",
                "rot": f"{metrics['rot_loss'].item():.3f}",
                "ctr": f"{metrics['center_loss'].item():.3f}",
                "deg": f"{metrics['rot_deg_mean'].item():.1f}",
                "px": f"{metrics['center_px_mean'].item():.1f}",
                "acc": f"{metrics['acc_5deg_5px'].item():.2f}",
            })

        for k in train_sums:
            train_sums[k] /= train_steps

        writer.add_scalar("train/total_loss", train_sums["total_loss"], epoch)
        writer.add_scalar("train/rot_loss", train_sums["rot_loss"], epoch)
        writer.add_scalar("train/center_loss", train_sums["center_loss"], epoch)
        writer.add_scalar("train/rot_deg", train_sums["rotDeg"], epoch)
        writer.add_scalar("train/center_px", train_sums["centerPx"], epoch)
        writer.add_scalar("train/acc_5deg_5px", train_sums["acc"], epoch)

        # VAL
        model.eval()
        val_sums = {"total_loss": 0, "rot_loss": 0, "center_loss": 0, 
                    "rotDeg": 0, "centerPx": 0, "acc": 0}
        val_steps = 0
        with torch.no_grad():
            for imgs, lbls, metas in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                sym_list = [[torch.eye(3).numpy()] for _ in metas]

                loss, metrics = criterion(lbls, model(imgs), sym_list)
                val_steps += 1
                val_sums["total_loss"] += metrics["total_loss"].item()
                val_sums["rot_loss"] += metrics["rot_loss"].item()
                val_sums["center_loss"] += metrics["center_loss"].item()
                val_sums["rotDeg"] += metrics["rot_deg_mean"].item()
                val_sums["centerPx"] += metrics["center_px_mean"].item()
                val_sums["acc"] += metrics["acc_5deg_5px"].item()

        for k in val_sums:
            val_sums[k] /= val_steps

        writer.add_scalar("val/total_loss", val_sums["total_loss"], epoch)
        writer.add_scalar("val/rot_loss", val_sums["rot_loss"], epoch)
        writer.add_scalar("val/center_loss", val_sums["center_loss"], epoch)
        writer.add_scalar("val/rot_deg", val_sums["rotDeg"], epoch)
        writer.add_scalar("val/center_px", val_sums["centerPx"], epoch)
        writer.add_scalar("val/acc_5deg_5px", val_sums["acc"], epoch)

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"TRAIN tot={train_sums['total_loss']:.3f}, rot={train_sums['rot_loss']:.3f}, ctr={train_sums['center_loss']:.3f}, "
            f"deg={train_sums['rotDeg']:.1f}, px={train_sums['centerPx']:.1f}, acc={train_sums['acc']:.2f} | "
            f"VAL tot={val_sums['total_loss']:.3f}, rot={val_sums['rot_loss']:.3f}, ctr={val_sums['center_loss']:.3f}, "
            f"deg={val_sums['rotDeg']:.1f}, px={val_sums['centerPx']:.1f}, acc={val_sums['acc']:.2f}"
        )

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_rot": best_val_rot,
        }
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if val_sums["rotDeg"] < best_val_rot:
            best_val_rot = val_sums["rotDeg"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  >> Saved best model with val_rotDeg={best_val_rot:.2f}")

    writer.close()
    print("DONE. best model =>", best_model_path)
    return model
