import math
import torch
import torch.nn as nn

def wrapped_euler_diff(a, b):
    """
    Minimal angular difference in each dimension.
    E.g. diff among (a-b), (a+2π-b), (a-2π-b).
    Returns shape (B,D).
    """
    diff1 = torch.abs(a - b)
    diff2 = torch.abs((a + 2 * math.pi) - b)
    diff3 = torch.abs((a - 2 * math.pi) - b)
    diff_min = torch.minimum(diff1, diff2)
    diff_min = torch.minimum(diff_min, diff3)
    return diff_min


class EulerAnglePoseLoss(nn.Module):
    def __init__(self, w_rot=1.0, w_center=1.0):
        super().__init__()
        self.w_rot = w_rot
        self.w_center = w_center

    def forward(self, labels_gt, preds, sym_list=None):
        keypoint_preds, pose_preds = preds  # Unpack the tuple

        # Extract ground truth and predicted values correctly
        angles_gt = labels_gt[:, :3]  # Ground truth rotation angles
        translations_gt = labels_gt[:, 3:]  # Ground truth translations

        angles_pr = pose_preds[:, :3]  # Predicted rotation angles
        translations_pr = pose_preds[:, 3:]  # Predicted translations

        # Compute rotation loss
        rotation_loss = self._wrapped_euler_loss(angles_pr, angles_gt)

        # Compute center loss (assuming 2D center points)
        center_gt = labels_gt[:, -2:]  # Extract last two values as 2D center
        center_pr = pose_preds[:, -2:]  # Same for predictions

        center_dist = torch.norm(center_gt - center_pr, dim=1)
        center_loss = center_dist.mean()

        # Compute total loss
        total_loss = self.w_rot * rotation_loss + self.w_center * center_loss

        # Compute additional metrics
        rot_deg_mean = rotation_loss * (180.0 / math.pi)
        center_px_mean = center_dist.mean()

        rad_thresh = 5.0 * (math.pi / 180.0)  # 5-degree threshold in radians
        acc_mask = (rotation_loss < rad_thresh) & (center_dist < 5.0)
        acc_5deg_5px = acc_mask.float().mean()

        metrics = {
            "total_loss": total_loss.detach(),
            "rot_loss": rotation_loss.detach(),
            "center_loss": center_loss.detach(),
            "rot_deg_mean": rot_deg_mean.detach(),
            "center_px_mean": center_px_mean.detach(),
            "acc_5deg_5px": acc_5deg_5px.detach(),
        }

        return total_loss, metrics

    def _wrapped_euler_loss(self, pred, gt):
        diff1 = torch.abs(pred - gt)
        diff2 = torch.abs(pred + 2 * math.pi - gt)
        diff3 = torch.abs(pred - 2 * math.pi - gt)
        diff_min = torch.minimum(diff1, diff2)
        diff_min = torch.minimum(diff_min, diff3)
        per_sample = torch.mean(diff_min, dim=1)
        return per_sample.mean()
