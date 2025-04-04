import os
import json
import math
import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms.functional as TF

from ultralytics import YOLO
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from pose.models.simple_pose_net import SimplePoseNet
from utils.data_utils import letterbox_preserving_aspect_ratio, calc_pose_matrix
from inference.utils.camera_utils import load_camera_params, compute_fundamental_matrix
from inference.epipolar_matching import compute_cost_matrix, match_objects, triangulate_multi_view
from inference.yolo_detection import detect_with_yolo

from dataclasses import dataclass



def load_pose_model(pose_model_path, device='cuda:0'):
    pose_model = SimplePoseNet(pretrained=False)
    pose_model.load_state_dict(torch.load(pose_model_path, map_location=device))
    pose_model.to(device).eval()
    return pose_model

class PosePrediction:
    def __init__(self, detections, capture):
        self.boxes = np.array([x['bbox'] for x in detections])
        self.centroids = np.array([x['bb_center'] for x in detections])
        self.capture = capture
        self.t = self.triangulate()
        
    def triangulate(self):
        proj_mats = []
        for idx in range(len(self.boxes)):
            K = self.capture.Ks[idx]
            RT = self.capture.RTs[idx]
            P = K @ RT[:3]
            proj_mats.append(P)

        X = triangulate_multi_view(proj_mats, self.centroids)
        return X
        
@dataclass
class PoseEstimatorParams:
    yolo_model_path: str = "yolo11-detection-obj11.pt"
    pose_model_path: str = "best_model.pth"
    matching_threshold: int = 30
    yolo_conf_thresh: float = 0.1

class PoseEstimator:
    def __init__(self, params):
        self.params = params
        self.yolo = YOLO(params.yolo_model_path).cuda()
        self.pose_model = load_pose_model(params.pose_model_path)

    def _detect(self, capture):
        camera_predictions = {}
        for idx, image in enumerate(capture.images):
            detections = self.yolo(image, imgsz=1280)[0]
            boxes = detections.boxes.xyxy.cpu().numpy()
            confs = detections.boxes.conf.cpu().numpy()
            clss  = detections.boxes.cls.cpu().numpy()
            if len(detections.boxes)==0:
                camera_predictions[cam_id] = []
                continue
            # Keep only class=0 with conf>=0.5
            valid = (clss==0) & (confs>=self.params.yolo_conf_thresh)
            boxes = boxes[valid]
            preds_cam = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cx = 0.5*(x1 + x2)
                cy = 0.5*(y1 + y2)

                preds_cam.append({
                    'bbox': (x1, y1, x2, y2),
                    'bb_center': (cx, cy),
                })

            camera_predictions[idx] = preds_cam
        return camera_predictions
           
    def _match(self, capture, detections):
        predictions = []
        det_cam1 = detections[0]
        det_cam2 = detections[1]
        det_cam3 = detections[2]
        K1, K2, K3 = capture.Ks
        R1, R2, R3 = [x[:3, :3] for x in capture.RTs]
        t1, t2, t3 = [x[:3, 3] for x in capture.RTs]
        F12 = compute_fundamental_matrix(K1, R1, t1, K2, R2, t2)
        F13 = compute_fundamental_matrix(K1, R1, t1, K3, R3, t3)
        F23 = compute_fundamental_matrix(K2, R2, t2, K3, R3, t3)


        if len(det_cam1)==0 or len(det_cam2)==0 or len(det_cam3)==0:
            print("\nAt least one camera has zero detections => no matching.")
            return predictions
        cost_matrix = compute_cost_matrix(det_cam1, det_cam2, det_cam3, F12, F13, F23)

        # Show cost matrix stats
        print("\n--- Cost Matrix Stats ---")
        print(f"Shape: {cost_matrix.shape}")
        print(f"Min: {cost_matrix.min():.4f}, Max: {cost_matrix.max():.4f}, Mean: {cost_matrix.mean():.4f}")

        # Random samples
        N, M, P = cost_matrix.shape
        sample_count = min(5, N*M*P)
        print("\nRandom samples from cost_matrix:")
        for _ in range(sample_count):
            i = np.random.randint(0, N)
            j = np.random.randint(0, M)
            k = np.random.randint(0, P)
            val = cost_matrix[i,j,k]
            print(f"  cost_matrix[{i},{j},{k}] = {val:.4f}")

        # Hungarian + threshold
        matches = match_objects(cost_matrix, threshold=self.params.matching_threshold)
        matches_sorted = sorted(matches, key=lambda t: cost_matrix[t[0], t[1], t[2]])

        for i, j, k in matches_sorted:
            detections = [det_cam1[i], det_cam2[j], det_cam3[k]]    
            predictions.append(PosePrediction(detections, capture))

        return predictions

    def _estimate_rotation(self, predictions):
        for p in predictions:
            rotation_preds = []
            for idx in range(len(p.centroids)):
                x1, y1, x2, y2 = p.boxes[idx]
                img = p.capture.images[idx]
                crop = img[y1:y2, x1:x2]
                letter_img, scale, dx, dy = letterbox_preserving_aspect_ratio(
                    crop, 256, (128,128,128)
                )
                letter_img_rgb = cv2.cvtColor(letter_img, cv2.COLOR_BGR2RGB)
                tens = TF.to_tensor(letter_img_rgb)
                tens = TF.normalize(tens, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                tens = tens.unsqueeze(0).to('cuda')
                with torch.no_grad():
                    pred_5d = self.pose_model(tens)[0].cpu().numpy()
                    print(pred_5d)
                    rotation_preds.append(p.capture.RTs[idx][:3, :3].T @ Rotation.from_euler('xyz', pred_5d[:3]).as_matrix())
            p.rotation_preds = rotation_preds
            p.final_rotation = rotation_preds[2]
            p.pose = calc_pose_matrix(p.final_rotation, p.t)

