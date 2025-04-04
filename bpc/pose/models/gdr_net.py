import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class RegionAttentionModule(nn.Module):
    """
    Region Attention Module (RAM) - Enhances feature extraction.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.attention = nn.Conv2d(256, 1, kernel_size=1)  # Output attention map

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        attention_map = torch.sigmoid(self.attention(x))  # (B,1,H,W)
        return x * attention_map  # Element-wise multiplication

class PixelwiseVotingModule(nn.Module):
    """
    Dense Pixel-wise Voting for keypoint offsets.
    """
    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_keypoints * 2, kernel_size=1)  # Predict (dx, dy) per keypoint

    def forward(self, x):
        offsets = self.conv(x)  # Shape: (B, num_keypoints*2, H, W)
        return offsets

class PoseRefinementModule(nn.Module):
    """
    Pose Refinement Network - refines initial pose prediction.
    """
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1024)  # Increased size for better representation
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 5)  # Output: [Rx, Ry, Rz, cx, cy]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class GDRNet_RGBD(nn.Module):
    """
    GDR-Net for 6D Object Pose Estimation with RGBD input.
    """
    def __init__(self, num_keypoints=8, pretrained=True):
        super().__init__()
        
        # Backbone: RGB Stream - ResNet-50
        rgb_resnet = models.resnet50(weights=(models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None))
        # Modify first conv layer to accept 3 channels
        self.rgb_backbone = nn.Sequential(*list(rgb_resnet.children())[:-2])  # Remove FC layer
        
        # Backbone: Depth Stream - ResNet-50
        depth_resnet = models.resnet50(weights=(models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None))
        # Modify first conv layer to accept 1 channel
        depth_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_backbone = nn.Sequential(*list(depth_resnet.children())[:-2])  # Remove FC layer
        
        # Feature fusion layer (2048 + 2048 = 4096)
        self.fusion = nn.Conv2d(4096, 2048, kernel_size=1)
        
        # Region Attention Module (RAM)
        self.ram = RegionAttentionModule(in_channels=2048)
        
        # Pixel-wise Voting Module
        self.voting = PixelwiseVotingModule(in_channels=256, num_keypoints=num_keypoints)
        
        # Global Feature Aggregation (Pooling for pose regression)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Pose Refinement Module (increased input features for better depth utilization)
        self.pose_regressor = PoseRefinementModule(in_features=256)
    
    def forward(self, x):
        """
        Input: RGBD image tensor (B, 4, H, W)
        Outputs:
        - Keypoint offsets (B, num_keypoints*2, H, W)
        - Pose prediction (B, 5) [Rotation (3) + Center (2)]
        """
        # Split input into RGB and depth
        rgb = x[:, :3]  # First 3 channels (RGB)
        depth = x[:, 3:4]  # Last channel (Depth)
        
        # Process RGB and depth streams separately
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)        
        # Fuse features
        fused_features = torch.cat([rgb_features, depth_features], dim=1)  # channel-wise concatenation
        fused_features = self.fusion(fused_features)
        
        # Continue with existing pipeline
        attn_features = self.ram(fused_features)  # Apply region attention
        keypoint_offsets = self.voting(attn_features)  # Dense pixel-wise voting
        pooled_features = self.global_pool(attn_features).flatten(start_dim=1)  # (B, 256)
        pose = self.pose_regressor(pooled_features)  # Final pose output
        
        return keypoint_offsets, pose