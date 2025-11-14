"""
ConvNeXt Backbone Wrapper for YOLOv5 Integration

This module provides a wrapper around timm's ConvNeXt to make it compatible 
with YOLOv5's multi-scale feature extraction requirements.
"""

import os
import platform
import sys
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.convnext import convnext_tiny


class ConvNeXtBackbone(nn.Module):
    """
    Wrapper for ConvNeXt to extract multi-scale features for YOLO.
    
    Extracts intermediate features at different spatial resolutions to feed
    into YOLO's neck (FPN/PAN). Default outputs are at 1/8, 1/16, 1/32 scales.
    """
    
    def __init__(self, variant='convnext_tiny', pretrained=True, out_stages=(1, 2, 3)):
        """
        Args:
            variant: Model size ('convnext_tiny' supported)
            pretrained: Load ImageNet pretrained weights
            out_stages: Stage indices to extract (0-3 for ConvNeXt's 4 stages)
        """
        super().__init__()
        self.out_stages = out_stages
        
        # Load full model
        if variant == 'convnext_tiny':
            full_model = convnext_tiny(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        
        # Extract feature extraction components (remove classifier)
        self.stem = full_model.stem
        self.stages = full_model.stages
        
        # ConvNeXt-Tiny has dims [96, 192, 384, 768] for stages 0-3
        stage_channels = [96, 192, 384, 768]
        self.out_channels = [stage_channels[i] for i in out_stages]
        
    def forward(self, x):
        """
        Extract multi-scale features compatible with YOLO neck.
        
        Args:
            x: Input (B, 3, H, W)
            
        Returns:
            List of feature tensors or single tensor depending on YOLO parsing
        """
        features = []
        x = self.stem(x)
        
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_stages:
                # Ensure NCHW format expected by YOLO
                if x.dim() == 3:  # If (B, N, C) from ViT-style
                    B, N, C = x.shape
                    H = W = int(N**0.5)
                    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
                features.append(x)
        
        # Return features in order expected by neck
        return features if len(features) > 1 else features[0]