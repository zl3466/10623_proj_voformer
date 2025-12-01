"""
DINOv2 vision processing wrappers for visual odometry tasks.
Contains only DINOv2-related components for image feature extraction.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from PIL import Image
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class DINOv2VisionEncoder(nn.Module):
    """
    DINOv2-based vision encoder for extracting image features.
    This replaces the built-in vision processing of Qwen2.5-VL models.
    """
    
    def __init__(self, model_name: str = "facebook/dinov2-base", hidden_size: int = 768):
        super().__init__()
        
        # Check if we're in distributed training mode
        # If using DDP, don't use device_map (let Trainer handle device placement)
        import torch.distributed as dist
        use_device_map = not (dist.is_available() and dist.is_initialized())
        
        # Load DINOv2 model
        self.dinov2 = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if use_device_map else None,
            trust_remote_code=True
        )
        
        # Freeze DINOv2 model - don't train its parameters
        for param in self.dinov2.parameters():
            param.requires_grad = False
        
        # Set DINOv2 to eval mode to ensure no dropout/batch norm updates
        self.dinov2.eval()
        
        # Get DINOv2's hidden size
        self.dinov2_hidden_size = self.dinov2.config.hidden_size
        
        # Projection layer to match Qwen's hidden size (this will be trained)
        self.projection = nn.Linear(self.dinov2_hidden_size, hidden_size)
        
        # Initialize projection layer with small weights to prevent gradient explosion
        # Use Xavier uniform initialization (similar to transformer initialization)
        nn.init.xavier_uniform_(self.projection.weight, gain=0.02)
        nn.init.zeros_(self.projection.bias)
        
        # Ensure projection layer uses bfloat16 to match Qwen model dtype
        self.projection = self.projection.to(dtype=torch.bfloat16)
        
        logger.info(f"DINOv2 Vision Encoder initialized with hidden size: {hidden_size}")
        logger.info(f"DINOv2 model: {model_name}")
        logger.info(f"Projection layer: trainable, dtype=bfloat16, initialized with Xavier uniform (gain=0.02)")
    
    def extract_features(self, images) -> torch.Tensor:
        """
        Extract DINOv2 features from images.
        
        Args:
            images: Tensor of shape [batch_size, num_images, channels, height, width]
            
        Returns:
            torch.Tensor: DINOv2 features of shape [batch_size, sequence_length, hidden_size]
        """

        # Process images with DINOv2
        # Handle tensor input: [batch_size, num_images, channels, height, width]
        if isinstance(images, torch.Tensor):
            batch_size, num_images, channels, height, width = images.shape
            
            # Reshape to [batch_size * num_images, channels, height, width] for DINOv2 processing
            batch_images = images.view(batch_size * num_images, channels, height, width)
            batch_images = batch_images.to(self.dinov2.device)
            
            # Extract features using DINOv2 (frozen, so use no_grad)
            with torch.no_grad():
                outputs = self.dinov2(batch_images)
            
            # Get patch features (excluding CLS token)
            patch_features = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token
            
            # Convert DINOv2 features to bfloat16 before projection (to match projection layer dtype)
            patch_features = patch_features.to(dtype=torch.bfloat16)
            
            # Project to target hidden size (trainable, so outside no_grad)
            projected_features = self.projection(patch_features)
            
            # Reshape back to [batch_size, num_images, n_patches, hidden_size]
            n_patches = projected_features.shape[1]
            
            # Log actual patches per image (may differ from theoretical calculation)
            # patches_per_image = n_patches // num_images if num_images > 0 else n_patches
            # print(f"Vision encoder: {num_images} images × {patches_per_image} patches/image = {n_patches} total patches")
            hidden_size = projected_features.shape[2]
            projected_features = projected_features.view(batch_size, num_images, n_patches, hidden_size)
            
            # Flatten multiple images into a single sequence
            # [batch_size, num_images, n_patches, hidden_size] -> [batch_size, num_images * n_patches, hidden_size]
            projected_features = projected_features.view(batch_size, num_images * n_patches, hidden_size)
            # print(f"converted projected_features shape: {projected_features.shape}")
            
        else:
            raise ValueError(f"Expected tensor input, got {type(images)}")
        
        return projected_features
