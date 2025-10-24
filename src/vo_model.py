import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
import math

class VOModel(nn.Module):
    """Simplified wrapper for Qwen2.5-VL-3B using standard forward pass"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", vocab_size: int = 1000, config=None):
        super().__init__()
        
        # Store config for reference
        self.user_config = config

        # Load Qwen2.5-VL-3B model - use AutoModel for vision-language models
        from transformers import AutoModel
        self.base_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Add pose prediction head for your custom pose tokens
        hidden_size = self.base_model.config.hidden_size
        self.pose_head = nn.Linear(hidden_size, vocab_size)
        
        # Ensure pose_head uses the same dtype as the base model
        self.pose_head = self.pose_head.to(dtype=self.base_model.dtype)

        # The model already has a language modeling head, so we don't need a custom one
        # We can use the standard forward pass and just adjust the loss computation if needed

    def forward(self, pixel_values=None, input_ids=None, labels=None, attention_mask=None, **kwargs):
        """
        Forward pass for pose prediction using Qwen2.5-VL base model.
        Uses custom pose tokenization instead of text-based approach.
        """
        
        # Debug: print all inputs
        print(f"Model inputs:")
        print(f"  pixel_values shape: {pixel_values.shape if pixel_values is not None else None}")
        print(f"  input_ids shape: {input_ids.shape if input_ids is not None else None}")
        print(f"  labels shape: {labels.shape if labels is not None else None}")
        print(f"  attention_mask shape: {attention_mask.shape if attention_mask is not None else None}\n")
        
        # Calculate image grid dimensions for Qwen2.5-VL
        if pixel_values is not None:
            batch_size = pixel_values.shape[0]
            
            if len(pixel_values.shape) == 3:  # [batch_size, seq_len, hidden_dim]
                seq_len = pixel_values.shape[1]
                
                # For multiple images, we need to calculate grid dimensions per image
                # Get number of images from config
                num_images = self.user_config['data']['num_input_frames']
                patches_per_image = seq_len // num_images
                
                # Calculate grid size per image (assuming square patches)
                grid_size_per_image = int(math.sqrt(patches_per_image))
                if grid_size_per_image * grid_size_per_image != patches_per_image:
                    raise ValueError(f"Expected squared patches per image, got {patches_per_image}")
                
                # Create image_grid_thw: [num_images, 3] where each row is [temporal, height, width]
                # For images: temporal=1, height=width=grid_size_per_image
                image_grid_thw = torch.tensor([[1, grid_size_per_image, grid_size_per_image] for _ in range(num_images)], 
                                             dtype=torch.long, device=pixel_values.device)
                
                print(f"Number of images: {num_images}")
                print(f"Patches per image: {patches_per_image}")
                print(f"Grid size per image: {grid_size_per_image}")
            else:
                raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")
        else:
            image_grid_thw = None
        print(f"Image grid THW: {image_grid_thw}\n")

        print(f"Pixel values: {pixel_values}\n")
        
        # Fuse image tokens with pose tokens in forward()
        if pixel_values is not None and input_ids is not None:
            batch_size = pixel_values.shape[0]
            
            # Calculate number of image tokens from the grid
            # Qwen2.5-VL uses spatial merging, so we need to account for spatial_merge_size
            if image_grid_thw is not None:
                # Get spatial_merge_size from the model config
                spatial_merge_size = self.base_model.config.vision_config.spatial_merge_size
                
                # Calculate total image tokens across all images
                # Each image contributes: (H * W) // spatial_merge_size^2 tokens
                tokens_per_image = (image_grid_thw[0, 1] * image_grid_thw[0, 2]) // (spatial_merge_size ** 2)
                num_image_tokens = tokens_per_image * image_grid_thw.shape[0]  # Multiply by number of images
                
                print(f"Spatial merge size: {spatial_merge_size}")
                print(f"Tokens per image: {tokens_per_image}")
                print(f"Number of images: {image_grid_thw.shape[0]}")
                print(f"Total image tokens: {num_image_tokens}")
            else:
                num_image_tokens = 0
            
            # Create image token IDs - use the correct Qwen2.5-VL image token
            image_token_id = self.base_model.config.image_token_id  # Qwen2.5-VL image token ID
            image_tokens = torch.full((batch_size, num_image_tokens), image_token_id, 
                                    dtype=torch.long, device=pixel_values.device)
            
            # Debug: check tensor dimensions
            # print(f"Image tokens shape: {image_tokens.shape}")
            # print(f"Input IDs shape: {input_ids.shape}")
            
            # Ensure input_ids has the same number of dimensions as image_tokens
            if input_ids.dim() > image_tokens.dim():
                # If input_ids has more dimensions, squeeze the extra ones
                while input_ids.dim() > image_tokens.dim():
                    input_ids = input_ids.squeeze(0)
            elif input_ids.dim() < image_tokens.dim():
                # If input_ids has fewer dimensions, add them
                while input_ids.dim() < image_tokens.dim():
                    input_ids = input_ids.unsqueeze(0)
            
            # print(f"Adjusted input_ids shape: {input_ids.shape}")
            
            # Combine image tokens with pose tokens: [image_tokens, pose_tokens]
            combined_input_ids = torch.cat([image_tokens, input_ids], dim=1)
            
            # Update attention mask to include image tokens
            image_attention_mask = torch.ones((batch_size, num_image_tokens), 
                                             dtype=torch.long, device=pixel_values.device)
            
            # Ensure attention_mask has the same number of dimensions as image_attention_mask
            if attention_mask.dim() > image_attention_mask.dim():
                while attention_mask.dim() > image_attention_mask.dim():
                    attention_mask = attention_mask.squeeze(0)
            elif attention_mask.dim() < image_attention_mask.dim():
                while attention_mask.dim() < image_attention_mask.dim():
                    attention_mask = attention_mask.unsqueeze(0)
            
            combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
            
            print(f"Combined input_ids shape: {combined_input_ids.shape}")
            print(f"Number of image tokens: {num_image_tokens}")
            print(f"Number of pose tokens: {input_ids.shape[1]}")
            
        else:
            combined_input_ids = input_ids
            combined_attention_mask = attention_mask
        
        # Use the Qwen2.5-VL base model forward pass
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True
        )
        
        # Get the last hidden states
        hidden_states = outputs.last_hidden_state
        
        # Apply pose prediction head to all hidden states
        pose_logits = self.pose_head(hidden_states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Labels only contain pose tokens, so we need to extract only the pose token predictions
            # The input_ids structure is: [image_tokens, pose_tokens]
            # We need to extract only the pose token predictions for loss computation
            
            # Use the same calculation as we did for creating image tokens
            if image_grid_thw is not None:
                spatial_merge_size = self.base_model.config.vision_config.spatial_merge_size
                num_image_tokens = (image_grid_thw[0, 1] * image_grid_thw[0, 2]) // (spatial_merge_size ** 2)
            else:
                num_image_tokens = 0
            
            # print(f"Loss computation - num_image_tokens: {num_image_tokens}")
            # print(f"Loss computation - hidden_states shape: {hidden_states.shape}")
            # print(f"Loss computation - labels shape: {labels.shape}")
            # print(f"Loss computation - labels content: {labels}")
            
            # Extract only the pose token predictions (skip image tokens)
            # hidden_states shape: [batch_size, total_tokens, hidden_dim]
            # We need to extract the last target_tokens from the sequence
            # Input: [image_tokens, input_pose_tokens, placeholder_tokens] (324 + 12 + 24 = 360)
            # Target: [target_pose_tokens] (24 tokens)
            # We predict the last 24 tokens (placeholder positions)
            pose_hidden_states = hidden_states[:, num_image_tokens:, :]
            pose_logits_only = self.pose_head(pose_hidden_states)
            
            # Extract only the target token predictions (last 24 tokens)
            num_target_tokens = labels.shape[-1]  # 24 target tokens
            target_hidden_states = pose_hidden_states[:, -num_target_tokens:, :]
            target_logits = self.pose_head(target_hidden_states)
            
            print(f"Hidden states shape: {hidden_states.shape}")
            print(f"Pose hidden states shape: {pose_hidden_states.shape}")
            print(f"Pose logits shape: {pose_logits_only.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # print(f"Loss computation - pose_hidden_states shape: {pose_hidden_states.shape}")
            # print(f"Loss computation - pose_logits_only shape: {pose_logits_only.shape}")
            
            # Input sequence: [image_tokens, input_pose_tokens, placeholder_tokens] (729 + 24 + 48 = 801 tokens)
            # Labels: [masked_input, target_pose_tokens] (24 masked + 48 target = 72 tokens)
            # We predict all 72 pose tokens, but compute loss only on the last 48 (placeholder positions)
            
            # print(f"Pose logits shape: {pose_logits_only.shape}")
            # print(f"Labels shape: {labels.shape}")
            
            # Use target logits and labels (only compute loss on target tokens)
            pose_logits = target_logits
            pose_labels = labels
            
            # Ensure labels have the same number of dimensions as logits
            if pose_labels.dim() > pose_logits.dim():
                while pose_labels.dim() > pose_logits.dim():
                    pose_labels = pose_labels.squeeze(0)
            elif pose_labels.dim() < pose_logits.dim():
                while pose_labels.dim() < pose_logits.dim():
                    pose_labels = pose_labels.unsqueeze(0)
            
            print(f"Final pose_logits shape: {pose_logits.shape}")
            print(f"Final pose_labels shape: {pose_labels.shape}")
            print(f"Pose logits flattened: {pose_logits.view(-1, pose_logits.size(-1)).shape}")
            print(f"Pose labels flattened: {pose_labels.view(-1).shape}")
            
            # Compute cross-entropy loss on pose tokens only
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pose_logits.view(-1, pose_logits.size(-1)), pose_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': pose_logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }