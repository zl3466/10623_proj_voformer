import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import math
from .vision import DINOv2VisionEncoder
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def fuse_features(image_features: torch.Tensor, pose_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Simple function to fuse DINOv2 image features with pose embeddings.
    Concatenates image features and pose embeddings together.
    
    Args:
        image_features: DINOv2 image features [batch_size, num_patches, hidden_size]
        pose_embeddings: Pose embeddings from Qwen model [batch_size, seq_len, hidden_size]
        
    Returns:
        torch.Tensor: Concatenated features [batch_size, num_patches + seq_len, hidden_size]
    """
    # print(f"Image features shape: {image_features.shape}")
    # print(f"Pose embeddings shape: {pose_embeddings.shape}")
    
    # Both embeddings should already be in [batch_size, sequence_length, hidden_size] format
    # Just need to squeeze pose embeddings if they have an extra dimension
    if pose_embeddings.dim() == 4:
        pose_embeddings = pose_embeddings.squeeze(1)
        # print(f"Pose embeddings after squeeze: {pose_embeddings.shape}")
    
    # Concatenate along sequence dimension (dim=1)
    fused_embeddings = torch.cat([image_features, pose_embeddings], dim=1)
    # print(f"Fused embeddings shape: {fused_embeddings.shape}")
    return fused_embeddings

class VOModel(nn.Module):
    """
    Vision Odometry Model implementing the exact flow:
    images -> dinov2 -> image embeddings
    poses -> pose tokenizer -> tokens -> qwen embedding layer -> pose delta embeddings
    fuse image embeddings and pose delta embeddings
    fused embeddings -> qwen -> logits
    logits -> masking -> predicted future delta pose logits -> compute cross entropy loss
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B", vocab_size: int = 1000, config=None):
        super().__init__()
        
        # Store config for reference
        self.user_config = config

        # Load Qwen 0.5B model (text-only)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Add pose tokens to the tokenizer during initialization
        if config and 'model' in config and 'vocab_size' in config['model']:
            pose_vocab_size = config['model']['vocab_size']
            
            # Get the tokenizer and add pose tokens
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            original_vocab_size = len(tokenizer)
            
            # Add pose tokens: ["<i0>", "<i1>", ..., "<i{pose_vocab_size-1}>"]
            discrete_tokens = [f"<i{v}>" for v in range(pose_vocab_size)]
            tokenizer.add_tokens(discrete_tokens)
            
            # Store the start index of pose tokens
            self.pose_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
            
            # Resize the model's embedding layer to accommodate new tokens
            self.llm.resize_token_embeddings(len(tokenizer))
            
            # Initialize new pose token embeddings with small random values
            embedding_layer = self.llm.get_input_embeddings()
            with torch.no_grad():
                new_embeddings = embedding_layer.weight[original_vocab_size:]
                new_embeddings.normal_(mean=0.0, std=0.02)
            
            # Update the model's config
            self.llm.config.vocab_size = len(tokenizer)
            self.tokenizer = tokenizer
            
            logger.info(f"Added {pose_vocab_size} pose tokens starting at index {self.pose_token_start_idx}")
        
        # Initialize DINOv2 vision encoder
        self.vision_encoder = DINOv2VisionEncoder(
            model_name=config.get('vision', {}).get('dinov2_model', 'facebook/dinov2-base') if config else 'facebook/dinov2-base',
            hidden_size=self.llm.config.hidden_size
        )

        logger.info(f"Vision Odometry Model initialized")
        logger.info(f"Qwen model: {model_name}")
        logger.info(f"DINOv2 model: {config.get('vision', {}).get('dinov2_model', 'facebook/dinov2-base') if config else 'facebook/dinov2-base'}")
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
        
        # Verify DINOv2 is frozen
        if hasattr(self.vision_encoder, 'is_dinov2_frozen'):
            logger.info(f"DINOv2 frozen: {self.vision_encoder.is_dinov2_frozen()}")

    def forward(self, pixel_values=None, input_ids=None, labels=None, attention_mask=None, images=None, **kwargs):
        """
        Forward pass implementing the exact flow:
        1. images -> dinov2 -> image embeddings
        2. poses -> pose tokenizer -> tokens -> qwen embedding layer -> pose delta embeddings
        3. fuse image embeddings and pose delta embeddings
        4. fused embeddings -> qwen -> logits
        5. logits -> masking -> predicted future delta pose logits -> compute cross entropy loss
        """
        
        batch_size = input_ids.shape[0]
        
        # Step 1: images -> dinov2 -> image embeddings
        image_embeddings = self.vision_encoder.extract_features(images).to(dtype=torch.bfloat16)
        
        # Step 2: poses -> pose tokenizer -> tokens -> qwen embedding layer -> pose delta embeddings
        # Check for invalid token IDs
        vocab_size = self.llm.config.vocab_size
        invalid_tokens = (input_ids < 0) | (input_ids >= vocab_size)
        if invalid_tokens.any():
            input_ids = torch.where(invalid_tokens, torch.zeros_like(input_ids), input_ids)
        

        pose_embeddings = self.llm.model.embed_tokens(input_ids).to(dtype=torch.bfloat16)
        
        # Step 3: fuse image embeddings and pose delta embeddings
        fused_embeddings = fuse_features(image_embeddings, pose_embeddings)
        
        # Create attention mask for the combined sequence
        seq_len = fused_embeddings.shape[1]
        attention_mask = torch.ones(batch_size, seq_len, device=fused_embeddings.device)

        
        # Step 4: fused embeddings -> qwen -> logits
        fused_embeddings = fused_embeddings.to(dtype=torch.bfloat16)
        
        
        outputs = self.llm(
            inputs_embeds=fused_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get logits directly from Qwen model
        logits = outputs.logits
        
        
        # Step 5: logits -> masking -> predicted future delta pose logits -> compute cross entropy loss
        loss = None
        if labels is not None:
            # Fix labels shape if it's 3D
            if labels.dim() == 3:
                labels = labels.squeeze(1)  # Remove middle dimension
            
            # Apply masking to get only the target pose token predictions
            # The sequence is: [image_tokens, input_pose_tokens, target_pose_placeholders]
            # We only want to predict the target pose tokens, not the image or input pose tokens
            num_image_tokens = image_embeddings.shape[1]
            
            # Get input pose tokens from the actual pose embeddings
            # The pose embeddings contain both input and target tokens
            num_target_pose_tokens = labels.shape[1]
            num_input_pose_tokens = pose_embeddings.shape[2] - num_target_pose_tokens
            
            # Get logits for target pose positions only
            target_start_idx = num_image_tokens + num_input_pose_tokens
            target_end_idx = target_start_idx + num_target_pose_tokens
            
            target_logits = logits[:, target_start_idx:target_end_idx, :]
            
            # Compute cross entropy loss only on target positions
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(target_logits.reshape(-1, target_logits.size(-1)), labels.reshape(-1))
            
            # print(f"loss: {loss}")
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }

