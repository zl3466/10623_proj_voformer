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
                grid_size = int(math.sqrt(seq_len))
                if grid_size * grid_size != seq_len:
                    raise ValueError(f"Expected squared sequence length, got {seq_len}")
                
                image_grid_thw = torch.tensor([[1, grid_size, grid_size] for _ in range(batch_size)], 
                                             dtype=torch.long, device=pixel_values.device)
            else:
                raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")
        else:
            image_grid_thw = None
        print(f"Image grid THW: {image_grid_thw}\n")

        print(f"Pixel values: {pixel_values}\n")
        
        # For Qwen2.5-VL with custom pose tokens, we need to create a combined input sequence
        # that includes image tokens + pose tokens
        if pixel_values is not None and input_ids is not None:
            batch_size = pixel_values.shape[0]
            
            # Calculate number of image tokens from the grid
            if image_grid_thw is not None:
                num_image_tokens = image_grid_thw[0, 1] * image_grid_thw[0, 2]  # H * W
            else:
                num_image_tokens = 0
            
            # Create image token IDs - use the correct Qwen2.5-VL image token
            image_token_id = 151645  # Qwen2.5-VL image token ID
            image_tokens = torch.full((batch_size, num_image_tokens), image_token_id, 
                                    dtype=torch.long, device=pixel_values.device)
            
            # Combine image tokens with pose tokens: [image_tokens, pose_tokens]
            combined_input_ids = torch.cat([image_tokens, input_ids], dim=1)
            
            # Update attention mask to include image tokens
            image_attention_mask = torch.ones((batch_size, num_image_tokens), 
                                             dtype=torch.long, device=pixel_values.device)
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
        
        # Apply pose prediction head
        pose_logits = self.pose_head(hidden_states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # We need to extract only the pose token predictions (skip image tokens)
            if pixel_values is not None and input_ids is not None and image_grid_thw is not None:
                num_image_tokens = image_grid_thw[0, 1] * image_grid_thw[0, 2]  # H * W
                
                # Extract only the pose token predictions (skip image tokens)
                pose_hidden_states = hidden_states[:, num_image_tokens:, :]
                pose_logits = self.pose_head(pose_hidden_states)
            else:
                pose_logits = self.pose_head(hidden_states)
            
            # Shift logits and labels for next-token prediction
            shift_logits = pose_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': pose_logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }