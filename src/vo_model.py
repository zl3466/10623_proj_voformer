import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
import math
import numpy as np

class VOModel(nn.Module):
    """VOModel for pose delta prediction with custom cross-entropy loss"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", vocab_size: int = 1000, config=None):
        super().__init__()
        
        # Store config for reference
        self.user_config = config

        # Load Qwen2.5-VL-3B base model (without LM head for custom processing)
        from transformers import AutoModel, AutoProcessor
        self.base_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load Qwen VL processor for tokenization
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # Get the tokenizer from processor
        self.tokenizer = self.processor.tokenizer
        
        # Custom pose prediction head for 48 output tokens (16 poses × 3 tokens each)
        hidden_size = self.base_model.config.hidden_size
        self.pose_head = nn.Linear(hidden_size, vocab_size)
        
        # Store vocab size
        self.vocab_size = vocab_size

    def process_inputs(self, images, poses):
        """Process images and poses using Qwen VL's processor"""
        # Convert poses to text representation
        pose_texts = []
        for pose in poses:
            if isinstance(pose, np.ndarray):
                pose_values = pose.tolist()
            else:
                pose_values = pose
            
            # Create text representation of pose
            pose_text = " ".join([f"{val:.4f}" for val in pose_values])
            pose_texts.append(pose_text)
        
        # Use Qwen VL's processor to handle both images and text
        processed = self.processor(
            images=images,
            text=pose_texts,
            return_tensors="pt",
            padding=True
        )
        
        return processed

    def forward(self, pixel_values=None, input_ids=None, labels=None, attention_mask=None, **kwargs):
        """
        Custom forward pass for pose delta prediction.
        Computes cross-entropy loss between predicted and ground truth pose tokens.
        """
        
        # Use Qwen2.5-VL base model for feature extraction
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get hidden states from the last layer
        hidden_states = outputs.last_hidden_state
        
        # Apply custom pose prediction head
        pose_logits = self.pose_head(hidden_states)
        
        # Compute custom cross-entropy loss for pose tokens
        loss = None
        if labels is not None:
            # For pose delta prediction, we want to predict the next 48 tokens
            # (16 poses × 3 tokens each)
            
            # Extract only the pose token positions for loss computation
            # Assuming input_ids contains the input pose tokens and we want to predict the next 48 tokens
            
            # Get the last 48 positions of the sequence for pose prediction
            batch_size, seq_len, vocab_size = pose_logits.shape
            
            # We want to predict the next 48 tokens after the input sequence
            if seq_len >= 48:
                # Take the last 48 positions for pose prediction
                pose_logits_pred = pose_logits[:, -48:, :]  # [batch_size, 48, vocab_size]
                pose_labels = labels[:, -48:]  # [batch_size, 48]
                
                # Compute cross-entropy loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    pose_logits_pred.reshape(-1, vocab_size), 
                    pose_labels.reshape(-1)
                )
            else:
                # If sequence is shorter than 48, pad or handle differently
                # For now, compute loss on available positions
                available_len = min(seq_len, 48)
                pose_logits_pred = pose_logits[:, -available_len:, :]
                pose_labels = labels[:, -available_len:]
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    pose_logits_pred.reshape(-1, vocab_size), 
                    pose_labels.reshape(-1)
                )
        
        return {
            'loss': loss,
            'logits': pose_logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }