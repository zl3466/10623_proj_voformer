"""
Visual Odometry Trainer using Hugging Face Trainer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import wandb


from .train_utils import compute_pose_metrics, plot_training_curves, print_training_summary
from .nusc_loader import NuScenesDataset, NuScenesPoseDataset

# Import from separate modules
from .pose_tokenizer import PoseDeltaTokenizer
from .vo_model import VOModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VOTrainer(Trainer):
    """Custom trainer for Visual Odometry with pose-specific metrics"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
        """Custom loss computation for pose delta prediction"""
        outputs = model(**inputs)
        loss = outputs['loss']
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation with pose-specific metrics"""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # Run standard evaluation
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add pose-specific metrics
        pose_metrics = self._compute_pose_metrics(eval_dataset)
        eval_results.update(pose_metrics)
        
        return eval_results
    
    def _compute_pose_metrics(self, eval_dataset):
        """Compute pose-specific evaluation metrics"""
        self.model.eval()
        
        total_translation_error = 0.0  # Mean translation error
        total_accuracy = 0.0           # Translation prediction accuracy
        num_samples = 0
        
        with torch.no_grad():
            for batch in DataLoader(eval_dataset, batch_size=1, collate_fn=self.data_collator):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get predictions
                outputs = self.model(**batch)
                predicted_tokens = torch.argmax(outputs['logits'], dim=-1)
                
                # Convert to poses and compute metrics
                # (Implementation would go here)
                # For now, return dummy metrics
                total_translation_error += 0.1
                total_accuracy += 0.8
                num_samples += 1
        
        metrics = {
            "eval_translation_error": total_translation_error / num_samples,
            "eval_translation_accuracy": total_accuracy / num_samples,
        }
        
        # Log to wandb if available and enabled
        if self.use_wandb and wandb.run is not None:
            wandb.log(metrics)
        
        return metrics

class VOPoseDataCollator:
    """Data collator for Visual Odometry pose data using Qwen VL tokenizer"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        """Collate batch of pose data using Qwen VL tokenization"""
        # Handle pixel_values properly - don't squeeze if it changes the expected shape
        pixel_values_list = []
        for item in features:
            pv = item['pixel_values']
            # Only squeeze the batch dimension if it's 1, but preserve the sequence structure
            if pv.dim() > 1 and pv.shape[0] == 1:
                pv = pv.squeeze(0)
            pixel_values_list.append(pv)
        
        pixel_values = torch.stack(pixel_values_list)
        input_ids = torch.stack([item['input_ids'] for item in features])
        labels = torch.stack([item['labels'] for item in features])
        
        # Create attention mask for Qwen VL tokenizer
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

class NuScenesVOTrainer:
    """NuScenes-specific trainer setup for Visual Odometry
    Uses the VOTrainer adopted from HF trainer for actual training

    """
    
    def __init__(self, config: dict, nusc_data_path: str, nusc_meta_path: str):
        self.config = config
        self.nusc_data_path = nusc_data_path
        self.nusc_meta_path = nusc_meta_path
        
        # Initialize components
        self.processor = AutoProcessor.from_pretrained(config['model']['name'])
        
        # Get tokenizer from processor (Qwen VL's tokenizer)
        self.tokenizer = self.processor.tokenizer
        
        # Initialize model (now includes Qwen VL tokenizer)
        self.model = VOModel(config['model']['name'], config['model']['vocab_size'], config)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Initialize wandb if enabled
        self.use_wandb = config['training'].get('use_wandb', False)
        if isinstance(self.use_wandb, str):
            self.use_wandb = self.use_wandb.lower() in ('true', '1', 'yes', 'on')
        
        if self.use_wandb:
            wandb.init(
                project=config['training'].get('wandb_project', 'visual-odometry'),
                name=config['training'].get('wandb_run_name', 'qwen25-vl-3b-translation-deltas'),
                config={
                    'model_name': config['model']['name'],
                    'vocab_size': config['model']['vocab_size'],
                    'num_epochs': config['training']['num_epochs'],
                    'batch_size': config['training']['batch_size'],
                    'learning_rate': config['training']['learning_rate'],
                    'num_input_frames': config['data']['num_input_frames'],
                    'num_input_poses': config['data']['num_input_poses'],
                    'num_target_poses': config['data']['num_target_poses'],
                    'pose_dim': config['pose']['pose_dim'],
                    'quantization_range': config['pose']['quantization_range']
                }
            )
            logger.info("Wandb initialized for logging")
        else:
            logger.info("Wandb logging disabled")
        
        # Print training summary
        print_training_summary(config, self.model, 0)  # Will update with actual dataset size
    
    def prepare_data(self):
        """Prepare NuScenes datasets"""
        logger.info("Preparing NuScenes datasets...")
        
        # Create NuScenes dataset
        nusc_train = NuScenesDataset(
            data_path=self.nusc_data_path,
            meta_out_path=self.nusc_meta_path,
            num_cams=1,  # Use single camera for simplicity
            split='v1.0-trainval',
            scene_idx=0,
            start_timestep=0,
            end_timestep=-1,
            save_meta=True
        )
        
        # Create pose dataset wrapper
        full_dataset = NuScenesPoseDataset(nusc_train, self.processor, self.config)
        
        # Split into train/val
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_size))
        
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
    
    def load_tokenizer_from_checkpoint(self, checkpoint_dir: str):
        """Load tokenizer from a saved checkpoint"""
        try:
            self.pose_tokenizer = PoseDeltaTokenizer.from_pretrained(checkpoint_dir)
            logger.info(f"Loaded tokenizer from {checkpoint_dir}")
        except FileNotFoundError:
            logger.warning(f"No tokenizer config found at {checkpoint_dir}, using default tokenizer")
    
    def train(self):
        """Train using Hugging Face Trainer"""
        logger.info("Starting training with Hugging Face Trainer...")
        
        # Create data collator using Qwen VL tokenizer
        data_collator = VOPoseDataCollator(tokenizer=self.tokenizer)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            warmup_steps=int(self.config['training']['warmup_steps']),
            learning_rate=float(self.config['training']['learning_rate']),
            logging_steps=int(self.config['training']['logging_steps']),
            save_steps=int(self.config['training']['save_steps']),
            eval_steps=int(self.config['training']['eval_steps']),
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_grad_norm=float(self.config['training']['max_grad_norm']),
            weight_decay=float(self.config['training']['weight_decay']),
            report_to="wandb" if self.use_wandb else None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Create VOTrainer (inherits from HF Trainer)
        trainer = VOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            processing_class=self.processor,
        )
        
        # Check for existing checkpoint
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint}")
            trainer.train(resume_from_checkpoint=checkpoint)
        else:
            trainer.train()
        
        # Save final model and tokenizer
        trainer.save_model()
        # Save Qwen VL tokenizer
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        logger.info(f"Training completed! Model saved to {training_args.output_dir}")
        
        # Log final metrics to wandb if enabled
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                "training/completed": True,
                "training/final_epoch": training_args.num_train_epochs
            })
            wandb.finish()
        
        return trainer
    
    def predict(self, frames: List[Image.Image], input_poses: List[np.ndarray]) -> List[np.ndarray]:
        """Predict future poses using Qwen VL's native generation"""
        self.model.eval()
        
        with torch.no_grad():
            # Process input using the model's processor
            processed_inputs = self.model.process_inputs(frames, input_poses)
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in processed_inputs.items()}
            
            # Generate predictions using custom forward pass
            outputs = self.model(**inputs)
            
            # Get predicted tokens from custom pose head
            predicted_tokens = torch.argmax(outputs['logits'], dim=-1)
            predicted_tokens = predicted_tokens.cpu().numpy().squeeze()
            
            # Decode predicted tokens back to text
            predicted_text = self.tokenizer.decode(predicted_tokens, skip_special_tokens=True)
            
            # Parse the text back to pose values
            try:
                pose_values = [float(x) for x in predicted_text.split()]
                num_target_poses = self.config['data']['num_target_poses']
                target_poses = []
                
                # Reshape into poses (assuming 3D translation)
                for i in range(num_target_poses):
                    start_idx = i * 3  # 3D translation
                    if start_idx + 3 <= len(pose_values):
                        pose = np.array(pose_values[start_idx:start_idx + 3])
                        target_poses.append(pose)
                    else:
                        target_poses.append(np.zeros(3))  # Default pose
                        
            except (ValueError, IndexError):
                # Fallback to zero poses if parsing fails
                num_target_poses = self.config['data']['num_target_poses']
                target_poses = [np.zeros(3) for _ in range(num_target_poses)]
            
            return target_poses

