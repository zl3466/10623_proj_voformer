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
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

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
    
    def _save_checkpoint(self, model, trial):
        """
        Override _save_checkpoint to limit the number of saved checkpoints to 3.
        Before saving a new checkpoint, delete the earliest one if there are already 3 checkpoints.
        """
        # Call parent method to save the checkpoint
        super()._save_checkpoint(model, trial)
        
        # After saving, check and clean up old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Keep only the latest 3 checkpoints, deleting the oldest ones"""
        import os
        import glob
        import shutil
        
        output_dir = self.args.output_dir
        
        # Find all checkpoint directories
        checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        
        if len(checkpoint_dirs) > 3:
            # Sort checkpoints by their step number
            checkpoint_steps = []
            for ckpt_dir in checkpoint_dirs:
                try:
                    # Extract step number from checkpoint directory name
                    step = int(os.path.basename(ckpt_dir).split('-')[-1])
                    checkpoint_steps.append((step, ckpt_dir))
                except ValueError:
                    continue
            
            # Sort by step number (ascending)
            checkpoint_steps.sort(key=lambda x: x[0])
            
            # Delete the oldest checkpoint(s)
            num_to_delete = len(checkpoint_steps) - 3
            for i in range(num_to_delete):
                oldest_checkpoint = checkpoint_steps[i][1]
                logger.info(f"Deleting old checkpoint: {oldest_checkpoint}")
                shutil.rmtree(oldest_checkpoint)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
        # print(f"compute loss inputs: {inputs}")
        """Custom loss computation"""
        outputs = model(**inputs)
        loss = outputs['loss']
        
        # Debug: Check for NaN or zero loss
        if loss is not None:
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN or Inf loss detected! Loss: {loss}")
                # Check inputs
                if 'labels' in inputs:
                    logger.error(f"Labels shape: {inputs['labels'].shape}, min: {inputs['labels'].min()}, max: {inputs['labels'].max()}")
                if 'logits' in outputs:
                    logger.error(f"Logits shape: {outputs['logits'].shape}, has NaN: {torch.isnan(outputs['logits']).any()}")
            elif loss.item() == 0.0:
                logger.warning(f"Zero loss detected at step {self.state.global_step if hasattr(self.state, 'global_step') else 'unknown'}")
                # Check if labels are valid
                if 'labels' in inputs:
                    labels = inputs['labels']
                    logger.warning(f"Labels shape: {labels.shape}, unique values: {torch.unique(labels).numel()}, min: {labels.min()}, max: {labels.max()}")
                # Check logits
                if 'logits' in outputs:
                    logits = outputs['logits']
                    logger.warning(f"Logits shape: {logits.shape}, min: {logits.min()}, max: {logits.max()}, has NaN: {torch.isnan(logits).any()}")
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation with pose-specific metrics"""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # Print evaluation checkpoint info
        if hasattr(self.state, 'global_step'):
            print(f"\nEvaluating checkpoint at step {self.state.global_step}")
        
        # Run standard evaluation
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add pose-specific metrics
        pose_metrics = self._compute_pose_metrics(eval_dataset)
        eval_results.update(pose_metrics)
        
        return eval_results
    
    def _compute_pose_metrics(self, eval_dataset):
        """Compute pose-specific evaluation metrics with min_ade and time horizons"""
        print(f"\nComputing pose metrics...")

        self.model.eval()
        
        # Time horizons for evaluation
        time_horizons = [5, 10, 30]
        
        # Storage for metrics
        min_ade_metrics = {}
        translation_errors = []
        num_samples = 0
        
        with torch.no_grad():
            for batch in DataLoader(eval_dataset, batch_size=1, collate_fn=self.data_collator):
                # Move to device
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get predictions
                outputs = self.model(**batch)
                logits = outputs['logits']
                
                # Extract only target pose logits (same as loss computation)
                # Use the model's stored sequence structure
                num_image_tokens = self.model.num_image_tokens
                num_input_pose_tokens = self.model.num_history_pose_tokens
                num_target_pose_tokens = self.model.num_future_pose_tokens
                
                # print(f"eval: num_image_tokens: {num_image_tokens}")
                # print(f"eval:num_input_pose_tokens: {num_input_pose_tokens}")
                # print(f"eval:num_target_pose_tokens: {num_target_pose_tokens}")
                target_start_idx = num_image_tokens + num_input_pose_tokens
                target_end_idx = target_start_idx + num_target_pose_tokens
                
                # Get logits for target positions only
                target_logits = logits[:, target_start_idx:target_end_idx, :]
                predicted_tokens = torch.argmax(target_logits, dim=-1)
                # print(f"eval: predicted tokens: {predicted_tokens.shape}\n {predicted_tokens}")
                
                # Convert Qwen token IDs back to pose token IDs (0-1000 range)
                pose_token_start_id = self.model.pose_token_start_id
                # print(f"eval: pose_token_start_id: {pose_token_start_id}")
                # Convert predicted tokens from Qwen vocabulary to pose vocabulary (subtract token id by the offset of pose token start id)
                predicted_pose_tokens = predicted_tokens - pose_token_start_id
                # print(f"eval: predicted pose tokens: {predicted_pose_tokens.shape} \n{predicted_pose_tokens}")
                predicted_pose_tokens = torch.clamp(predicted_pose_tokens, 0, self.pose_tokenizer.vocab_size - 1)
                
                gt_pose_tokens = batch['labels'] - pose_token_start_id
                # print(f"eval: gt pose tokens: {gt_pose_tokens.shape} \n{gt_pose_tokens}")
                # Convert tokens back to poses using tokenizer (these are deltas)
                pred_deltas = self._tokens_to_poses(predicted_pose_tokens, self.pose_tokenizer)
                gt_deltas = self._tokens_to_poses(gt_pose_tokens, self.pose_tokenizer)
                
                # Accumulate deltas to get absolute trajectories for min_ade computation
                # Start from origin (0, 0, 0) for both - this ensures fair comparison
                pred_poses = self._accumulate_pose_deltas(pred_deltas)
                gt_poses = self._accumulate_pose_deltas(gt_deltas)
                
                # Compute min_ade for each time horizon on absolute coordinates
                # pred_poses and gt_poses are now absolute trajectories (including start position)
                # So we need to account for the +1 in length
                for horizon in time_horizons:
                    # horizon is the number of steps, but trajectories include start (length = horizon + 1)
                    if horizon + 1 <= len(pred_poses) and horizon + 1 <= len(gt_poses):
                        # Take up to horizon+1 positions (start + horizon steps)
                        min_ade = self._compute_min_ade(pred_poses[:horizon+1], gt_poses[:horizon+1])
                        if f'min_ade_{horizon}' not in min_ade_metrics:
                            min_ade_metrics[f'min_ade_{horizon}'] = []
                        min_ade_metrics[f'min_ade_{horizon}'].append(min_ade)
                
                # Compute translation error (XY plane only)
                translation_error = self._compute_translation_error(pred_poses, gt_poses)
                translation_errors.append(translation_error)
                
                num_samples += 1

        
        # Compute final metrics
        metrics = {}
        
        # Min ADE metrics
        for horizon in time_horizons:
            key = f'min_ade_{horizon}'
            if key in min_ade_metrics and min_ade_metrics[key]:
                metrics[f'eval_{key}'] = np.mean(min_ade_metrics[key])
                metrics[f'eval_{key}_std'] = np.std(min_ade_metrics[key])
        
        # Translation error metrics
        if translation_errors:
            metrics['eval_translation_error'] = np.mean(translation_errors)
            metrics['eval_translation_error_std'] = np.std(translation_errors)
            metrics['eval_translation_error_min'] = np.min(translation_errors)
            metrics['eval_translation_error_max'] = np.max(translation_errors)
            
            # Accuracy as percentage of samples with error < threshold
            threshold = 0.5  # 0.5 meters threshold
            accuracy = np.mean([err < threshold for err in translation_errors])
            metrics['eval_translation_accuracy'] = accuracy
        
        metrics['eval_num_samples'] = num_samples
        # print(f"\nComputed metrics for {num_samples} samples: {list(metrics.keys())}")
        
        # Log to wandb if available

        if wandb.run is not None:
            wandb.log(metrics)
            # print(f"Logged {len(metrics)} evaluation metrics to wandb")
        print(f"eval metrics:\n {metrics}")
        return metrics
    
    def _tokens_to_poses(self, tokens, tokenizer):
        """Convert tokenized poses back to continuous poses"""
        # Convert tokens to numpy and reshape
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        
        # Reshape tokens to pose format
        tokens_reshaped = tokens.reshape(-1, tokenizer.num_joints, tokenizer.pose_dim)
        
        # Convert to continuous poses
        poses = tokenizer.dequantize_pose(tokens_reshaped)
        
        return poses
    
    def _accumulate_pose_deltas(self, deltas, start_pose=None):
        """
        Accumulate pose deltas into absolute positions
        
        Args:
            deltas: Array of pose deltas [T, num_joints, pose_dim]
            start_pose: Starting pose [num_joints, pose_dim]. If None, starts from origin.
        
        Returns:
            Absolute poses [T+1, num_joints, pose_dim] (includes start position)
        """
        if start_pose is None:
            # Start from origin
            start_pose = np.zeros((deltas.shape[1], deltas.shape[2]))
        
        # Initialize trajectory with start pose
        trajectory = [start_pose.copy()]
        
        # Accumulate deltas
        current_pose = start_pose.copy()
        for delta in deltas:
            current_pose = current_pose + delta
            trajectory.append(current_pose.copy())
        
        return np.array(trajectory)
    
    def _compute_min_ade(self, pred_poses, gt_poses):
        """Compute Minimum Average Displacement Error (min_ade)"""
        # Ensure both sequences have the same length
        min_len = min(len(pred_poses), len(gt_poses))
        pred_poses = pred_poses[:min_len]
        gt_poses = gt_poses[:min_len]
        
        # Extract XY coordinates (first 2 dimensions)
        pred_xy = pred_poses[:, :, :2]  # [T, num_joints, 2]
        gt_xy = gt_poses[:, :, :2]     # [T, num_joints, 2]
        
        # Compute displacement error for each timestep and joint
        displacement_errors = np.linalg.norm(pred_xy - gt_xy, axis=2)  # [T, num_joints]
        
        # Average displacement error over time and joints
        ade = np.mean(displacement_errors)
        
        return ade
    
    def _compute_translation_error(self, pred_poses, gt_poses):
        """Compute translation error in XY plane"""
        # Ensure both sequences have the same length
        min_len = min(len(pred_poses), len(gt_poses))
        pred_poses = pred_poses[:min_len]
        gt_poses = gt_poses[:min_len]
        
        # Extract XY coordinates (first 2 dimensions)
        pred_xy = pred_poses[:, :, :2]  # [T, num_joints, 2]
        gt_xy = gt_poses[:, :, :2]     # [T, num_joints, 2]
        
        # Compute translation error for each timestep and joint
        translation_errors = np.linalg.norm(pred_xy - gt_xy, axis=2)  # [T, num_joints]
        
        # Return mean translation error
        return np.mean(translation_errors)
    
    def _log_sample_visualizations(self, eval_dataset, num_samples=3):
        """Log sample images, poses, and predictions to wandb for visualization"""
            
        self.model.eval()
        sample_data = []
        
        with torch.no_grad():
            for i, batch in enumerate(DataLoader(eval_dataset, batch_size=1, collate_fn=self.data_collator)):
                if i >= num_samples:
                    break
                    
                # Move to device
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get model outputs
                outputs = self.model(**batch)
                predicted_tokens = torch.argmax(outputs['logits'], dim=-1)
                
                # Extract sample data for visualization
                sample = {
                    'images': batch.get('images', None),
                    'input_poses': batch.get('input_ids', None),
                    'predicted_poses': predicted_tokens,
                    'ground_truth_poses': batch.get('labels', None),
                    'loss': outputs['loss'].item() if outputs['loss'] is not None else 0.0
                }
                sample_data.append(sample)
        
        # Create wandb visualizations
        self._create_wandb_visualizations(sample_data)
    
    def _create_wandb_visualizations(self, sample_data):
        """Create and log wandb visualizations for sample data
        not working now -- wrong way of getting predicted and ground truth poses
        """

        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create figure with subplots for each sample
        fig, axes = plt.subplots(len(sample_data), 4, figsize=(16, 4 * len(sample_data)))
        if len(sample_data) == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample in enumerate(sample_data):
            # Sample images (if available)
            if sample['images'] is not None and sample['images'].shape[0] > 0:
                # Take first image from the batch and convert to float32
                img = sample['images'][0, 0].cpu().float().numpy()  # Convert to float32
                if img.shape[0] == 3:  # CHW format
                    img = np.transpose(img, (1, 2, 0))
                img = np.clip(img, 0, 1)
                axes[i, 0].imshow(img)
                axes[i, 0].set_title(f'Sample {i+1}: Input Image')
            else:
                axes[i, 0].text(0.5, 0.5, 'No Image', ha='center', va='center')
                axes[i, 0].set_title(f'Sample {i+1}: No Image')
            
            # Input poses visualization
            if sample['input_poses'] is not None:
                input_poses = sample['input_poses'][0].cpu().float().numpy()  # Convert to float32
                axes[i, 1].plot(input_poses)
                axes[i, 1].set_title(f'Sample {i+1}: Input Poses')
                axes[i, 1].set_xlabel('Token Index')
                axes[i, 1].set_ylabel('Token Value')
            else:
                axes[i, 1].text(0.5, 0.5, 'No Input Poses', ha='center', va='center')
            
            # Predicted vs Ground Truth poses
            if sample['predicted_poses'] is not None and sample['ground_truth_poses'] is not None:
                pred_poses = sample['predicted_poses'][0].cpu().float().numpy()  # Convert to float32
                gt_poses = sample['ground_truth_poses'][0].cpu().float().numpy()  # Convert to float32
                
                # Plot both on same axes
                axes[i, 2].plot(pred_poses, label='Predicted', alpha=0.7)
                axes[i, 2].plot(gt_poses, label='Ground Truth', alpha=0.7)
                axes[i, 2].set_title(f'Sample {i+1}: Predicted vs GT')
                axes[i, 2].legend()
                axes[i, 2].set_xlabel('Token Index')
                axes[i, 2].set_ylabel('Token Value')
                
                # Error visualization
                error = np.abs(pred_poses - gt_poses)
                axes[i, 3].bar(range(len(error)), error, alpha=0.7)
                axes[i, 3].set_title(f'Sample {i+1}: Prediction Error')
                axes[i, 3].set_xlabel('Token Index')
                axes[i, 3].set_ylabel('Absolute Error')
            else:
                axes[i, 2].text(0.5, 0.5, 'No Predictions', ha='center', va='center')
                axes[i, 3].text(0.5, 0.5, 'No Error Data', ha='center', va='center')
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({"eval/sample_visualizations": wandb.Image(fig)})
        plt.close(fig)
        print(f"Logged sample visualizations to wandb: {len(sample_data)} samples")
            

class VOPoseDataCollator:
    """Data collator for Visual Odometry pose data"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        """Collate batch of visual and pose data"""
        # Convert images to tensors for DINOv2 processing
        batch_images = []
        for item in features:
            if 'images' in item:
                # Convert list of PIL images to tensor
                image_tensors = []
                for img in item['images']:
                    # Convert PIL to tensor and normalize
                    import numpy as np
                    img_tensor = torch.from_numpy(np.array(img)).float()
                    if img_tensor.dim() == 3:  # HWC
                        img_tensor = img_tensor.permute(2, 0, 1) / 255.0  # HWC to CHW
                    
                    # Normalize using ImageNet stats (DINOv2 expects this)
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_tensor = (img_tensor - mean) / std
                    image_tensors.append(img_tensor)
                
                # Stack images for this sample: [num_images, channels, height, width]
                sample_images = torch.stack(image_tensors)
                batch_images.append(sample_images)
        
        # Stack all samples: [batch_size, num_images, channels, height, width]
        if batch_images:
            images_tensor = torch.stack(batch_images)
        else:
            images_tensor = None
        
        input_ids = torch.stack([item['input_ids'] for item in features])
        labels = torch.stack([item['labels'] for item in features])
        attention_mask = torch.stack([item['attention_mask'] for item in features])
        
        # Convert images to bfloat16
        if images_tensor is not None:
            images_tensor = images_tensor.to(dtype=torch.bfloat16)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'images': images_tensor
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
        # No need for separate vision processor - everything is handled in the model
        
        self.pose_tokenizer = PoseDeltaTokenizer(
            num_joints=config['pose']['num_joints'],
            pose_dim=config['pose']['pose_dim'],
            vocab_size=config['model']['vocab_size']
        )
        
        # Initialize model
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
                name=config['training'].get('wandb_run_name', 'qwen25-0.5B-translation-deltas'),
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
        
        # Load all scenes from NuScenes dataset
        all_datasets = []

        max_scenes = 850  # there are 850 scenes in the train-val split

        nusc = NuScenes(version='v1.0-trainval', dataroot=self.nusc_data_path, verbose=True)
        for scene_idx in tqdm(range(max_scenes)):
            try:
                nusc_scene = NuScenesDataset(
                    data_path=self.nusc_data_path,
                    meta_out_path="",
                    num_cams=1,
                    nusc=nusc,
                    scene_idx=scene_idx,
                    save_meta=False)
                
                # Create pose dataset wrapper for this scene
                scene_pose_dataset = NuScenesPoseDataset(nusc_scene, None, self.config)
                scene_pose_dataset.model = self.model  # Pass model reference for pose token offset
                
                # Check if scene has enough samples
                if len(scene_pose_dataset) > 0:
                    all_datasets.append(scene_pose_dataset)
                else:
                    continue  # Skip this scene and move to next
                    
            except Exception as e:
                continue  # Skip this scene and move to next
        
        if not all_datasets:
            raise RuntimeError("No scenes could be loaded from the dataset")
        
        # Combine all scene datasets
        full_dataset = torch.utils.data.ConcatDataset(all_datasets)
        
        # Split into train/val using configurable split ratio
        total_size = len(full_dataset)
        train_split = self.config['data'].get('train_split', 0.8)  # Default to 80/20 if not specified
        train_size = int(train_split * total_size)
        
        print(f"Total dataset size: {total_size}")
        print(f"Train size: {train_size}")
        print(f"Val size: {total_size - train_size}")
        
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_size))
        
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    
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
        
        # Create data collator
        data_collator = VOPoseDataCollator(tokenizer=self.pose_tokenizer)
        
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
            save_safetensors=False,  # Disable safetensors to avoid shared tensor issues
        )
        
        # Create VOTrainer (inherits from HF Trainer)
        trainer = VOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )
        
        # Pass the pose tokenizer to the trainer
        trainer.pose_tokenizer = self.pose_tokenizer
        
        # Check for existing checkpoint
        # checkpoint = get_last_checkpoint(training_args.output_dir)
        # if checkpoint:
        #     logger.info(f"Resuming from checkpoint: {checkpoint}")
        #     trainer.train(resume_from_checkpoint=checkpoint)
        # else:
        trainer.train()
        
        # Save final model and tokenizer
        trainer.save_model()
        # save tokenizer configuration
        self.pose_tokenizer.save_pretrained(training_args.output_dir)
        
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
        """Predict future poses using DINOv2 + Qwen 0.5B with the new flow"""
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize input poses
            input_pose_tokens = self.pose_tokenizer.tokenize_sequence(input_poses)
            input_tokens = np.concatenate([tokens.flatten() for tokens in input_pose_tokens])
            input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Generate predictions using the model
            # The model handles: images->dinov2->embeddings, poses->tokenizer->tokens->qwen_embeddings
            outputs = self.model(
                input_ids=input_tokens,  # Pose tokens
                images=frames  # Raw images for DINOv2 processing
            )
            
            # Get predicted tokens
            predicted_tokens = torch.argmax(outputs['logits'], dim=-1)
            predicted_tokens = predicted_tokens.cpu().numpy().squeeze()
            
            # Convert back to poses
            num_target_poses = self.config['data']['num_target_poses']
            tokens_per_pose = self.pose_tokenizer.num_joints * self.pose_tokenizer.pose_dim
            target_tokens = predicted_tokens[-num_target_poses * tokens_per_pose:]
            
            # Reshape and convert to poses
            target_poses = []
            for i in range(num_target_poses):
                start_idx = i * tokens_per_pose
                end_idx = (i + 1) * tokens_per_pose
                pose_tokens = target_tokens[start_idx:end_idx].reshape(
                    self.pose_tokenizer.num_joints, self.pose_tokenizer.pose_dim
                )
                pose = self.pose_tokenizer.dequantize_pose(pose_tokens)
                target_poses.append(pose)
            
            return target_poses

def main():
    """Example usage of NuScenes VO trainer"""
    
    # Create configuration
    config = VOTrainingConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        output_dir="./vo_output",
        num_epochs=5,
        batch_size=2,
        learning_rate=1e-5
    )
    
    # Initialize trainer with NuScenes paths
    trainer = NuScenesVOTrainer(
        config=config,
        nusc_data_path="/home/zl3466/Documents/dataset/NuScenes",
        nusc_meta_path="./nusc_meta.json"
    )
    
    # Prepare data
    trainer.prepare_data()
    
    # Train model
    trainer.train()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()