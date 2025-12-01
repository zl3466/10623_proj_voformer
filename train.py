"""
Main training script for Visual Odometry with NuScenes data
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.vo_trainer import NuScenesVOTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Visual Odometry model on NuScenes data')
    
    # Data arguments
    parser.add_argument('--nusc_data_path', type=str, required=True,
                        help='Path to NuScenes dataset')
    parser.add_argument('--nusc_meta_path', type=str, default='',
                        help='Path to save/load NuScenes metadata')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./vo_output',
                        help='Output directory for model checkpoints')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B',
                        help='Hugging Face model name (must be a causal LM, not VL model)')
    parser.add_argument('--vocab_size', type=int, default=1000,
                        help='Vocabulary size for pose tokenization')
    
    # Sequence arguments
    parser.add_argument('--num_input_frames', type=int, default=8,
                        help='Number of input frames')
    parser.add_argument('--num_input_poses', type=int, default=8,
                        help='Number of input poses')
    parser.add_argument('--num_target_poses', type=int, default=16,
                        help='Number of target poses to predict')
    
    # Image processing arguments
    parser.add_argument('--input_image_size', type=int, default=224,
                        help='Input image size (will be resized to this)')
    parser.add_argument('--num_tokens_image', type=int, default=256,
                        help='Fixed number of image tokens per image')
    
    # Pose tokenization arguments
    parser.add_argument('--pose_representation', type=str, default='6dof',
                        choices=['6dof', 'translation', 'full_matrix'],
                        help='Pose representation method')
    parser.add_argument('--num_tokens_pose', type=int, default=6,
                        help='Number of tokens per pose')
    parser.add_argument('--pose_quantization_range', type=float, default=2.0,
                        help='Range for pose quantization [-range, range]')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    # Wandb logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='visual-odometry',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='qwen25-0.5B-translation-deltas',
                        help='Wandb run name')
    
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    logger.info("Starting Visual Odometry training with NuScenes data")
    
    # Load configuration from YAML
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    # All command line arguments override the config values
    
    # Model configuration
    config['model']['name'] = args.model_name
    config['model']['vocab_size'] = args.vocab_size
    
    # Data configuration
    config['data']['num_input_frames'] = args.num_input_frames
    config['data']['num_input_poses'] = args.num_input_poses
    config['data']['num_target_poses'] = args.num_target_poses
    
    # Image configuration
    config['image']['input_size'] = args.input_image_size
    config['image']['num_tokens'] = args.num_tokens_image
    
    # Pose configuration
    config['pose']['pose_representation'] = args.pose_representation
    config['pose']['num_tokens_pose'] = args.num_tokens_pose
    config['pose']['quantization_range'] = args.pose_quantization_range
    
    # Training configuration
    config['training']['batch_size'] = args.batch_size
    config['training']['num_epochs'] = args.num_epochs
    config['training']['learning_rate'] = args.learning_rate
    config['training']['output_dir'] = args.output_dir
    
    # Wandb configuration
    # Only override use_wandb if explicitly specified in command line
    if '--use_wandb' in sys.argv:
        config['training']['use_wandb'] = args.use_wandb
    # Always override wandb_project and wandb_run_name if they have defaults
    config['training']['wandb_project'] = args.wandb_project
    config['training']['wandb_run_name'] = args.wandb_run_name
    
    logger.info("Command line arguments override config.yaml values")
    logger.info(f"  Model: {args.model_name}, Vocab size: {args.vocab_size}")
    logger.info(f"  Data: {args.num_input_frames} frames, {args.num_input_poses} input poses, {args.num_target_poses} target poses")
    logger.info(f"  Training: batch_size={args.batch_size}, epochs={args.num_epochs}, lr={args.learning_rate}")
    logger.info(f"  Wandb: enabled={args.use_wandb}")
    
    # Initialize trainer
    trainer = NuScenesVOTrainer(
        config=config,
        nusc_data_path=args.nusc_data_path,
        nusc_meta_path=args.nusc_meta_path
    )
    
    # Prepare data
    trainer.prepare_data()
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
