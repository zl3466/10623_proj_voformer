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
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for model checkpoints (default: from config.yaml)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of training epochs (default: from config.yaml)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (default: from config.yaml)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (default: from config.yaml)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default=None,
                        help='Hugging Face model name (must be a causal LM, not VL model) (default: from config.yaml)')
    parser.add_argument('--vocab_size', type=int, default=None,
                        help='Vocabulary size for pose tokenization (default: from config.yaml)')
    
    # Sequence arguments
    parser.add_argument('--num_input_frames', type=int, default=None,
                        help='Number of input frames (default: from config.yaml)')
    parser.add_argument('--num_input_poses', type=int, default=None,
                        help='Number of input poses (default: from config.yaml)')
    parser.add_argument('--num_target_poses', type=int, default=None,
                        help='Number of target poses to predict (default: from config.yaml)')
    
    # Image processing arguments
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='Input image size (will be resized to this) (default: from config.yaml)')
    parser.add_argument('--num_tokens_image', type=int, default=None,
                        help='Fixed number of image tokens per image (default: from config.yaml)')
    
    # Pose tokenization arguments
    parser.add_argument('--pose_representation', type=str, default=None,
                        choices=['6dof', 'translation', 'full_matrix'],
                        help='Pose representation method (default: from config.yaml)')
    parser.add_argument('--num_tokens_pose', type=int, default=None,
                        help='Number of tokens per pose (default: from config.yaml)')
    parser.add_argument('--pose_quantization_range', type=float, default=None,
                        help='Range for pose quantization [-range, range] (default: from config.yaml)')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    # Wandb logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging (default: from config.yaml)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging (overrides config)')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Wandb project name (default: from config.yaml)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (default: from config.yaml)')
    
    
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
    
    # Override config with command line arguments (only if explicitly provided)
    # Mapping: (config_path_tuple, arg_name, arg_value)
    config_overrides = [
        (['model', 'name'], 'model_name', args.model_name),
        (['model', 'vocab_size'], 'vocab_size', args.vocab_size),
        (['data', 'num_input_frames'], 'num_input_frames', args.num_input_frames),
        (['data', 'num_input_poses'], 'num_input_poses', args.num_input_poses),
        (['data', 'num_target_poses'], 'num_target_poses', args.num_target_poses),
        (['image', 'input_size'], 'input_image_size', args.input_image_size),
        (['image', 'num_tokens'], 'num_tokens_image', args.num_tokens_image),
        (['pose', 'pose_representation'], 'pose_representation', args.pose_representation),
        (['pose', 'num_tokens_pose'], 'num_tokens_pose', args.num_tokens_pose),
        (['pose', 'quantization_range'], 'pose_quantization_range', args.pose_quantization_range),
        (['training', 'batch_size'], 'batch_size', args.batch_size),
        (['training', 'num_epochs'], 'num_epochs', args.num_epochs),
        (['training', 'learning_rate'], 'learning_rate', args.learning_rate),
        (['training', 'output_dir'], 'output_dir', args.output_dir),
        (['training', 'wandb_project'], 'wandb_project', args.wandb_project),
        (['training', 'wandb_run_name'], 'wandb_run_name', args.wandb_run_name),
    ]
    
    overrides = []
    for config_path, arg_name, arg_value in config_overrides:
        if arg_value is not None:
            # Navigate to the config path and set the value
            target = config
            for key in config_path[:-1]:
                target = target[key]
            target[config_path[-1]] = arg_value
            overrides.append(f"{arg_name}={arg_value}")
    
    # Handle wandb boolean flag separately
    if args.no_wandb:
        config['training']['use_wandb'] = False
        overrides.append("use_wandb=False")
    elif '--use_wandb' in sys.argv:
        config['training']['use_wandb'] = True
        overrides.append("use_wandb=True")
    
    if overrides:
        logger.info(f"Command line overrides: {', '.join(overrides)}")
    else:
        logger.info("Using all values from config.yaml")
    
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
