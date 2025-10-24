"""
Training utilities for Visual Odometry
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import json

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # convert numpy array to list
        return super(NumpyEncoder, self).default(obj)


def compute_pose_metrics(predicted_poses: List[np.ndarray], 
                         target_poses: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute pose-specific evaluation metrics
    
    Args:
        predicted_poses: List of predicted pose arrays
        target_poses: List of target pose arrays
    
    Returns:
        Dictionary of metrics
    """
    if len(predicted_poses) != len(target_poses):
        raise ValueError("Predicted and target poses must have same length")
    
    # Convert to numpy arrays for easier computation
    pred_array = np.array(predicted_poses)
    target_array = np.array(target_poses)
    
    # Mean Per Joint Position Error (MPJPE)
    mpjpe = np.mean(np.linalg.norm(pred_array - target_array, axis=-1))
    
    # Percentage of Correct Keypoints (PCK) with threshold
    threshold = 0.1  # 10cm threshold
    joint_errors = np.linalg.norm(pred_array - target_array, axis=-1)
    pck = np.mean(joint_errors < threshold) * 100
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(pred_array - target_array))
    
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((pred_array - target_array) ** 2))
    
    return {
        'mpjpe': float(mpjpe),
        'pck': float(pck),
        'mae': float(mae),
        'rmse': float(rmse)
    }

def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float], 
                        save_path: str = "training_curves.png"):
    """
    Plot and save training curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training curves saved to {save_path}")

def plot_pose_predictions(predicted_poses: List[np.ndarray], 
                         target_poses: List[np.ndarray], 
                         save_path: str = "pose_predictions.png"):
    """
    Plot pose predictions vs targets
    
    Args:
        predicted_poses: List of predicted pose arrays
        target_poses: List of target pose arrays
        save_path: Path to save the plot
    """
    if len(predicted_poses) == 0:
        logger.warning("No poses to plot")
        return
    
    # Take first pose for visualization
    pred_pose = predicted_poses[0]
    target_pose = target_poses[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot predicted pose
    ax1.scatter(pred_pose[:, 0], pred_pose[:, 1], c='red', s=50, label='Predicted')
    ax1.set_title('Predicted Pose')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot target pose
    ax2.scatter(target_pose[:, 0], target_pose[:, 1], c='blue', s=50, label='Target')
    ax2.set_title('Target Pose')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Pose predictions plot saved to {save_path}")

def create_dummy_data(num_sequences: int = 10, 
                     num_frames: int = 24) -> List[Dict[str, Any]]:
    """
    Create dummy data for testing
    
    Args:
        num_sequences: Number of sequences to create
        num_frames: Number of frames per sequence
    
    Returns:
        List of dummy data samples
    """
    dummy_data = []
    
    for i in range(num_sequences):
        # Create dummy frames (random images)
        frames = []
        for j in range(num_frames):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            frames.append(img_array)
        
        # Create dummy poses
        poses = []
        for j in range(num_frames):
            pose = np.random.randn(17, 3) * 0.5
            poses.append(pose)
        
        dummy_data.append({
            'frames': frames,
            'poses': poses,
            'sequence_id': f'seq_{i:03d}'
        })
    
    return dummy_data

def save_training_config(config: Any, save_path: str):
    """
    Save training configuration to file
    
    Args:
        config: Configuration object
        save_path: Path to save configuration
    """
    import json
    
    # Convert dataclass to dictionary
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = vars(config)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Training configuration saved to {save_path}")

def load_training_config(load_path: str) -> Dict[str, Any]:
    """
    Load training configuration from file
    
    Args:
        load_path: Path to load configuration from
    
    Returns:
        Configuration dictionary
    """
    import json
    
    with open(load_path, 'r') as f:
        config_dict = json.load(f)
    
    logger.info(f"Training configuration loaded from {load_path}")
    return config_dict

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info(f"Logging setup complete (level: {log_level})")

def get_device_info() -> Dict[str, Any]:
    """
    Get device information for training
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
    }
    
    if torch.cuda.is_available():
        device_info['device_name'] = torch.cuda.get_device_name()
        device_info['memory_allocated'] = torch.cuda.memory_allocated()
        device_info['memory_reserved'] = torch.cuda.memory_reserved()
    
    return device_info

def print_training_summary(config: Any, model: torch.nn.Module, dataset_size: int):
    """
    Print training summary information
    
    Args:
        config: Training configuration
        model: Model to summarize
        dataset_size: Size of dataset
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"Model: {config['model']['name']}")
    print(f"Dataset size: {dataset_size}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Output directory: {config['training']['output_dir']}")
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Non-trainable: {total_params - trainable_params:,}")
    
    # Device info
    device_info = get_device_info()
    print(f"\nDevice: {device_info}")
    
    print("="*60)
