import numpy as np
from typing import List


class PoseDeltaTokenizer:
    """Tokenizes delta poses into discrete tokens for language modeling"""

    def __init__(self, num_joints: int = 17, pose_dim: int = 3, vocab_size: int = 1000):
        self.num_joints = num_joints
        self.pose_dim = pose_dim
        self.vocab_size = vocab_size
        self.pose_dimension = num_joints * pose_dim  # Total pose dimensions

        # Create quantization bins for each dimension
        self.quantization_bins = np.linspace(-2, 2, vocab_size)

    def quantize_pose(self, pose: np.ndarray) -> np.ndarray:
        """Quantize continuous pose to discrete tokens"""
        # Flatten pose to 1D
        flat_pose = pose.flatten()

        # Quantize each dimension
        quantized = np.digitize(flat_pose, self.quantization_bins) - 1
        quantized = np.clip(quantized, 0, self.vocab_size - 1)

        return quantized.astype(np.int32)

    def dequantize_pose(self, tokens: np.ndarray) -> np.ndarray:
        """Convert tokens back to continuous poses"""
        # Reshape tokens back to pose shape
        pose_tokens = tokens.reshape(-1, self.num_joints, self.pose_dim)

        # Convert tokens back to continuous values
        continuous_poses = self.quantization_bins[pose_tokens]

        return continuous_poses

    def tokenize_sequence(self, poses: List[np.ndarray]) -> List[np.ndarray]:
        """Tokenize a sequence of poses"""
        return [self.quantize_pose(pose) for pose in poses]

    def detokenize_sequence(self, token_sequences: List[np.ndarray]) -> List[np.ndarray]:
        """Detokenize a sequence of pose tokens"""
        return [self.dequantize_pose(tokens) for tokens in token_sequences]
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer configuration to directory"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save tokenizer configuration
        config = {
            "num_joints": self.num_joints,
            "pose_dim": self.pose_dim,
            "vocab_size": self.vocab_size,
            "pose_dimension": self.pose_dimension,
            "quantization_bins": self.quantization_bins.tolist()
        }
        
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Tokenizer configuration saved to {config_path}")
    
    @classmethod
    def from_pretrained(cls, save_directory: str):
        """Load tokenizer from saved configuration"""
        import os
        import json
        
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Tokenizer config not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create tokenizer instance
        tokenizer = cls(
            num_joints=config["num_joints"],
            pose_dim=config["pose_dim"],
            vocab_size=config["vocab_size"]
        )
        
        # Restore quantization bins
        tokenizer.quantization_bins = np.array(config["quantization_bins"])
        
        return tokenizer