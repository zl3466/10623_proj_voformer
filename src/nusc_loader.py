import json
import logging
import os

import numpy as np
import torch
from nuscenes.nuscenes import LidarPointCloud, NuScenes
from pyquaternion import Quaternion
from torch import Tensor
from train_utils import NumpyEncoder
from PIL import Image

logger = logging.getLogger()

class NuScenesDataset():
    # ORIGINAL_SIZE = [[900, 1600] for _ in range(6)]
    OPENCV2DATASET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    def __init__(
        self,
        data_path: str,
        meta_out_path: str,
        num_cams: int = 1,
        nusc: NuScenes = None,
        split: str = 'train',
        scene_idx: int = 0,
        start_timestep: int = 0,
        end_timestep: int = -1,
        save_meta=True
    ):

        self.data_path = data_path
        self.meta_out_path = meta_out_path
        self.num_cams = num_cams
        self.split = split
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.save_meta = save_meta

        self.nusc = nusc
        self.scene_idx = scene_idx
        self.meta_dict = self.create_or_load_metas()
        self.create_all_filelist()
        self.load_calibrations()


    def create_or_load_metas(self):
        # ---- define camera list ---- #
        if self.num_cams == 1:
            self.camera_list = ["CAM_FRONT"]
        elif self.num_cams == 3:
            self.camera_list = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
        elif self.num_cams == 6:
            self.camera_list = [
                "CAM_FRONT_LEFT",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
            ]
        else:
            raise NotImplementedError(
                f"num_cams: {self.num_cams} not supported for nuscenes dataset"
            )

        if os.path.exists(self.meta_out_path):
            # print(self.meta_out_path)
            with open(self.meta_out_path, "r") as f:
                meta_dict = json.load(f)
            # logger.info(f"[Nuscenes] Loaded camera meta from {self.meta_out_path}")
            return meta_dict
        
        if self.nusc is None:
            self.nusc = NuScenes(
                version=self.split, dataroot=self.data_path, verbose=True
            )
        self.scene = self.nusc.scene[self.scene_idx]

        meta_dict = {
            camera: {
                "timestamp": [],
                "filepath": [],
                "ego_pose_original": [],
                "ego_pose_matrix": [],
                "cam_id": [],
                "extrinsics": [],
                "intrinsics": [],
            }
            for i, camera in enumerate(self.camera_list)
        }

        # ---- get the first sample of each camera ---- #
        current_camera_data_tokens = {camera: None for camera in self.camera_list}
        first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        for camera in self.camera_list:
            current_camera_data_tokens[camera] = first_sample["data"][camera]

        while not all(token == "" for token in current_camera_data_tokens.values()):
            for i, camera in enumerate(self.camera_list):
                # skip if the current camera data token is empty
                if current_camera_data_tokens[camera] == "":
                    continue

                current_camera_data = self.nusc.get(
                    "sample_data", current_camera_data_tokens[camera]
                )

                # ---- timestamp and cam_id ---- #
                meta_dict[camera]["cam_id"].append(i)
                meta_dict[camera]["timestamp"].append(current_camera_data["timestamp"])
                meta_dict[camera]["filepath"].append(current_camera_data["filename"])

                # ---- intrinsics and extrinsics ---- #
                calibrated_sensor_record = self.nusc.get(
                    "calibrated_sensor", current_camera_data["calibrated_sensor_token"]
                )
                # intrinsics
                intrinsic = calibrated_sensor_record["camera_intrinsic"]
                meta_dict[camera]["intrinsics"].append(np.array(intrinsic))

                # extrinsics
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = Quaternion(
                    calibrated_sensor_record["rotation"]
                ).rotation_matrix
                extrinsic[:3, 3] = np.array(calibrated_sensor_record["translation"])
                meta_dict[camera]["extrinsics"].append(extrinsic)

                # ---- ego pose ---- #
                ego_pose_record = self.nusc.get(
                    "ego_pose", current_camera_data["ego_pose_token"]
                )
                ego_pose = np.eye(4)
                ego_pose[:3, :3] = Quaternion(
                    ego_pose_record["rotation"]
                ).rotation_matrix
                ego_pose[:3, 3] = np.array(ego_pose_record["translation"])
                meta_dict[camera]["ego_pose_original"].append(ego_pose_record)
                meta_dict[camera]["ego_pose_matrix"].append(ego_pose)

                current_camera_data_tokens[camera] = current_camera_data["next"]

        if self.save_meta:
            with open(self.meta_out_path, "w") as f:
                json.dump(meta_dict, f, cls=NumpyEncoder)
            logger.info(f"[Pixel] Saved camera meta to {self.meta_out_path}")
        return meta_dict

    def create_all_filelist(self):
        # NuScenes dataset is not synchronized, so we need to find the minimum shared
        # scene length, and only use the frames within the shared scene length.
        # we also define the start and end timestep within the shared scene length

        # ---- find the minimum shared scene length ---- #
        num_timestamps = 100000000
        for camera in self.camera_list:
            if len(self.meta_dict[camera]["timestamp"]) < num_timestamps:
                num_timestamps = len(self.meta_dict[camera]["timestamp"])
        # logger.info(f"[Pixel] Min shared scene length: {num_timestamps}")
        self.scene_total_num_timestamps = num_timestamps

        if self.end_timestep == -1:
            self.end_timestep = num_timestamps - 1
        else:
            self.end_timestep = min(self.end_timestep, num_timestamps - 1)

        # to make sure the last timestep is included
        self.end_timestep += 1
        self.start_timestep = min(self.start_timestep, self.end_timestep - 1)

        # logger.info(f"[Pixel] Start timestep: {self.start_timestep}")
        # logger.info(f"[Pixel] End timestep: {self.end_timestep}")

        img_filepaths, rel_img_filepaths, feat_filepaths, sky_mask_filepaths = [], [], [], []
        # TODO: support dynamic masks

        for t in range(self.start_timestep, self.end_timestep):
            for cam_idx in self.camera_list:
                img_filepath = os.path.join(
                    self.data_path, self.meta_dict[cam_idx]["filepath"][t]
                )
                img_filepaths.append(img_filepath)
                rel_img_filepaths.append(self.meta_dict[cam_idx]["filepath"][t])

        self.img_filepaths = np.array(img_filepaths)
        self.rel_img_filepaths = np.array(rel_img_filepaths)


    def load_calibrations(self):
        # compute per-image poses and intrinsics
        cam_to_worlds = []
        intrinsics, timesteps, cam_ids = [], [], []
        timestamps = []

        # we tranform the camera poses w.r.t. the first timestep to make the origin of
        # the first ego pose  as the origin of the world coordinate system.
        initial_ego_to_global = self.meta_dict["CAM_FRONT"]["ego_pose_matrix"][
            self.start_timestep
        ]
        global_to_initial_ego = np.linalg.inv(initial_ego_to_global)

        min_timestamp = 1e20
        max_timestamp = 0
        for cam_name in self.camera_list:
            min_timestamp = min(
                min_timestamp,
                self.meta_dict[cam_name]["timestamp"][self.start_timestep],
            )
            max_timestamp = max(
                max_timestamp,
                self.meta_dict[cam_name]["timestamp"][self.end_timestep - 1],
            )
        self.min_timestamp = min_timestamp
        self.max_timestamp = max_timestamp

        for t in range(self.start_timestep, self.end_timestep):
            for cam_name in self.camera_list:
                cam_to_ego = self.meta_dict[cam_name]["extrinsics"][t]
                ego_to_global_current = self.meta_dict[cam_name]["ego_pose_matrix"][t]
                # compute ego_to_world transformation
                ego_to_world = global_to_initial_ego @ ego_to_global_current
                # Because we use opencv coordinate system to generate camera rays,
                # we need to store the transformation from opencv coordinate system to dataset
                # coordinate system. However, the nuScenes dataset uses the same coordinate
                # system as opencv, so we just store the identity matrix.
                # opencv coordinate system: x right, y down, z front
                cam_to_ego = cam_to_ego @ self.OPENCV2DATASET
                cam2world = ego_to_world @ cam_to_ego
                cam_to_worlds.append(cam2world)
                intrinsics.append(self.meta_dict[cam_name]["intrinsics"][t])
                timesteps.append(t - self.start_timestep)
                cam_ids.append(self.meta_dict[cam_name]["cam_id"][t])
                timestamps.append(
                    self.meta_dict[cam_name]["timestamp"][t]
                    * np.ones_like(
                        self.meta_dict[cam_name]["cam_id"][t], dtype=np.float64
                    )
                )

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()

        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        self.global_to_initial_ego = torch.from_numpy(global_to_initial_ego).float()
        self.cam_ids = torch.from_numpy(np.stack(cam_ids, axis=0)).long()

        # the underscore here is important.
        self._timestamps = torch.tensor(timestamps, dtype=torch.float64)
        self._timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).long()


class NuScenesPoseDataset:
    """Dataset wrapper for NuScenes data with pose prediction"""
    
    def __init__(self, nusc_dataset: NuScenesDataset, processor, config):
        self.nusc_dataset = nusc_dataset
        self.processor = processor  # Can be None since we don't use it anymore
        self.config = config
        
        
        # Extract poses from camera poses (ego poses)
        self.poses = self._extract_poses()
        self.images = self._extract_images()
        
        # logger.info(f"Loaded {len(self.poses)} poses and {len(self.images)} images")
    
    def _extract_poses(self):
        """Extract poses from NuScenes camera poses"""
        poses = []
        
        # Get camera-to-world transformations
        cam_to_worlds = self.nusc_dataset.cam_to_worlds.numpy()
        
        # Extract translation and rotation for each pose
        for i in range(len(cam_to_worlds)):
            pose_matrix = cam_to_worlds[i]
            
            # Extract translation (x, y, z)
            translation = pose_matrix[:3, 3]
            
            # Extract rotation as euler angles or quaternion
            # For simplicity, we'll use translation as the main pose component
            # and add some rotation information
            rotation_matrix = pose_matrix[:3, :3]
            
            # Convert to a simplified pose representation
            # This is a simplified approach - you might want to use full 6DOF poses
            pose = np.concatenate([translation, rotation_matrix.flatten()[:3]])  # 6D pose
            poses.append(pose)
        
        return np.array(poses)
    
    def _extract_images(self):
        """Extract image paths from NuScenes dataset"""
        return self.nusc_dataset.img_filepaths
    
    def __len__(self):
        num_input_poses_needed = self.config['data']['num_input_poses'] + 1
        num_target_poses_needed = self.config['data']['num_target_poses'] + 1
        
        # We need enough poses for both input and target
        total_poses_needed = num_input_poses_needed + num_target_poses_needed
        
        if len(self.poses) < total_poses_needed:
            return 0  # Not enough data for any valid sequences
        
        # We can start from idx=0 and go up to the last valid starting point
        # The last valid idx is: len(self.poses) - total_poses_needed
        max_valid_idx = len(self.poses) - total_poses_needed
        
        return max_valid_idx + 1  # +1 because we include idx=0
    
    def __getitem__(self, idx):
        """Get a single sample"""
        # Adjust idx to account for the offset in __len__
        num_input_images_needed = self.config['data']['num_input_frames']
        actual_idx = idx + num_input_images_needed
        
        # Load image
        img_path = self.images[actual_idx]
        image = Image.open(img_path).convert('RGB')
        
        # Resize image if specified in config
        if self.config and 'image' in self.config and 'input_size' in self.config['image']:
            image = image.resize((self.config['image']['input_size'], self.config['image']['input_size']), Image.Resampling.LANCZOS)
        
        # Get pose
        pose = self.poses[idx]
        
        # Get sequence parameters from config
        num_input_deltas = self.config['data']['num_input_poses'] if self.config else 8  # Number of deltas we want
        num_target_deltas = self.config['data']['num_target_poses'] if self.config else 16  # Number of deltas we want
        
        # For deltas, we need n+1 poses to get n deltas
        num_input_poses_needed = num_input_deltas + 1  # Need 9 poses to get 8 deltas
        num_target_poses_needed = num_target_deltas + 1  # Need 17 poses to get 16 deltas
        
        # Images should match the number of poses needed
        num_input_images_needed = self.config['data']['num_input_frames']
        
        # For training, we only need input images (9 images) to predict future pose deltas (16 deltas)
        # We don't need target images, only target poses
        
        # Check if we have enough poses for target
        target_start_idx = idx + num_input_poses_needed
        target_end_idx = target_start_idx + num_target_poses_needed
        
        if target_end_idx > len(self.poses):
            # Not enough poses for target, skip this sample
            raise IndexError(f"Not enough poses for target sequence at idx {actual_idx}")
        
        # Create sequence - only input images and poses
        input_poses = [self.poses[i] for i in range(idx, idx + num_input_poses_needed)]
        target_poses = [self.poses[i] for i in range(target_start_idx, target_end_idx)]
        input_images = []
        # Only load input images (9 images)
        for i in range(idx, idx + num_input_images_needed):
            img = Image.open(self.images[i]).convert('RGB')
            if self.config and 'image' in self.config and 'input_size' in self.config['image']:
                img = img.resize((self.config['image']['input_size'], self.config['image']['input_size']), Image.Resampling.LANCZOS)
            input_images.append(img)
        
        # Step 1: Images will be processed by DINOv2 in the model forward pass
        # No preprocessing needed here
        
        # Step 2: Tokenize poses using your custom pose delta tokenizer
        input_pose_tokens, target_pose_tokens = self._tokenize_poses_with_config(input_poses, target_poses)
        
        # Step 3: Create extended input sequence with placeholder tokens
        # Input sequence: [input_pose_tokens, placeholder_tokens] (24 + 48 = 72 tokens)
        # Labels: [input_pose_tokens, target_pose_tokens] (24 + 48 = 72 tokens)
        # Loss will be computed only on placeholder positions (last 48 tokens)
        
        # Create placeholder tokens (use a special token ID, e.g., 0 or a specific placeholder ID)
        placeholder_token_id = 0  # Use 0 as placeholder token
        placeholder_tokens = np.full(len(target_pose_tokens), placeholder_token_id, dtype=np.int32)
        
        # Combine input pose tokens with placeholder tokens
        extended_pose_tokens = np.concatenate([input_pose_tokens, placeholder_tokens])
        
        # Create labels: only target tokens (for loss computation)
        # We only want to compute loss on target pose tokens, not input tokens
        extended_labels = target_pose_tokens
        
        pose_tokens = torch.tensor(extended_pose_tokens, dtype=torch.long)
        pose_labels = torch.tensor(extended_labels, dtype=torch.long)
        
        # Ensure proper dimensions
        if pose_tokens.dim() == 1:
            pose_tokens = pose_tokens.unsqueeze(0)
        if pose_labels.dim() == 1:
            pose_labels = pose_labels.unsqueeze(0)
        
        # Create attention mask for extended pose tokens
        pose_attention_mask = torch.ones_like(pose_tokens, dtype=torch.long)
        
        # print(f"Input pose tokens length: {len(input_pose_tokens)}")
        # print(f"Target pose tokens length: {len(target_pose_tokens)}")
        # print(f"Extended pose tokens shape: {pose_tokens.shape}")
        # print(f"Extended labels shape: {pose_labels.shape}")
        
        return {
            'input_ids': pose_tokens,  # Pose tokens for Qwen embedding layer
            'labels': pose_labels,     # Target pose tokens for loss computation
            'attention_mask': pose_attention_mask,
            'images': input_images,    # Raw images for DINOv2 processing
        }
    
    def _extract_pose_features(self, pose):
        """Extract pose features based on config representation"""
        pose_representation = self.config['pose'].get('pose_representation', 'translation')
        
        if pose_representation == "6dof":
            # For 6DOF, pose is already [translation(3) + rotation(3)]
            features = pose  # Use the full 6D pose
        elif pose_representation == "translation":
            # Extract only translation (first 3 elements)
            features = pose[:3]
        elif pose_representation == "full_matrix":
            # Use all pose elements
            features = pose
        else:
            raise ValueError(f"Unsupported pose_representation: {pose_representation}")
        
        return features
    
    def _extract_pose_deltas(self, poses):
        """Extract translation deltas between consecutive poses"""
        if len(poses) < 2:
            # If only one pose, return zeros
            return [np.zeros(3)]
        
        deltas = []
        for i in range(1, len(poses)):
            # Compute translation delta: current - previous
            current_translation = poses[i][:3]  # First 3 elements are translation
            previous_translation = poses[i-1][:3]
            delta = current_translation - previous_translation
            deltas.append(delta)
        
        return deltas
    
    def _tokenize_poses_with_config(self, input_poses, target_poses):
        """Tokenize pose deltas using VOTrainingConfig parameters"""
        # Create quantization bins based on config
        quantization_bins = np.linspace(
            -self.config['pose']['quantization_range'],
            self.config['pose']['quantization_range'],
            self.config['model']['vocab_size']
        )
        
        # Extract pose deltas for input poses
        input_deltas = self._extract_pose_deltas(input_poses)
        
        # Extract pose deltas for target poses
        target_deltas = self._extract_pose_deltas(target_poses)
        
        # Convert to bfloat16 for consistency
        input_deltas = [delta.astype(np.float32) for delta in input_deltas]  # numpy doesn't support bfloat16
        target_deltas = [delta.astype(np.float32) for delta in target_deltas]
        
        
        # Get pose token offset from the model's tokenizer
        # This should be set when the model is initialized with pose tokens
        pose_token_offset = getattr(self, 'pose_token_offset', 1000)  # Default fallback
        if hasattr(self, 'model') and hasattr(self.model, 'pose_token_start_id'):
            pose_token_offset = self.model.pose_token_start_id
        pose_vocab_size = self.config['model']['vocab_size']
        
        # Tokenize input deltas
        input_tokens = []
        for delta in input_deltas:
            # Quantize delta features
            quantized = np.digitize(delta, quantization_bins) - 1
            quantized = np.clip(quantized, 0, pose_vocab_size - 1)
            # Offset to pose token range
            quantized = quantized + pose_token_offset
            input_tokens.append(quantized.astype(np.int32))
        
        # Tokenize target deltas
        target_tokens = []
        for delta in target_deltas:
            # Quantize delta features
            quantized = np.digitize(delta, quantization_bins) - 1
            quantized = np.clip(quantized, 0, pose_vocab_size - 1)
            # Offset to pose token range
            quantized = quantized + pose_token_offset
            target_tokens.append(quantized.astype(np.int32))
        
        # Flatten tokens
        input_tokens_flat = np.concatenate([tokens.flatten() for tokens in input_tokens])
        target_tokens_flat = np.concatenate([tokens.flatten() for tokens in target_tokens])
        
        return input_tokens_flat, target_tokens_flat