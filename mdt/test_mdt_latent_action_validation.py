#!/usr/bin/env python3
"""
Test script for validating MDTLatentActionAgent model performance.
This script loads a trained model and tests:
1. Latent action generation consistency with ground truth
2. Image generation quality using pretrained decoder
3. Forward pass validation without training
"""

import os
import sys
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from mdt.models.mdt_3d_latent_action import MDT3dLatentActionAgent
from mdt.utils.data_loading import load_dataset_calvin
from mdt.models.latent_motion_decoder import LatentMotionDecoder

logger = logging.getLogger(__name__)


class MDTValidationTester:
    """
    Tester class for validating MDT model performance without training.
    """
    
    def __init__(
        self,
        model_ckpt_path: str,
        config_path: str,
        decoder_ckpt_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4,
        max_batches: int = 10,
    ):
        self.model_ckpt_path = model_ckpt_path
        self.config_path = config_path
        self.decoder_ckpt_path = decoder_ckpt_path
        self.device = device
        self.batch_size = batch_size
        self.max_batches = max_batches
        
        # Load configuration
        self.cfg = self._load_config()
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize decoder if provided
        self.decoder = self._load_decoder() if decoder_ckpt_path else None
        
        # Setup data module
        self.datamodule = self._setup_datamodule()
        
        # Results storage
        self.results = {
            'latent_action_similarities': [],
            'ground_truth_latent_actions': [],
            'predicted_latent_actions': [],
            'generated_images': [],
            'batch_losses': [],
            'validation_metrics': {}
        }
    
    def _load_config(self) -> DictConfig:
        """Load configuration from file."""
        try:
            if self.config_path.endswith('.yaml'):
                cfg = OmegaConf.load(self.config_path)
            else:
                # If config_path is a directory, look for config.yaml
                config_file = Path(self.config_path) / "config.yaml"
                if config_file.exists():
                    cfg = OmegaConf.load(config_file)
                else:
                    raise FileNotFoundError(f"Config file not found: {config_file}")
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return cfg
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _load_model(self) -> MDT3dLatentActionAgent:
        """Load the trained MDT model from checkpoint."""
        try:
            # Load model from checkpoint
            model = MDT3dLatentActionAgent.load_from_checkpoint(
                self.model_ckpt_path,
                strict=False,
                map_location=self.device
            )
            model.eval()
            model.to(self.device)
            
            # Print model setup report
            model.print_model_setup_report()
            
            logger.info(f"Loaded model from {self.model_ckpt_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_decoder(self) -> Optional[LatentMotionDecoder]:
        """Load the pretrained latent motion decoder."""
        try:
            # Initialize decoder with appropriate config
            decoder = LatentMotionDecoder(
                latent_motion_dim=self.model.latent_motion_dim,
                output_channels=3,  # RGB
                image_size=224,
                num_layers=4,
            )
            
            # Load pretrained weights
            checkpoint = torch.load(self.decoder_ckpt_path, map_location=self.device)
            decoder.load_state_dict(checkpoint, strict=False)
            decoder.eval()
            decoder.to(self.device)
            
            logger.info(f"Loaded decoder from {self.decoder_ckpt_path}")
            return decoder
        except Exception as e:
            logger.warning(f"Failed to load decoder: {e}")
            return None
    
    def _setup_datamodule(self):
        """Setup the data module using existing configuration."""
        try:
            # Use hydra to instantiate datamodule from config
            datamodule = hydra.utils.instantiate(self.cfg.datamodule)
            datamodule.batch_size = self.batch_size
            datamodule.setup("test")
            
            logger.info("Setup datamodule successfully")
            return datamodule
        except Exception as e:
            logger.error(f"Failed to setup datamodule: {e}")
            raise
    
    @torch.no_grad()
    def run_validation_test(self):
        """
        Run validation test to compare predicted vs ground truth latent actions.
        """
        logger.info("Starting validation test...")
        
        # Get validation dataloader
        val_dataloader = self.datamodule.val_dataloader()
        
        batch_count = 0
        total_similarity = 0.0
        
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_count >= self.max_batches:
                break
                
            try:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Run validation step to get predicted latent actions
                val_results = self._run_single_validation_batch(batch, batch_idx)
                
                # Calculate similarity metrics
                similarity_metrics = self._calculate_latent_action_similarity(val_results)
                
                # Store results
                self.results['latent_action_similarities'].append(similarity_metrics)
                self.results['batch_losses'].append(val_results.get('loss', 0.0))
                
                total_similarity += similarity_metrics.get('cosine_similarity', 0.0)
                batch_count += 1
                
                logger.info(f"Batch {batch_idx}: Cosine similarity = {similarity_metrics.get('cosine_similarity', 0.0):.4f}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Calculate average metrics
        avg_similarity = total_similarity / batch_count if batch_count > 0 else 0.0
        self.results['validation_metrics']['avg_cosine_similarity'] = avg_similarity
        
        logger.info(f"Validation test completed. Average cosine similarity: {avg_similarity:.4f}")
    
    @torch.no_grad()
    def run_forward_test(self):
        """
        Run forward test to validate model inference capabilities.
        """
        logger.info("Starting forward test...")
        
        # Get test dataloader or use validation dataloader
        try:
            test_dataloader = self.datamodule.test_dataloader()
        except:
            test_dataloader = self.datamodule.val_dataloader()
        
        batch_count = 0
        
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_count >= self.max_batches:
                break
                
            try:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Run forward pass
                forward_results = self._run_single_forward_batch(batch, batch_idx)
                
                # Generate images if decoder is available
                if self.decoder and forward_results.get('predicted_latent_actions') is not None:
                    generated_images = self._generate_images_from_latent_actions(
                        forward_results['predicted_latent_actions']
                    )
                    self.results['generated_images'].append(generated_images)
                
                batch_count += 1
                logger.info(f"Forward test batch {batch_idx} completed")
                
            except Exception as e:
                logger.error(f"Error in forward test batch {batch_idx}: {e}")
                continue
        
        logger.info("Forward test completed")
    
    def _move_batch_to_device(self, batch):
        """Move batch data to the specified device."""
        def move_to_device(item):
            if isinstance(item, torch.Tensor):
                return item.to(self.device)
            elif isinstance(item, dict):
                return {k: move_to_device(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [move_to_device(item) for item in item]
            else:
                return item
        
        return move_to_device(batch)
    
    @torch.no_grad()
    def _run_single_validation_batch(self, batch, batch_idx):
        """Run validation on a single batch."""
        results = {}
        
        for modality_scope, dataset_batch in batch.items():
            self.model.modality_scope = modality_scope
            
            # Compute input embeddings
            perceptual_emb, latent_goal, image_latent_goal = self.model.compute_input_embeddings(dataset_batch)
            
            # Generate ground truth latent motion indices (if latent motion prediction is enabled)
            if self.model.latent_motion_pred and self.model.use_pretrained_encoder:
                gt_latent_motion_indices = self._generate_ground_truth_latent_actions(dataset_batch)
                
                # Generate predicted latent motion
                combined_features, combined_mask = self.model.compute_latent_goal_embeddings_and_mask(dataset_batch)
                motion_results = self.model.motion_transformer(
                    perceptual_features=combined_features,
                    latent_motion_ids=gt_latent_motion_indices,
                    attention_mask=combined_mask,
                    train=False  # Inference mode
                )
                
                results['ground_truth_latent_actions'] = gt_latent_motion_indices
                results['predicted_latent_actions'] = motion_results.get('latent_motion_preds')
                results['motion_loss'] = motion_results.get('loss', 0.0)
            
            # Run standard validation step
            val_output = self.model.validation_step({modality_scope: dataset_batch}, batch_idx)
            results.update(val_output)
            
            break  # Process only first modality for simplicity
        
        return results
    
    @torch.no_grad()
    def _run_single_forward_batch(self, batch, batch_idx):
        """Run forward pass on a single batch."""
        results = {}
        
        for modality_scope, dataset_batch in batch.items():
            # Prepare observation and goal data for forward pass
            obs = {
                'rgb_obs': dataset_batch['rgb_obs']
            }
            
            if 'lang' in modality_scope:
                goal = {
                    'lang_text': dataset_batch.get('lang_text', [''])
                }
            else:
                goal = {
                    'visual_goal': dataset_batch['rgb_obs']['rgb_static'][:, -1]
                }
            
            # Run forward pass
            try:
                predicted_actions = self.model.forward(obs, goal)
                results['predicted_actions'] = predicted_actions
                
                # Get predicted latent actions if available
                if hasattr(self.model, 'motion_transformer') and self.model.latent_motion_pred:
                    # Re-run latent motion prediction for forward pass
                    self.model.modality_scope = modality_scope
                    combined_features, combined_mask = self.model.compute_latent_goal_embeddings_and_mask(dataset_batch)
                    
                    # Generate with no ground truth (pure inference)
                    motion_results = self.model.motion_transformer(
                        perceptual_features=combined_features,
                        latent_motion_ids=None,  # No ground truth
                        attention_mask=combined_mask,
                        train=False
                    )
                    results['predicted_latent_actions'] = motion_results.get('latent_motion_preds')
                
            except Exception as e:
                logger.error(f"Forward pass failed: {e}")
                results['predicted_actions'] = None
            
            break  # Process only first modality
        
        return results
    
    @torch.no_grad()
    def _generate_ground_truth_latent_actions(self, dataset_batch):
        """Generate ground truth latent motion indices using the 3D motion tokenization pipeline."""
        try:
            # Extract RGB observations
            t = dataset_batch['rgb_obs']['rgb_static'].shape[1]
            rgb_static = dataset_batch['rgb_obs']['rgb_static'][:, :-1] if t > 1 else dataset_batch['rgb_obs']['rgb_static']
            rgb_gripper = dataset_batch['rgb_obs']['rgb_gripper'][:, :-1] if t > 1 else dataset_batch['rgb_obs']['rgb_gripper']
            
            # Resize gripper images
            rgb_gripper = self.model.interpolate_img(rgb_gripper, size=(224, 224))
            
            B, T = rgb_static.shape[:2]
            
            # Flatten for batch processing
            rgb_static_flat = rgb_static.view(B * T, *rgb_static.shape[2:])
            rgb_gripper_flat = rgb_gripper.view(B * T, *rgb_gripper.shape[2:])
            
            # 1. Image encoding using pretrained ViT-MAE
            static_features = self.model.pretrained_image_encoder(rgb_static_flat).last_hidden_state
            gripper_features = self.model.pretrained_image_encoder(rgb_gripper_flat).last_hidden_state
            
            # 2. MFormer processing
            static_obs_features = self.model.pretrained_m_former(static_features).last_hidden_state
            gripper_obs_features = self.model.pretrained_m_former(gripper_features).last_hidden_state
            
            # Reshape and prepare for 3D fusion
            static_obs_features = static_obs_features.view(B, T, *static_obs_features.shape[1:])
            gripper_obs_features = gripper_obs_features.view(B, T, *gripper_obs_features.shape[1:])
            
            # 3. 3D fusion using MFormer3D
            combined_features = torch.cat([static_obs_features, gripper_obs_features], dim=2)
            combined_features_flat = combined_features.view(B, -1, combined_features.shape[-1])
            fused_features = self.model.pretrained_m_former3d(combined_features_flat).last_hidden_state
            
            # 4. VQ down resampling
            downsampled_features = self.model.pretrained_vq_down_resampler(fused_features)
            
            # 5. Vector quantization
            vq_output = self.model.pretrained_vq(downsampled_features)
            latent_motion_indices = vq_output[1]  # Get indices
            
            return latent_motion_indices
            
        except Exception as e:
            logger.error(f"Error generating ground truth latent actions: {e}")
            return None
    
    def _calculate_latent_action_similarity(self, val_results):
        """Calculate similarity metrics between predicted and ground truth latent actions."""
        metrics = {}
        
        gt_latent = val_results.get('ground_truth_latent_actions')
        pred_latent = val_results.get('predicted_latent_actions')
        
        if gt_latent is not None and pred_latent is not None:
            try:
                # Convert to same device and type
                gt_latent = gt_latent.float()
                pred_latent = pred_latent.float()
                
                # Flatten for comparison
                gt_flat = gt_latent.view(-1)
                pred_flat = pred_latent.view(-1)
                
                # Ensure same length
                min_len = min(len(gt_flat), len(pred_flat))
                gt_flat = gt_flat[:min_len]
                pred_flat = pred_flat[:min_len]
                
                # Calculate cosine similarity
                cosine_sim = F.cosine_similarity(
                    gt_flat.unsqueeze(0), 
                    pred_flat.unsqueeze(0)
                ).item()
                
                # Calculate L2 distance
                l2_distance = torch.norm(gt_flat - pred_flat).item()
                
                # Calculate accuracy (for discrete indices)
                if gt_latent.dtype == torch.long and pred_latent.dtype == torch.long:
                    accuracy = (gt_flat == pred_flat).float().mean().item()
                    metrics['accuracy'] = accuracy
                
                metrics['cosine_similarity'] = cosine_sim
                metrics['l2_distance'] = l2_distance
                
            except Exception as e:
                logger.error(f"Error calculating similarity: {e}")
                metrics['cosine_similarity'] = 0.0
                metrics['l2_distance'] = float('inf')
        
        return metrics
    
    @torch.no_grad()
    def _generate_images_from_latent_actions(self, latent_actions):
        """Generate images using the pretrained decoder."""
        if self.decoder is None or latent_actions is None:
            return None
        
        try:
            # Generate images from latent actions
            generated_images = self.decoder(latent_actions)
            return generated_images.cpu()
        except Exception as e:
            logger.error(f"Error generating images: {e}")
            return None
    
    def save_results(self, output_dir: str = "./test_results"):
        """Save test results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, "validation_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write("MDT Latent Action Validation Results\n")
            f.write("=" * 40 + "\n")
            
            if self.results['validation_metrics']:
                for key, value in self.results['validation_metrics'].items():
                    f.write(f"{key}: {value}\n")
            
            f.write(f"\nTotal batches processed: {len(self.results['latent_action_similarities'])}\n")
            
            if self.results['latent_action_similarities']:
                cosine_sims = [m.get('cosine_similarity', 0.0) for m in self.results['latent_action_similarities']]
                l2_distances = [m.get('l2_distance', 0.0) for m in self.results['latent_action_similarities']]
                
                f.write(f"Cosine similarity - Mean: {np.mean(cosine_sims):.4f}, Std: {np.std(cosine_sims):.4f}\n")
                f.write(f"L2 distance - Mean: {np.mean(l2_distances):.4f}, Std: {np.std(l2_distances):.4f}\n")
        
        # Save generated images if available
        if self.results['generated_images']:
            img_dir = os.path.join(output_dir, "generated_images")
            os.makedirs(img_dir, exist_ok=True)
            
            for idx, images in enumerate(self.results['generated_images']):
                if images is not None:
                    # Save as numpy arrays or images
                    np.save(os.path.join(img_dir, f"batch_{idx}_generated.npy"), images.numpy())
        
        logger.info(f"Results saved to {output_dir}")
    
    def run_full_test(self):
        """Run both validation and forward tests."""
        logger.info("Starting full MDT validation test...")
        
        try:
            # Run validation test
            self.run_validation_test()
            
            # Run forward test
            self.run_forward_test()
            
            # Save results
            self.save_results()
            
            logger.info("Full test completed successfully!")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise


def main():
    """Main function to run the validation test."""
    
    # Configuration - Adjust these paths according to your setup
    MODEL_CKPT_PATH = "/home/hlwang/mdt_policy/checkpoints/your_trained_model.ckpt"
    CONFIG_PATH = "/home/hlwang/mdt_policy/conf/config_latent.yaml"  # or your config directory
    DECODER_CKPT_PATH = None  # "/path/to/your/decoder.ckpt" if available
    
    # Validate paths
    if not os.path.exists(MODEL_CKPT_PATH):
        logger.error(f"Model checkpoint not found: {MODEL_CKPT_PATH}")
        logger.info("Please update MODEL_CKPT_PATH to point to your trained model checkpoint")
        return
    
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config file not found: {CONFIG_PATH}")
        logger.info("Please update CONFIG_PATH to point to your configuration file")
        return
    
    # Initialize tester
    tester = MDTValidationTester(
        model_ckpt_path=MODEL_CKPT_PATH,
        config_path=CONFIG_PATH,
        decoder_ckpt_path=DECODER_CKPT_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=4,
        max_batches=10,
    )
    
    # Run tests
    tester.run_full_test()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()