import logging
import os
from typing import Any, Dict, NamedTuple, Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch.distributed as dist
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import einops 
import torch.optim as optim
import wandb

from mdt.models.edm_diffusion.gc_sampling import *
from mdt.models.edm_diffusion.utils import append_dims
from mdt.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from mdt.models.perceptual_encoders.no_encoder import NoEncoder
from mdt.models.networks.transformers.transformer_blocks import ClipStyleProjection
from mdt.callbacks.ema import EMA
from mdt.models.perceptual_encoders.resnets import BesoResNetEncoder
from mdt.models.networks.transformers.transformer_blocks import *
from mdt.models.simplified_latent_motion_predictor import SimplifiedLatentMotionPredictor
# AutoTokenizer
from transformers import AutoTokenizer
# T5 encoder
from transformers import T5EncoderModel
# ViTMAEModel
from transformers import ViTMAEModel, ViTConfig
# vq
from mdt.models.vector_quantizer import VectorQuantizer2
# MFormer models
from mdt.models.m_former import MFormer, MFormer3D
# Latent motion decoder
from mdt.models.latent_motion_decoder import LatentMotionDecoder
# MFormer models
from mdt.models.m_former import MFormer, MFormer3D
# Latent motion decoder
from mdt.models.latent_motion_decoder import LatentMotionDecoder

logger = logging.getLogger(__name__)

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    for name, submodule in model.named_modules():
        # Adjusting the condition to capture the desired layers
        if '.' not in name or name.count('.') <= 10:  # Can be adjusted based on your model structure
            # Counting parameters including submodules
            submodule_params = sum(p.numel() for p in submodule.parameters())
            if submodule_params > 0:
                print(f"{name} - Total Params: {submodule_params}")


# class VQModule(nn.Module):
#     """Vector Quantization module for generating discrete latent codes."""
    
#     def __init__(self, codebook_size, embedding_dim, commitment_cost=0.25):
#         super().__init__()
#         self.codebook_size = codebook_size
#         self.embedding_dim = embedding_dim
#         self.commitment_cost = commitment_cost
        
#         # Initialize codebook
#         self.embedding = nn.Embedding(codebook_size, embedding_dim)
#         self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)
        
#     def forward(self, inputs):
#         """
#         Args:
#             inputs: (B, D) continuous embeddings
#         Returns:
#             quantized: (B, D) quantized embeddings
#             indices: (B,) discrete indices
#             loss: VQ loss
#         """
#         # Flatten input
#         flat_input = inputs.view(-1, self.embedding_dim)
        
#         # Calculate distances to codebook
#         distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
#                     + torch.sum(self.embedding.weight**2, dim=1)
#                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
#         # Get closest codebook entries
#         encoding_indices = torch.argmin(distances, dim=1)
#         encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=inputs.device)
#         encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
#         # Quantize
#         quantized = torch.matmul(encodings, self.embedding.weight)
#         quantized = quantized.view_as(inputs)
        
#         # Loss
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)
#         q_latent_loss = F.mse_loss(quantized, inputs.detach())
#         loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
#         # Straight through estimator
#         quantized = inputs + (quantized - inputs).detach()
        
#         return quantized, encoding_indices.view(inputs.shape[:-1]), loss

    
class MDT3dLatentActionAgent(pl.LightningModule):
    """
    The lightning module used for training.
    """
    def __init__(
        self,
        language_goal: DictConfig,
        visual_goal: DictConfig,
        img_gen: DictConfig,
        model: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        latent_dim: int = 512,
        multistep: int = 10,
        sampler_type: str = 'ddim',
        num_sampling_steps: int = 10,
        sigma_data: float = 0.5,
        sigma_min: float = 0.001,
        sigma_max: float = 80,
        noise_scheduler: str = 'exponential',
        sigma_sample_density_type: str = 'loglogistic',
        use_lr_scheduler: bool = True,
        act_window_size: int = 10,
        cont_alpha: int =1, 
        masked_beta: float = 1,
        use_distributed_clip: bool = False,
        use_text_not_embedding: bool = False,
        ckpt_path=None,
        seed: int = 42,
        # New parameters for latent motion prediction
        latent_motion_codebook_size: int = 128,
        latent_motion_dim: int = 32,
        per_latent_motion_len: int = 8,
        latent_motion_pred: bool = True,
        transformer_n_layers: int = 6,
        transformer_n_heads: int = 8,
        use_pretrained_encoder: bool = True,
        encoder_ckpt_path: Optional[str] = None,
        vq_ckpt_path: Optional[str] = None,
        # New parameters for 3D motion tokenization
        image_encoder_ckpt_path: Optional[str] = None,
        m_former_ckpt_path: Optional[str] = None,
        m_former3d_ckpt_path: Optional[str] = None,
        vq_down_resampler_ckpt_path: Optional[str] = None,
        tokenizer_ckpt_path: Optional[str] = None,
        lang_feat_dim: int = 768,
        img_feat_dim: int = 1024,
        # Soft codebook expectation for training
        use_soft_codebook_training: bool = True,
        soft_code_temp: float = 1.0,
        soft_code_detach: bool = True,
    ):
        super(MDT3dLatentActionAgent, self).__init__()
        self.latent_dim = latent_dim
        img_gen['context_dim'] = self.latent_dim 
        self.static_resnet = BesoResNetEncoder(self.latent_dim)
        self.gripper_resnet = BesoResNetEncoder(self.latent_dim)
        self.act_window_size = act_window_size
        self.gen_img = hydra.utils.instantiate(img_gen)
        self.seed = seed
        self.use_lr_scheduler = use_lr_scheduler
        
        # Goal processing attributes
        self.use_delta_goal = False  # Add missing attribute
        
        # Latent motion parameters
        self.latent_motion_codebook_size = latent_motion_codebook_size
        self.latent_motion_dim = latent_motion_dim
        self.per_latent_motion_len = per_latent_motion_len
        self.latent_motion_pred = latent_motion_pred
        self.use_pretrained_encoder = use_pretrained_encoder
        
        # GPT-style transformer for latent motion prediction
        if self.latent_motion_pred:
            
            self.motion_transformer = SimplifiedLatentMotionPredictor(
                input_dim=self.latent_dim,
                hidden_size=self.latent_dim,
                per_latent_motion_len=self.per_latent_motion_len,
                latent_motion_codebook_size=self.latent_motion_codebook_size,
                n_layers=transformer_n_layers,
                n_heads=transformer_n_heads,
                use_pos_embedding=True,
                mask_probability=0.0,
                parallel_prediction=True,
            )
            
            # Pretrained models for 3D motion tokenization pipeline
            if self.use_pretrained_encoder:
                # Load pretrained models following LatentMotionTokenizer3D approach
                self.pretrained_image_encoder = self._load_pretrained_image_encoder(image_encoder_ckpt_path)
                self.pretrained_m_former = self._load_pretrained_m_former(m_former_ckpt_path)
                self.pretrained_m_former3d = self._load_pretrained_m_former3d(m_former3d_ckpt_path)
                self.pretrained_vq_down_resampler = self._load_pretrained_vq_down_resampler(vq_down_resampler_ckpt_path)
                self.pretrained_vq = self._load_pretrained_vq(vq_ckpt_path)

                # Load all components from unified LatentMotionTokenizer3D checkpoint if provided

                if tokenizer_ckpt_path is not None:
                    self.load_from_latent_motion_tokenizer3d(tokenizer_ckpt_path)
                
                # Freeze all pretrained components
                for param in self.pretrained_image_encoder.parameters():
                    param.requires_grad = False
                for param in self.pretrained_m_former.parameters():
                    param.requires_grad = False
                for param in self.pretrained_m_former3d.parameters():
                    param.requires_grad = False
                for param in self.pretrained_vq_down_resampler.parameters():
                    param.requires_grad = False
                for param in self.pretrained_vq.parameters():
                    param.requires_grad = False
        
        # goal encoders
        self.visual_goal = hydra.utils.instantiate(visual_goal)
        self.language_goal = hydra.utils.instantiate(language_goal) if language_goal else None
        # policy network
        self.model = hydra.utils.instantiate(model)
        self.modality_scope = "vis"
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_hyperparameters()
        self.masked_beta = masked_beta
        # diffusion stuff
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        # for inference
        self.rollout_step_counter = 0
        self.multistep = multistep
        self.latent_goal = None
        self.plan = None
        self.state_recons = False
        self.cont_alpha = cont_alpha
        self.use_text_not_embedding = use_text_not_embedding
        # print_model_parameters(self.perceptual_encoder.perceiver_resampler)
        # for clip loss ground truth plot
        self.cont_loss = self.clip_auxiliary_loss
        self.cont_loss_type = 'infonce'
        self.use_distributed_clip = use_distributed_clip
        self.clip_proj = ClipStyleProjection(
            clip_style='single_token', 
            token_dim=self.latent_dim,
            clip_token_index=1,
            num_token=3,
        )
        self.clip_loss_type = 'symmetric'
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.ema_callback_idx = None
        if ckpt_path is not None:
            self.load_pretrained_parameters(ckpt_path)
        # t5 tokenizer and encoder
        self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.t5_encoder = T5EncoderModel.from_pretrained("t5-base")
        # t5 encoder freeze
        self.t5_encoder.eval()
        for param in self.t5_encoder.parameters():
            param.requires_grad = False
        # embedding layers for t5 and mae
        self.lang_feat_dim = lang_feat_dim
        self.img_feat_dim = img_feat_dim
        self.embed_lang = nn.Linear(self.lang_feat_dim, self.latent_dim)
        self.embed_mae = nn.Linear(self.img_feat_dim, self.latent_dim)
        self.embed_patch = nn.Linear(self.img_feat_dim, self.latent_dim)

        # embedding latent motion input to diffusion policy
        if self.latent_motion_pred:
            self.embed_latent_motion_input = nn.Linear(self.latent_motion_dim, self.latent_dim)
            # soft codebook config
            self.use_soft_codebook_training = use_soft_codebook_training
            self.soft_code_temp = soft_code_temp
            self.soft_code_detach = soft_code_detach

    def _load_pretrained_image_encoder(self, image_encoder_ckpt_path):
        """Load pretrained image encoder (ViT-MAE)."""
        # Initialize ViT-MAE model - try to use local cache first, fallback to HF
        try:
            model = ViTMAEModel.from_pretrained(
                "/home/yyang-infobai/.cache/huggingface/hub/models--facebook--vit-mae-large/snapshots/142cb8c25e1b1bc1769997a919aa1b5a2345a6b8"
            )
        except:
            # Fallback to HuggingFace hub
            model = ViTMAEModel.from_pretrained("facebook/vit-mae-large")
            
        model.config.mask_ratio = 0.0
        model.eval()
        
        # Load pretrained weights if checkpoint provided
        if image_encoder_ckpt_path is not None:
            checkpoint = torch.load(image_encoder_ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded image encoder from {image_encoder_ckpt_path}")
        # requires no grad
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _load_pretrained_m_former(self, m_former_ckpt_path):
        """Load pretrained m_former transformer."""
        # Initialize MFormer with config from yaml
        config = ViTConfig(
            query_num=8,
            input_hidden_size=1024,
            num_patches=197,  # include the [CLS] token
            attention_probs_dropout_prob=0.0,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            model_type="vit",
            num_attention_heads=12,
            num_hidden_layers=4,
            qkv_bias=True
        )
        
        model = MFormer(
            config=config,
            add_pooling_layer=False,
            use_flow=False
        )
        model.eval()
        
        # Load pretrained weights if checkpoint provided
        if m_former_ckpt_path is not None:
            checkpoint = torch.load(m_former_ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded m_former from {m_former_ckpt_path}")
        # requires no grad
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _load_pretrained_m_former3d(self, m_former3d_ckpt_path):
        """Load pretrained m_former3d transformer for 3D fusion."""
        # Initialize MFormer3D with config from yaml
        config = ViTConfig(
            query_num=8,  # Note: yaml shows query_num=8 for m_former3d
            input_hidden_size=1024,
            num_patches=197,  # include the [CLS] token  
            attention_probs_dropout_prob=0.0,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            model_type="vit",
            num_attention_heads=12,
            num_hidden_layers=4,
            qkv_bias=True
        )
        
        model = MFormer3D(
            config=config,
            add_pooling_layer=False
        )
        model.eval()
        
        # Load pretrained weights if checkpoint provided
        if m_former3d_ckpt_path is not None:
            checkpoint = torch.load(m_former3d_ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded m_former3d from {m_former3d_ckpt_path}")

        # requires no grad
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _load_pretrained_vq_down_resampler(self, vq_down_resampler_ckpt_path):
        """Load pretrained VQ down resampler."""
        # Initialize VQ down resampler as per LatentMotionTokenizer3D structure
        # m_former3d hidden_size (768) -> decoder hidden_size (768) -> codebook_dim (32)
        model = nn.Sequential(
            nn.Linear(768, 768),  # m_former3d hidden_size -> decoder hidden_size
            nn.Tanh(),
            nn.Linear(768, self.latent_motion_dim)    # decoder hidden_size -> codebook_dim
        )
        model.eval()
        
        # Load pretrained weights if checkpoint provided  
        if vq_down_resampler_ckpt_path is not None:
            checkpoint = torch.load(vq_down_resampler_ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded vq_down_resampler from {vq_down_resampler_ckpt_path}")

        # requires no grad
        for param in model.parameters():
            param.requires_grad = False

        return model
    
    def _load_pretrained_vq(self, vq_ckpt_path):
        """Load pretrained VQ module for generating ground truth latent motion indices."""
        # Initialize VectorQuantizer2 with config from yaml
        model = VectorQuantizer2(
            n_e=self.latent_motion_codebook_size,  # 128 from yaml
            e_dim=self.latent_motion_dim,  # codebook_dim from yaml
            beta=0.25,
            remap=None,
            sane_index_shape=True
        )
        model.eval()
        
        # Load pretrained weights if checkpoint provided
        if vq_ckpt_path is not None:
            checkpoint = torch.load(vq_ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded VQ from {vq_ckpt_path}")
        
        # requires no grad
        for param in model.parameters():
            param.requires_grad = False
            
        return model

    def load_from_latent_motion_tokenizer3d(self, tokenizer_ckpt_path):
        """
        Load pretrained components from a unified LatentMotionTokenizer3D checkpoint.
        
        Args:
            tokenizer_ckpt_path: Path to pytorch_model.bin from LatentMotionTokenizer3D
        """
        print(f"Loading components from LatentMotionTokenizer3D checkpoint: {tokenizer_ckpt_path}")
        
        # Load the unified checkpoint
        checkpoint = torch.load(tokenizer_ckpt_path, map_location='cpu')
        
        # Extract component states from unified checkpoint
        component_states = {
            # 'image_encoder': {},
            'm_former': {},
            'm_former3d': {},
            'vq_down_resampler': {},
            'vector_quantizer': {}
        }
        
        # Parse checkpoint keys and distribute to components
        for key, value in checkpoint.items():
            # if key.startswith('image_encoder.'):
            #     component_key = key.replace('image_encoder.', '')
            #     component_states['image_encoder'][component_key] = value
            if key.startswith('m_former.'):
                component_key = key.replace('m_former.', '')
                component_states['m_former'][component_key] = value
            elif key.startswith('m_former3d.'):
                component_key = key.replace('m_former3d.', '')
                component_states['m_former3d'][component_key] = value
            elif key.startswith('vq_down_resampler.'):
                component_key = key.replace('vq_down_resampler.', '')
                component_states['vq_down_resampler'][component_key] = value
            elif key.startswith('vector_quantizer.'):
                component_key = key.replace('vector_quantizer.', '')
                component_states['vector_quantizer'][component_key] = value
        
        # Load each component with extracted state dict
        # if component_states['image_encoder']:
        #     print("Loading image_encoder from unified checkpoint...")
        #     self.pretrained_image_encoder.load_state_dict(component_states['image_encoder'], strict=False)
            
        if component_states['m_former']:
            print("Loading m_former from unified checkpoint...")
            missing_keys, unexpected_keys = self.pretrained_m_former.load_state_dict(component_states['m_former'], strict=False)
            print(f"m_former missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
            
        if component_states['m_former3d']:
            print("Loading m_former3d from unified checkpoint...")
            missing_keys, unexpected_keys = self.pretrained_m_former3d.load_state_dict(component_states['m_former3d'], strict=False)
            print(f"m_former3d missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")

        if component_states['vq_down_resampler']:
            print("Loading vq_down_resampler from unified checkpoint...")
            missing_keys, unexpected_keys = self.pretrained_vq_down_resampler.load_state_dict(component_states['vq_down_resampler'], strict=False)
            print(f"vq_down_resampler missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")

        if component_states['vector_quantizer']:
            print("Loading vector_quantizer from unified checkpoint...")
            missing_keys, unexpected_keys = self.pretrained_vq.load_state_dict(component_states['vector_quantizer'], strict=False)
            print(f"vector_quantizer missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")

        # print("Successfully loaded all components from LatentMotionTokenizer3D checkpoint!")
        
        # Report loaded components
        loaded_components = [name for name, state in component_states.items() if state]
        print(f"Loaded components: {loaded_components}")

    def validate_model_setup(self) -> Dict[str, Any]:
        """
        Validate the model setup including device placement, parameter counts, and gradient requirements.
        Returns a comprehensive report for debugging.
        """
        report = {
            "devices": {},
            "parameter_counts": {},
            "gradient_requirements": {},
            "frozen_modules": [],
            "trainable_modules": [],
        }
        
        # Check main model components
        components = {
            "model": self.model,
            "static_resnet": self.static_resnet,
            "gripper_resnet": self.gripper_resnet,
            "visual_goal": self.visual_goal,
            "language_goal": self.language_goal,
            "gen_img": self.gen_img,
            "embed_lang": self.embed_lang,
            "embed_mae": self.embed_mae,
            "embed_patch": self.embed_patch,
            "clip_proj": self.clip_proj,
            "t5_encoder": self.t5_encoder,
        }
        
        # Add latent motion components if enabled
        if self.latent_motion_pred:
            components.update({
                "motion_transformer": self.motion_transformer,
                "embed_latent_motion_input": self.embed_latent_motion_input,
            })
            
            if self.use_pretrained_encoder:
                components.update({
                    "pretrained_image_encoder": self.pretrained_image_encoder,
                    "pretrained_m_former": self.pretrained_m_former,
                    "pretrained_m_former3d": self.pretrained_m_former3d,
                    "pretrained_vq_down_resampler": self.pretrained_vq_down_resampler,
                    "pretrained_vq": self.pretrained_vq,
                })
        
        for name, module in components.items():
            if module is not None:
                # Get device info
                try:
                    first_param = next(module.parameters())
                    report["devices"][name] = str(first_param.device)
                except StopIteration:
                    report["devices"][name] = "No parameters"
                
                # Count parameters
                total_params = sum(p.numel() for p in module.parameters())
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                report["parameter_counts"][name] = {
                    "total": total_params,
                    "trainable": trainable_params,
                    "frozen": total_params - trainable_params
                }
                
                # Check gradient requirements
                has_grad = any(p.requires_grad for p in module.parameters())
                report["gradient_requirements"][name] = has_grad
                
                if has_grad:
                    report["trainable_modules"].append(name)
                else:
                    report["frozen_modules"].append(name)
        
        # Additional checks for single parameters
        single_params = {"logit_scale": self.logit_scale}
        for name, param in single_params.items():
            if param is not None:
                report["devices"][name] = str(param.device)
                report["parameter_counts"][name] = {
                    "total": param.numel(),
                    "trainable": param.numel() if param.requires_grad else 0,
                    "frozen": 0 if param.requires_grad else param.numel()
                }
                report["gradient_requirements"][name] = param.requires_grad
                
                if param.requires_grad:
                    report["trainable_modules"].append(name)
                else:
                    report["frozen_modules"].append(name)
        
        return report
    
    def print_model_setup_report(self):
        """Print a human-readable model setup report."""
        report = self.validate_model_setup()
        
        print("=" * 80)
        print("MODEL SETUP VALIDATION REPORT")
        print("=" * 80)
        
        print(f"Device Distribution:")
        for name, device in report["devices"].items():
            print(f"  {name:30} -> {device}")
        
        print(f"\nParameter Counts:")
        total_trainable = sum(info["trainable"] for info in report["parameter_counts"].values())
        total_frozen = sum(info["frozen"] for info in report["parameter_counts"].values())
        total_all = sum(info["total"] for info in report["parameter_counts"].values())
        
        print(f"  {'TOTAL TRAINABLE:':30} {total_trainable:,}")
        print(f"  {'TOTAL FROZEN:':30} {total_frozen:,}")
        print(f"  {'TOTAL ALL:':30} {total_all:,}")
        print()
        
        for name, counts in report["parameter_counts"].items():
            if counts["total"] > 0:
                print(f"  {name:30} -> Total: {counts['total']:8,} | Trainable: {counts['trainable']:8,} | Frozen: {counts['frozen']:8,}")
        
        print(f"\nTrainable Modules ({len(report['trainable_modules'])}):")
        for module in report["trainable_modules"]:
            print(f"  ✓ {module}")
        
        print(f"\nFrozen Modules ({len(report['frozen_modules'])}):")
        for module in report["frozen_modules"]:
            print(f"  ❄ {module}")
        
        print("=" * 80)

    def load_pretrained_parameters(self, ckpt_path):
        """
        Load the pretrained parameters from the provided path.
        """
        print("Loading pretrained parameters")
        checkpoint_data = torch.load(ckpt_path)
        '''if 'callbacks'''
        if "ema_weights" in checkpoint_data['callbacks']['EMA']:
            ema_weights_list = checkpoint_data['callbacks']['EMA']['ema_weights']
            
            # Convert list of tensors to a state_dict format
            ema_weights_dict = {name: ema_weights_list[i] for i, (name, _) in enumerate(self.named_parameters())}
            
            self.load_state_dict(ema_weights_dict)
            print("Successfully loaded EMA weights from checkpoint!")
        else:
            self.load_state_dict(checkpoint_data['state_dict'])
        print("Successfully loaded weights from checkpoint!")

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        # Configuration for models using transformer weight decay
        '''optim_groups = self.action_decoder.model.inner_model.get_optim_groups(
            weight_decay=self.optimizer_config.transformer_weight_decay
        )'''
        optim_groups = [
            {"params": self.model.inner_model.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ]
        optim_groups.extend([
            # {"params": self.visual_goal.parameters(), "weight_decay": self.optimizer_config.obs_encoder_weight_decay},
            {"params": self.gen_img.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": self.static_resnet.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": self.gripper_resnet.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},

        ])
        
        # Add parameters for latent motion prediction
        if self.latent_motion_pred:
            optim_groups.extend([
                {"params": self.motion_transformer.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
                {"params": self.embed_latent_motion_input.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
            ])
        
        # Add embedding layers for T5 and MAE features
        optim_groups.extend([
            {"params": self.embed_lang.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": self.embed_mae.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": self.embed_patch.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ])
        
        optim_groups.extend([
            {"params": self.clip_proj.parameters(), "weight_decay": self.optimizer_config.obs_encoder_weight_decay},
            {"params": self.logit_scale, "weight_decay":self.optimizer_config.obs_encoder_weight_decay},
        ])

        optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate, betas=self.optimizer_config.betas)

        # Optionally initialize the scheduler
        if self.use_lr_scheduler:
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def on_before_zero_grad(self, optimizer=None):
        total_grad_norm = 0.0
        total_param_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
            total_param_norm += p.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        total_param_norm = total_param_norm ** 0.5

        self.log("train/grad_norm", total_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/param_norm", total_param_norm, on_step=True, on_epoch=False, sync_dist=True)

    
    def clip_extra_forward(self, perceptual_emb, latent_goal, actions, sigmas, noise):

        self.model.train()
        noised_input = actions + noise * append_dims(sigmas, actions.ndim)
        context = self.model.forward_context_only(perceptual_emb, noised_input, latent_goal, sigmas)
        return context 
    
    def interpolate_img(self, img, size=(224, 224)):
        B = img.shape[0]
        img = rearrange(img, 'b t c h w -> (b t) c h w')
        img = F.interpolate(img, size=size, mode='bilinear', align_corners=False)
        img = rearrange(img, '(b t) c h w -> b t c h w', b=B)
        return img

    def training_step(self, batch: Dict[str, Dict], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss for the MDT Agent.
        The training loss consists of the score matching loss of the diffusion model 
        and the contrastive loss of the CLIP model for the multimodal encoder.
        
        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.
            
        Returns:
            loss tensor
        """
        total_loss, action_loss, cont_loss, id_loss, img_gen_loss, latent_motion_loss_total = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )
        encoders_dict = {}
        batch_size: Dict[str, int] = {}
        total_bs = 0
        for self.modality_scope, dataset_batch in batch.items():
            # print(f"Modality Scope: {self.modality_scope}")
            # Compute the required embeddings
            perceptual_emb, latent_goal, image_latent_goal = self.compute_input_embeddings(dataset_batch)

            # Latent motion prediction
            latent_motion_loss = torch.tensor(0.0).to(self.device)
            latent_motion_emb = None
            
            if self.latent_motion_pred:
                # Generate ground truth latent motion indices using 3D motion tokenization pipeline
                with torch.no_grad():
                    # Extract the 4 image modalities
                    rgb_static = dataset_batch["rgb_obs"]['rgb_static'][:, :-1]    # (B, T, C, H, W) - condition images for view1
                    rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'][:, :-1]  # (B, T, C, H, W) - condition images for view2
                    # reshape: 84*84-> 224*224
                    rgb_gripper = self.interpolate_img(rgb_gripper, size=(224, 224))
                    
                    if 'gen_static' in dataset_batch["rgb_obs"] and 'gen_gripper' in dataset_batch["rgb_obs"]:
                        gen_static = dataset_batch["rgb_obs"]['gen_static']        # (B, T, C, H, W) - target images for view1
                        gen_gripper = dataset_batch["rgb_obs"]['gen_gripper']      # (B, T, C, H, W) - target images for view2
                        # reshape: 112*112-> 224*224
                        gen_static = self.interpolate_img(gen_static, size=(224, 224))
                        gen_gripper = self.interpolate_img(gen_gripper, size=(224, 224))
                    else:
                        # Fallback: use current frame as target (no motion)
                        gen_static = rgb_static
                        gen_gripper = rgb_gripper
                    
                    B, T = rgb_static.shape[:2]
                    
                    # Process each timestep to generate ground truth motion indices following LatentMotionTokenizer3D pipeline
                    gt_latent_motion_indices = []
                    for t in range(T):
                        # Step 1: Encode images using pretrained image encoder
                        cond_static_encoded = self.pretrained_image_encoder(rgb_static[:, t])      # Output with last_hidden_state: (B, num_patches, hidden_dim)
                        target_static_encoded = self.pretrained_image_encoder(gen_static[:, t])    # Output with last_hidden_state: (B, num_patches, hidden_dim)
                        cond_gripper_encoded = self.pretrained_image_encoder(rgb_gripper[:, t])    # Output with last_hidden_state: (B, num_patches, hidden_dim)
                        target_gripper_encoded = self.pretrained_image_encoder(gen_gripper[:, t])  # Output with last_hidden_state: (B, num_patches, hidden_dim)
                        
                        # Step 2: Extract motion tokens for each viewpoint using m_former
                        # View 1 (static camera): condition=rgb_static, target=gen_static
                        motion_tokens_view1 = self.pretrained_m_former(
                            cond_hidden_states=cond_static_encoded.last_hidden_state,
                            target_hidden_states=target_static_encoded.last_hidden_state
                        ).last_hidden_state[:, :self.pretrained_m_former.query_num]  # (B, query_num, hidden_dim)
                        
                        # View 2 (gripper camera): condition=rgb_gripper, target=gen_gripper  
                        motion_tokens_view2 = self.pretrained_m_former(
                            cond_hidden_states=cond_gripper_encoded.last_hidden_state,
                            target_hidden_states=target_gripper_encoded.last_hidden_state
                        ).last_hidden_state[:, :self.pretrained_m_former.query_num]  # (B, query_num, hidden_dim)
                        
                        # Step 3: Fuse two viewpoints using m_former3d
                        combined_tokens = torch.cat([motion_tokens_view1, motion_tokens_view2], dim=1)  # (B, query_num*2, hidden_dim)
                        fused_3d_motion_tokens = self.pretrained_m_former3d(combined_tokens).last_hidden_state[:, :self.pretrained_m_former3d.query_num]  # (B, query_3d, hidden_dim)
                        
                        # Step 4: Down-sample and quantize using VQ
                        motion_tokens_down = self.pretrained_vq_down_resampler(fused_3d_motion_tokens)  # (B, query_3d, codebook_dim)
                        
                        # # Quantize each query token separately and collect indices
                        # query_3d = motion_tokens_down.shape[1]
                        # indices_list = []
                        # for q in range(query_3d):
                        #     _, indices_q, _ = self.pretrained_vq(motion_tokens_down[:, q])  # indices_q: (B,)
                        #     indices_list.append(indices_q)
                        # indices_t = torch.stack(indices_list, dim=1)  # (B, query_3d)
                        
                        # gt_latent_motion_indices.append(indices_t)
                        _, indices, _ = self.pretrained_vq(motion_tokens_down) # quant: (B, query_3d, codebook_dim), indices: (B, query_3d)
                        gt_latent_motion_indices.append(indices)
                    
                    # Stack all timesteps: (B, T, query_3d)  
                    gt_latent_motion_indices = torch.stack(gt_latent_motion_indices, dim=1)
                    
                    # Expand or truncate to match per_latent_motion_len
                    if self.per_latent_motion_len != gt_latent_motion_indices.shape[-1]:
                        if self.per_latent_motion_len > gt_latent_motion_indices.shape[-1]:
                            # Repeat the last dimension to match per_latent_motion_len
                            repeat_factor = self.per_latent_motion_len // gt_latent_motion_indices.shape[-1]
                            remainder = self.per_latent_motion_len % gt_latent_motion_indices.shape[-1]
                            
                            repeated_indices = gt_latent_motion_indices.repeat(1, 1, repeat_factor)
                            if remainder > 0:
                                repeated_indices = torch.cat([repeated_indices, gt_latent_motion_indices[:, :, :remainder]], dim=-1)
                            gt_latent_motion_indices = repeated_indices
                        else:
                            # Truncate if per_latent_motion_len is smaller
                            gt_latent_motion_indices = gt_latent_motion_indices[:, :, :self.per_latent_motion_len]
                
                # Prepare perceptual features for motion transformer
                # Combine static and gripper features
                combined_features, combined_mask = self.compute_latent_goal_embeddings_and_mask(dataset_batch)

                # Generate latent motion using transformer
                motion_results = self.motion_transformer(
                    perceptual_features=combined_features,
                    latent_motion_ids=gt_latent_motion_indices,
                    attention_mask=combined_mask,
                    train=True,
                    seq_len=T
                )
                
                latent_motion_loss += motion_results['loss']
                
                # Get latent motion embeddings for diffusion conditioning
                predicted_motion_logits = motion_results.get('latent_motion_preds')
                if predicted_motion_logits is not None:
                    # Option A: hard sampling (default)
                    if not self.use_soft_codebook_training:
                        B, seq_len, per_len, vocab_size = predicted_motion_logits.shape
                        logits_flat = predicted_motion_logits.view(-1, vocab_size)
                        probs = F.softmax(logits_flat, dim=-1)
                        sampled_idx_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
                        sampled_idx = sampled_idx_flat.view(B, seq_len, per_len)
                        latent_motion_emb = self.pretrained_vq.get_codebook_entry(sampled_idx)
                    else:
                        # Option B: soft expectation over codebook for stability
                        B, seq_len, per_len, vocab_size = predicted_motion_logits.shape
                        temp = max(self.soft_code_temp, 1e-6)
                        logits_flat = predicted_motion_logits.view(-1, vocab_size) / temp
                        probs = F.softmax(logits_flat, dim=-1)  # (B*seq_len*per_len, vocab)
                        codebook = self.pretrained_vq.embedding.weight  # (vocab, e_dim)
                        soft_emb_flat = probs @ codebook  # (B*seq_len*per_len, e_dim)
                        soft_emb = soft_emb_flat.view(B, seq_len, per_len, -1)
                        latent_motion_emb = soft_emb.detach() if self.soft_code_detach else soft_emb
                else:
                    latent_motion_emb = None

            # Modified diffusion loss to include latent motion conditioning
            act_loss, sigmas, noise = self.diffusion_loss(
                    perceptual_emb,
                    latent_goal,
                    dataset_batch["actions"],
                    latent_motion_emb=latent_motion_emb,
                )
            latent_encoder_emb = self.model.inner_model.latent_encoder_emb

            # Compute the masked generative foresight loss
            if not isinstance(self.gen_img, NoEncoder):
                rgb_static_goal = dataset_batch["rgb_obs"]['gen_static']
                rgb_gripper_goal = dataset_batch["rgb_obs"]['gen_gripper']
                img_gen_frame_diff = dataset_batch['future_frame_diff'] if "future_frame_diff" in dataset_batch else 3
                # combine both goal images
                rgb_pred_goal = torch.cat([rgb_static_goal, rgb_gripper_goal], dim=1)
                img_gen_embed =  latent_encoder_emb
                img_gen_loss_part = self.compute_img_gen_loss(img_gen_embed, rgb_pred_goal, 
                    img_gen_frame_diff=img_gen_frame_diff)
                img_gen_loss += img_gen_loss_part * self.masked_beta
                total_loss += img_gen_loss_part * self.masked_beta
            # use contrastive loss
            # Compute the Contrastive Latent Alignment Loss
            cont_loss_part = self.compute_contrastive_loss(
                perceptual_emb, 
                latent_goal, 
                image_latent_goal, 
                dataset_batch, 
                sigmas, 
                noise
            )
            cont_loss += self.cont_alpha * cont_loss_part
            total_loss += self.cont_alpha * cont_loss_part

            action_loss += act_loss
            total_loss += act_loss
            
            # Add latent motion loss
            if self.latent_motion_pred:
                latent_motion_loss_total += latent_motion_loss
                total_loss += latent_motion_loss
            
            batch_size[self.modality_scope] = dataset_batch["actions"].shape[0]
            total_bs += dataset_batch["actions"].shape[0]

        batch_len = len(batch)
        total_loss = total_loss / batch_len  # divide accumulated gradients by number of datasets
        cont_loss = cont_loss / batch_len
        action_loss = action_loss / batch_len
        img_gen_loss = img_gen_loss / batch_len
        latent_motion_loss_total = latent_motion_loss_total / batch_len
        
        # Log the metrics
        # self.on_before_zero_grad()
        self._log_training_metrics(action_loss, total_loss, cont_loss, img_gen_loss, latent_motion_loss_total, total_bs)
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Dict], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.
        During the validation step, the diffusion model predicts the next action sequence given the current state
        
        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.
         
        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal networks, as well as the
            episode indices.
        """
        output = {}
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        for self.modality_scope, dataset_batch in batch.items():
            # Compute the required embeddings
            perceptual_emb, latent_goal, image_latent_goal = self.compute_input_embeddings(dataset_batch)

            # Latent motion prediction (validation mode - no ground truth generation needed)
            latent_motion_emb = None
            
            if self.latent_motion_pred:
                with torch.no_grad():
                    rgb_static = dataset_batch["rgb_obs"]['rgb_static'][:, :-1]    # (B, T, C, H, W) - condition images for view1
                    rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'][:, :-1]  # (B, T, C, H, W) - condition images for view2
                    # reshape: 84*84-> 224*224
                    rgb_gripper = self.interpolate_img(rgb_gripper, size=(224, 224))
                    
                    if 'gen_static' in dataset_batch["rgb_obs"] and 'gen_gripper' in dataset_batch["rgb_obs"]:
                        gen_static = dataset_batch["rgb_obs"]['gen_static']        # (B, T, C, H, W) - target images for view1
                        gen_gripper = dataset_batch["rgb_obs"]['gen_gripper']      # (B, T, C, H, W) - target images for view2
                        # reshape: 112*112-> 224*224
                        gen_static = self.interpolate_img(gen_static, size=(224, 224))
                        gen_gripper = self.interpolate_img(gen_gripper, size=(224, 224))
                    else:
                        # Fallback: use current frame as target (no motion)
                        gen_static = rgb_static
                        gen_gripper = rgb_gripper
                    
                    B, T = rgb_static.shape[:2]
                    
                    gt_latent_motion_indices = []
                    for t in range(T):
                        # Step 1: Encode images using pretrained image encoder
                        cond_static_encoded = self.pretrained_image_encoder(rgb_static[:, t])      # Output with last_hidden_state: (B, num_patches, hidden_dim)
                        target_static_encoded = self.pretrained_image_encoder(gen_static[:, t])    # Output with last_hidden_state: (B, num_patches, hidden_dim)
                        cond_gripper_encoded = self.pretrained_image_encoder(rgb_gripper[:, t])    # Output with last_hidden_state: (B, num_patches, hidden_dim)
                        target_gripper_encoded = self.pretrained_image_encoder(gen_gripper[:, t])  # Output with last_hidden_state: (B, num_patches, hidden_dim)
                        
                        # Step 2: Extract motion tokens for each viewpoint using m_former
                        # View 1 (static camera): condition=rgb_static, target=gen_static
                        motion_tokens_view1 = self.pretrained_m_former(
                            cond_hidden_states=cond_static_encoded.last_hidden_state,
                            target_hidden_states=target_static_encoded.last_hidden_state
                        ).last_hidden_state[:, :self.pretrained_m_former.query_num]  # (B, query_num, hidden_dim)
                        
                        # View 2 (gripper camera): condition=rgb_gripper, target=gen_gripper  
                        motion_tokens_view2 = self.pretrained_m_former(
                            cond_hidden_states=cond_gripper_encoded.last_hidden_state,
                            target_hidden_states=target_gripper_encoded.last_hidden_state
                        ).last_hidden_state[:, :self.pretrained_m_former.query_num]  # (B, query_num, hidden_dim)
                        
                        # Step 3: Fuse two viewpoints using m_former3d
                        combined_tokens = torch.cat([motion_tokens_view1, motion_tokens_view2], dim=1)  # (B, query_num*2, hidden_dim)
                        fused_3d_motion_tokens = self.pretrained_m_former3d(combined_tokens).last_hidden_state[:, :self.pretrained_m_former3d.query_num]  # (B, query_3d, hidden_dim)
                        
                        # Step 4: Down-sample and quantize using VQ
                        motion_tokens_down = self.pretrained_vq_down_resampler(fused_3d_motion_tokens)  # (B, query_3d, codebook_dim)
                        
                        # # Quantize each query token separately and collect indices
                        # query_3d = motion_tokens_down.shape[1]
                        # indices_list = []
                        # for q in range(query_3d):
                        #     _, indices_q, _ = self.pretrained_vq(motion_tokens_down[:, q])  # indices_q: (B,)
                        #     indices_list.append(indices_q)
                        # indices_t = torch.stack(indices_list, dim=1)  # (B, query_3d)
                        
                        # gt_latent_motion_indices.append(indices_t)
                        _, indices, _ = self.pretrained_vq(motion_tokens_down) # quant: (B, query_3d, codebook_dim), indices: (B, query_3d)
                        gt_latent_motion_indices.append(indices)
                    
                    # Stack all timesteps: (B, T, query_3d)  
                    gt_latent_motion_indices = torch.stack(gt_latent_motion_indices, dim=1)
                    
                    # Expand or truncate to match per_latent_motion_len
                    if self.per_latent_motion_len != gt_latent_motion_indices.shape[-1]:
                        if self.per_latent_motion_len > gt_latent_motion_indices.shape[-1]:
                            # Repeat the last dimension to match per_latent_motion_len
                            repeat_factor = self.per_latent_motion_len // gt_latent_motion_indices.shape[-1]
                            remainder = self.per_latent_motion_len % gt_latent_motion_indices.shape[-1]
                            
                            repeated_indices = gt_latent_motion_indices.repeat(1, 1, repeat_factor)
                            if remainder > 0:
                                repeated_indices = torch.cat([repeated_indices, gt_latent_motion_indices[:, :, :remainder]], dim=-1)
                            gt_latent_motion_indices = repeated_indices
                        else:
                            # Truncate if per_latent_motion_len is smaller
                            gt_latent_motion_indices = gt_latent_motion_indices[:, :, :self.per_latent_motion_len]

                # Prepare perceptual features for motion transformer (same as training)
                combined_features, combined_mask = self.compute_latent_goal_embeddings_and_mask(dataset_batch)

                # For validation, we don't need ground truth latent motion indices
                # Instead, we can use a dummy/zero tensor or let the transformer predict from context
                B, T = dataset_batch["rgb_obs"]['rgb_static'][:, :-1].shape[:2]
                
                # Create dummy latent motion indices for transformer input (not used for supervision)
                dummy_latent_motion_indices = torch.zeros((B, T, self.per_latent_motion_len), 
                                                         dtype=torch.long, device=self.device)

                # Generate latent motion using transformer (validation mode)
                motion_results = self.motion_transformer(
                    perceptual_features=combined_features,
                    latent_motion_ids=dummy_latent_motion_indices,
                    attention_mask=combined_mask,
                    train=False,  # Set to False for validation
                    seq_len=T
                )
                
                # Get latent motion embeddings for diffusion conditioning
                predicted_motion_indices = motion_results.get('latent_motion_id_preds')
                if predicted_motion_indices is not None:
                    # predicted_motion_ids shape: (B, seq_len, per_latent_motion_len, vocab_size)
                    # Contains raw logits (unnormalized), both positive and negative values
                    
                    # Use same sampling strategy as inference for consistency
                    # B, seq_len, per_len, vocab_size = predicted_motion_ids.shape
                    # predicted_motion_ids_flat = predicted_motion_ids.view(-1, vocab_size)  # (B*seq_len*per_len, vocab_size)
                    
                    # # Apply softmax and multinomial sampling (same as inference)
                    # probs = F.softmax(predicted_motion_ids_flat, dim=-1)
                    # predicted_indices_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B*seq_len*per_len,)
                    # predicted_indices = predicted_indices_flat.view(B, seq_len, per_len)  # (B, seq_len, per_latent_motion_len)
                    
                    # use pretrained vq codebook and predicted indices to get embeddings
                    latent_motion_emb = self.pretrained_vq.get_codebook_entry(predicted_motion_indices)
                else:
                    latent_motion_emb = None
            
            # latent action generation loss
            gt_latent_motion_embeddings = self.pretrained_vq.get_codebook_entry(gt_latent_motion_indices)
            cosine_sim = torch.nn.CosineSimilarity(dim=-1)
            motion_sim = cosine_sim(latent_motion_emb, gt_latent_motion_embeddings).mean()
            self.log(f"val/{self.modality_scope}_latent_motion_cosine_sim", motion_sim, on_step=False, on_epoch=True, sync_dist=True)

            # predict the next action sequence with latent motion conditioning
            action_pred = self.denoise_actions(
                torch.zeros_like(latent_goal).to(latent_goal.device),
                perceptual_emb,
                latent_goal,
                inference=True,
                latent_motion_emb=latent_motion_emb,  # Add latent motion conditioning
            )
            # compute the mse action loss
            pred_loss = torch.nn.functional.mse_loss(action_pred, dataset_batch["actions"])
            latent_encoder_emb = self.model.inner_model.latent_encoder_emb
            val_total_act_loss_pp += pred_loss
            
            # next compute the image generation loss
            if not isinstance(self.gen_img, NoEncoder):
                rgb_static_goal = dataset_batch["rgb_obs"]['gen_static']
                rgb_gripper_goal = dataset_batch["rgb_obs"]['gen_gripper']
                img_gen_frame_diff = dataset_batch['future_frame_diff'] if "future_frame_diff" in dataset_batch else 3
                # combine both goal images
                rgb_pred_goal = torch.cat([rgb_static_goal, rgb_gripper_goal], dim=1)
                
                img_gen_embed = latent_encoder_emb

                img_gen_loss = self.compute_img_gen_loss(
                    img_gen_embed, 
                    rgb_pred_goal, 
                    store_img=False, 
                    batch_idx=batch_idx,
                    img_gen_frame_diff=img_gen_frame_diff,
                )
            else:
                img_gen_loss = torch.tensor(0.0).to(self.device)
            
            self._log_validation_metrics(pred_loss, img_gen_loss, val_total_act_loss_pp)

            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]
            output["validation_loss"] = val_total_act_loss_pp
        return output
    
    
    def compute_input_embeddings(self, dataset_batch):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        # 1. extract the revelant visual observations
        latent_goal = None
        rgb_static_goal = dataset_batch["rgb_obs"]['rgb_static'][:, -1]
        rgb_static = dataset_batch["rgb_obs"]['rgb_static'][:, :-1]

        rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'][:, :-1]

        # 2. Compute the latent goal embedding for the visual goal
        if not isinstance(self.visual_goal, NoEncoder):
            latent_goal = self.visual_goal(rgb_static_goal).to(rgb_static.dtype)

        lang_text = dataset_batch["lang_text"] if "lang" in self.modality_scope else None
        
        # 3. we compute the language goal if the language modality is in the scope
        if "lang" in self.modality_scope:
            image_latent_goal = latent_goal.to(rgb_static.dtype)
            if self.use_text_not_embedding:
                latent_goal = self.language_goal(dataset_batch["lang_text"]).to(rgb_static.dtype)
            else:
                latent_goal = self.language_goal(dataset_batch["lang"]).to(rgb_static.dtype)
        else:
            image_latent_goal = None

        perceptual_emb = self.embed_visual_obs(rgb_static, rgb_gripper)
        perceptual_emb['modality'] = self.modality_scope
        return perceptual_emb, latent_goal, image_latent_goal
    
    def compute_latent_goal_embeddings_and_mask(self, dataset_batch):
        """
        Compute the latent goal embeddings for the language and mae modalities.
        """
        t5_lang_emb = None
        mae_emb = None
        lang_attention_mask = None
        if "lang" in self.modality_scope:
            lang_text = dataset_batch["lang_text"]
            tokenized = self.t5_tokenizer(
                lang_text,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            lang_input_ids = tokenized.input_ids
            lang_attention_mask = tokenized.attention_mask

            # already freezed
            with torch.no_grad():
                t5_outputs = self.t5_encoder(
                    input_ids=lang_input_ids,
                    attention_mask=lang_attention_mask
                )
                t5_lang_emb = t5_outputs.last_hidden_state
            t5_lang_emb = self.embed_lang(t5_lang_emb.float()) # (b, lang_tokens, hidden_dim)
            if len(t5_lang_emb.shape) == 2:
                t5_lang_emb = t5_lang_emb.unsqueeze(1)
            if len(lang_attention_mask.shape) == 1:
                lang_attention_mask = lang_attention_mask.unsqueeze(0)
        
        # mae embeddings
        t = dataset_batch['rgb_obs']['rgb_static'].shape[1]
        rgb_static = dataset_batch['rgb_obs']['rgb_static'][:, :-1] if t > 1 else dataset_batch['rgb_obs']['rgb_static']
        rgb_gripper = dataset_batch['rgb_obs']['rgb_gripper'][:, :-1] if t > 1 else dataset_batch['rgb_obs']['rgb_gripper']
        rgb_gripper = self.interpolate_img(rgb_gripper, size=(224, 224))
        B, T = rgb_static.shape[:2]
        rgb_static_flat = rgb_static.view(B * T, *rgb_static.shape[2:])  # (B*T, C, H, W)
        rgb_gripper_flat = rgb_gripper.view(B * T, *rgb_gripper.shape[2:])  # (B*T, C, H, W)
        static_features = self.pretrained_image_encoder(rgb_static_flat).last_hidden_state  # (B*T, num_patches+1, feat_dim)
        gripper_features = self.pretrained_image_encoder(rgb_gripper_flat).last_hidden_state  # (B*T, num_patches+1, feat_dim)
        static_obs_features = static_features[:, 0, :].view(B, T, -1)  # (B, T, feat_dim)
        gripper_obs_features = gripper_features[:, 0, :].view(B, T, -1)  # (B, T, feat_dim)
        static_patch_features = static_features[:, 1:, :].view(B, -1, static_features.shape[-1])  # (B, T*num_patches, feat_dim)
        gripper_patch_features = gripper_features[:, 1:, :].view(B, -1, gripper_features.shape[-1])  # (B, T*num_patches, feat_dim)
        obs_features = torch.cat([static_obs_features, gripper_obs_features], dim=1)  # (B, 2*T, feat_dim)
        patch_features = torch.cat([static_patch_features, gripper_patch_features], dim=1)  # (B, 2*T*num_patches, feat_dim)
        obs_mae_emb = self.embed_mae(obs_features.float())  # (B, 2*T, hidden_dim)
        patch_mae_emb = self.embed_patch(patch_features.float())  # (B, 2*T*num_patches, hidden_dim)
        mae_emb = torch.cat([obs_mae_emb, patch_mae_emb], dim=1)  # (B, 2*T + 2*T*num_patches, hidden_dim)

        #attention masks
        if lang_attention_mask is not None:
            lang_mask = lang_attention_mask  # (B, lang_tokens)
            mae_mask = torch.ones((B, mae_emb.shape[1]), device=self.device)  # (B, 2*T + 2*T*num_patches)
            combined_mask = torch.cat([lang_mask, mae_mask], dim=1)  # (B, lang_tokens + 2*T + 2*T*num_patches)
        else:
            combined_mask = torch.ones((B, mae_emb.shape[1]), device=self.device)  # (B, 2*T + 2*T*num_patches)

        if t5_lang_emb is not None:
            combined_emb = torch.cat([t5_lang_emb, mae_emb], dim=1)  # (B, lang_tokens + 2*T + 2*T*num_patches, hidden_dim)
        else:
            combined_emb = mae_emb  # (B, 2*T + 2*T*num_patches, hidden_dim)
        return combined_emb, combined_mask
    
    def embed_visual_obs(self, rgb_static, rgb_gripper):
        # reshape rgb_static and rgb_gripper
        rgb_static = einops.rearrange(rgb_static, 'b t c h w -> (b t) c h w')
        rgb_gripper = einops.rearrange(rgb_gripper, 'b t c h w -> (b t) c h w')

        static_tokens = self.static_resnet(rgb_static)
        gripper_tokens = self.gripper_resnet(rgb_gripper)
        static_tokens = einops.rearrange(static_tokens, 'b (t d) -> b t d', t=1)
        gripper_tokens = einops.rearrange(gripper_tokens, 'b (t d) -> b t d', t=1)
        token_seq = {
            'static': static_tokens,
            'gripper': gripper_tokens,
        }
        return token_seq
    
    def clip_extra_forward(self, perceptual_emb, latent_goal, actions, sigmas, noise):    
        self.model.train()
        noised_input = actions + noise * append_dims(sigmas, actions.ndim)
        context = self.model.forward_context_only(perceptual_emb, noised_input, latent_goal, sigmas)
        return context

    def compute_img_gen_loss(self, latent_embeddings, goal_img, store_img=False, img_gen_frame_diff=3, batch_idx=0):
        """
        Compute the image generation loss based on the provided embeddings and dataset batch.
        """   
        if len(goal_img.shape) == 5:
            goal_img = goal_img.squeeze(1) 
        # the goal is not to reconstruct all the details but to get the general shape
        # 1. predict the future image patches
        img_gen_pred, mask, restore_idxs, visible_patches = self.gen_img(latent_embeddings, goal_img, img_gen_frame_diff)
        # 2. compute the loss
        img_gen_loss = self.gen_img.compute_loss(goal_img, img_gen_pred, mask, restore_idxs)
        if store_img:
            file_path = os.getcwd() + f'/img_gen_pred_{batch_idx}.png'
            self.gen_img.reconstruct_image(
                predictions=img_gen_pred, 
                goal_images=goal_img,
                mask=mask,
                restore_idxs=restore_idxs,
                file_path=file_path, 
                )
            try:
                self.logger.experiment.log({f"generated_img_{batch_idx}": wandb.Image(os.path.abspath(file_path))})
            except Exception as e:
                print(f"An error occurred while saving or logging image: {e}")
                # Optionally, you can log the error to wandb as well
                self.logger.experiment.log({"error": str(e)})
                
        return img_gen_loss     

    def compute_contrastive_loss(self, perceptual_emb, latent_goal, image_latent_goal, dataset_batch, sigma,  noise):
        """
        Compute the contrastive loss based on the provided embeddings and dataset batch.
        """
        if "lang" in self.modality_scope:
            latent_language_embed = self.model.inner_model.latent_encoder_emb
            
            latent_vis_embed = self.clip_extra_forward(
                    perceptual_emb,
                    image_latent_goal,
                    dataset_batch["actions"],
                    sigma,  # Assuming you don't need sigmas and noise here
                    noise
                )
            latent_language_embed = self.clip_proj(latent_language_embed)
            latent_vis_embed = self.clip_proj(latent_vis_embed)


            is_distributed = self.trainer.global_rank >= 0 and dist.is_initialized()

            if is_distributed and self.use_distributed_clip:

               all_latent_vis_embed = self.all_gather(latent_vis_embed, sync_grads=True)
               all_latent_language_embed = self.all_gather(latent_language_embed, sync_grads=True)
               all_latent_language_embed = einops.rearrange(all_latent_language_embed, 'n b d -> (n b) d')
               all_latent_vis_embed = einops.rearrange(all_latent_vis_embed, 'n b d -> (n b) d')

            else:
                all_latent_vis_embed = latent_vis_embed
                all_latent_language_embed = latent_language_embed


            lang_text = dataset_batch["lang_text"] if "lang_text" in dataset_batch else None

            # Compute contrastive loss with gathered embeddings
            cont_loss_part = self.cont_loss(
                all_latent_vis_embed, 
                all_latent_language_embed, 
                mode=self.clip_loss_type, 
                lang_text=lang_text
            )

            return cont_loss_part 
        else:
            return torch.tensor(0.0).to(self.device)  # Return a zero tensor if "lang" is not in the modality scope

    
    def _log_training_metrics(self, action_loss, total_loss, cont_loss, img_gen_loss, latent_motion_loss, total_bs):
        """
        Log the training metrics.
        """
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True,batch_size=total_bs)
        self.log("train/cont_loss", cont_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/img_gen_loss", img_gen_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/latent_motion_loss", latent_motion_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        
    def _log_validation_metrics(self, pred_loss, img_gen_loss, val_total_act_loss_pp):
        """
        Log the validation metrics.
        """
        self.log(f"val_act/{self.modality_scope}_act_loss_pp", pred_loss, sync_dist=True)
        self.log(
            "val_act/action_loss",
            val_total_act_loss_pp / len(self.trainer.datamodule.modalities),  # type:ignore
            sync_dist=True,
        )
        self.log(f"val_act/img_gen_loss_pp", img_gen_loss, sync_dist=True)
    
    def diffusion_loss(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        latent_motion_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        Optionally includes latent motion embeddings as additional conditioning.
        """
        self.model.train()
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)
        
        # If latent motion embeddings are provided, concatenate them with latent_goal
        if self.latent_motion_pred and latent_motion_emb is not None:
            latent_motion_emb = self.embed_latent_motion_input(latent_motion_emb)
            latent_motion_emb = latent_motion_emb.view(latent_motion_emb.shape[0], -1, latent_motion_emb.shape[-1]) # (B, seq_len*per_latent_motion_len, hidden_dim)
        if latent_motion_emb is not None:
            if latent_goal is not None:
                if len(latent_goal.shape) == 2:
                    latent_goal = latent_goal.unsqueeze(1) # (B, 1, hidden_dim)
                enhanced_latent_goal = torch.cat([latent_goal, latent_motion_emb], dim=1) # (B, 1 + seq_len*per_latent_motion_len, hidden_dim)
            else:
                enhanced_latent_goal = latent_motion_emb
        else:
            enhanced_latent_goal = latent_goal
            
        loss, _ = self.model.loss(perceptual_emb, actions, enhanced_latent_goal, noise, sigmas)
        return loss, sigmas, noise
    
    def denoise_actions(  # type: ignore
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        inference: Optional[bool] = False,
        extra_args={},
        latent_motion_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the next sequence of actions 
        """
        if inference:
            sampling_steps = self.num_sampling_steps
        else:
            sampling_steps = 10
        self.model.eval()
        
        # Process latent motion embeddings if provided (same logic as diffusion_loss)
        if latent_motion_emb is not None:
            latent_motion_emb = self.embed_latent_motion_input(latent_motion_emb)
            latent_motion_emb = latent_motion_emb.view(latent_motion_emb.shape[0], -1, latent_motion_emb.shape[-1]) # (B, seq_len*per_latent_motion_len, hidden_dim)
            
            if latent_goal is not None:
                if len(latent_goal.shape) == 2:
                    latent_goal = latent_goal.unsqueeze(1) # (B, 1, hidden_dim)
                enhanced_latent_goal = torch.cat([latent_goal, latent_motion_emb], dim=1) # (B, 1 + seq_len*per_latent_motion_len, hidden_dim)
            else:
                enhanced_latent_goal = latent_motion_emb
        else:
            enhanced_latent_goal = latent_goal
            
        if len(enhanced_latent_goal.shape) < len(perceptual_emb['static'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape): 
            enhanced_latent_goal = enhanced_latent_goal.unsqueeze(1) # .expand(-1, seq_len, -1)
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)
        if len(enhanced_latent_goal.shape) == 2:
            enhanced_latent_goal = einops.rearrange(enhanced_latent_goal, 'b d -> 1 b d')

        x = torch.randn((len(enhanced_latent_goal), self.act_window_size, 7), device=self.device) * self.sigma_max

        actions = self.sample_loop(sigmas, x, input_state, enhanced_latent_goal, latent_plan, self.sampler_type, extra_args)

        return actions

    def make_sample_density(self):
        """ 
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)
        
        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)
        
        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)
        
        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps*1e5, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')

    def sample_loop(
        self, 
        sigmas, 
        x_t: torch.Tensor,
        state: torch.Tensor, 
        goal: torch.Tensor, 
        latent_plan: torch.Tensor,
        sampler_type: str,
        extra_args={}, 
        ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x:extra_args[x] for x in keys}
        else:
            reduced_args = {}
        if use_scaler:
            scaler = self.scaler
        else:
            scaler=None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min, disable=True)
        # ODE deterministic 
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True) 
        # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas), disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0    
    
    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7, self.device) # rho=7 is the default from EDM karras
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0
    
    def forward(self, obs, goal):
        """
        Method for doing inference with the model.
        """
        if 'lang' in goal:
            modality = 'lang'
            if self.use_text_not_embedding:
                # print(goal.keys())
                latent_goal = self.language_goal(goal["lang_text"])
                latent_goal = latent_goal.to(torch.float32)
            else:
                latent_goal = self.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(obs["rgb_obs"]['rgb_static'].device)
        else:
            modality = 'vis'
            if self.use_delta_goal:
                perceptual_goal_emb = self.visual_goal(obs["rgb_obs"]['rgb_static'].squeeze(0))
            else:
                perceptual_goal_emb = self.visual_goal(obs["rgb_obs"]['rgb_static'][:, -1]).unsqueeze(1) #[:, -1])
            
            latent_goal = perceptual_goal_emb
        
        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        perceptual_emb = self.embed_visual_obs(rgb_static, rgb_gripper)
        perceptual_emb['modality'] = modality

        # Latent motion prediction for inference (similar to validation)
        latent_motion_emb = None
        if self.latent_motion_pred:
            # Create a mock dataset_batch for compute_latent_goal_embeddings_and_mask
            # This is needed for inference when we don't have the full dataset_batch structure
            mock_dataset_batch = {
                'rgb_obs': obs['rgb_obs'],
                'lang_text': goal.get('lang_text', ['']),  # Provide default empty string
            }
            mock_dataset_batch['modality_scope'] = modality
            
            # Set modality scope for compute_latent_goal_embeddings_and_mask
            self.modality_scope = modality
            
            try:
                combined_features, combined_mask = self.compute_latent_goal_embeddings_and_mask(mock_dataset_batch)
                
                B, T = rgb_static[:, :-1].shape[:2] if rgb_static.shape[1] > 1 else (rgb_static.shape[0], 1)
                
                # Create dummy latent motion indices for transformer input
                dummy_latent_motion_indices = torch.zeros((B, T, self.per_latent_motion_len), 
                                                         dtype=torch.long, device=self.device)

                # Generate latent motion using transformer (inference mode)
                motion_results = self.motion_transformer(
                    perceptual_features=combined_features,
                    latent_motion_ids=dummy_latent_motion_indices,
                    attention_mask=combined_mask,
                    train=False,
                    seq_len=T
                )
                
                # Get latent motion embeddings for diffusion conditioning
                predicted_motion_ids = motion_results.get('latent_motion_id_preds')
                if predicted_motion_ids is not None:
                    # Use same sampling strategy as other inference modes
                    # B, seq_len, per_len, vocab_size = predicted_motion_ids.shape
                    # predicted_motion_ids_flat = predicted_motion_ids.view(-1, vocab_size)
                    
                    # # Apply softmax and multinomial sampling
                    # probs = F.softmax(predicted_motion_ids_flat, dim=-1)
                    # predicted_indices_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    # predicted_indices = predicted_indices_flat.view(B, seq_len, per_len)

                    latent_motion_emb = self.pretrained_vq.get_codebook_entry(predicted_motion_ids)
            except Exception as e:
                # Fallback: if latent motion prediction fails, continue without it
                print(f"Warning: Latent motion prediction failed during inference: {e}")
                latent_motion_emb = None

        act_seq = self.denoise_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            perceptual_emb,
            latent_goal,
            inference=True,
            latent_motion_emb=latent_motion_emb,
        )
        return act_seq

    def step(self, obs, goal):
        """
        Do one step of inference with the model. THis method handles the action chunking case.
        Our model is trained to predict a sequence of actions. 
        We only compute the sequence once every self.multistep steps.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        if self.rollout_step_counter % self.multistep == 0:
            pred_action_seq = self(obs, goal)

            self.pred_action_seq = pred_action_seq  
            
        current_action = self.pred_action_seq[0, self.rollout_step_counter]
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0
        
        return current_action
    
    def on_train_start(self)-> None:
        
        self.model.to(dtype=self.dtype)
        self.static_resnet.to(dtype=self.dtype)
        self.gripper_resnet.to(dtype=self.dtype)
        if self.language_goal is not None:
            self.language_goal.to(dtype=self.dtype)
        self.visual_goal.to(dtype=self.dtype)
        self.gen_img.to(dtype=self.dtype)
        
        # Handle new embedding layers
        self.embed_lang.to(dtype=self.dtype)
        self.embed_mae.to(dtype=self.dtype)
        self.embed_patch.to(dtype=self.dtype)
        
        # Handle T5 encoder (should remain frozen)
        self.t5_encoder.to(dtype=self.dtype)
        
        # Handle latent motion related modules
        if self.latent_motion_pred:
            self.motion_transformer.to(dtype=self.dtype)
            self.embed_latent_motion_input.to(dtype=self.dtype)
            
            # Pretrained models should stay in their original dtype (usually float32)
            # but ensure they are on the correct device
            if self.use_pretrained_encoder:
                self.pretrained_image_encoder.to(device=self.device)
                self.pretrained_m_former.to(device=self.device)
                self.pretrained_m_former3d.to(device=self.device)
                self.pretrained_vq_down_resampler.to(device=self.device)
                self.pretrained_vq.to(device=self.device)
        
        # Handle CLIP projection
        self.clip_proj.to(dtype=self.dtype)
        
        for idx, callback in enumerate(self.trainer.callbacks):
            if isinstance(callback, EMA):
                self.ema_callback_idx = idx
                break
    
    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")
        
    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def clip_auxiliary_loss(self, image_features, lang_features, mode='symmetric', lang_text=None):
        # Normalize the features
        image_features = F.normalize(image_features, dim=-1)
        lang_features = F.normalize(lang_features, dim=-1)
        logit_scale = self.logit_scale.exp()

        # Compute the cosine similarity
        similarity_matrix = logit_scale * image_features @ lang_features.t()

        # InfoNCE loss
        labels = torch.arange(similarity_matrix.shape[0], device=image_features.device)
        infonce_loss = F.cross_entropy(similarity_matrix, labels)

        if mode == 'symmetric':
            similarity_matrix_lang_img = logit_scale * lang_features @ image_features.t()
            # similarity_matrix_lang_img.masked_fill_(~unique_mask, float('-inf'))
            infonce_loss_lang_img = F.cross_entropy(similarity_matrix_lang_img, labels)
            infonce_loss = (infonce_loss + infonce_loss_lang_img) / 2
        elif mode == 'img_to_text':
            pass  # already computed above
        elif mode == 'text_to_img':
            similarity_matrix = similarity_matrix.t()  # transpose for text-to-image
            infonce_loss = F.cross_entropy(similarity_matrix, labels)
        else:
            raise ValueError("Invalid mode. Expected one of: 'symmetric', 'img_to_text', 'text_to_img'.")
        return infonce_loss
    
    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")
        
    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")


    
@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)
