"""Latent Motion Tokenizer model."""
import torch
import torch.nn.functional as F
from torch import nn
import lpips
from einops import rearrange
from transformers import ViTMAEModel
from PIL import Image
from torchvision import transforms as T
import time
from collections import OrderedDict
from unimatch.unimatch.unimatch import UniMatch
from typing import List, Dict, Optional, Tuple, Union, Any
from unimatch.utils.flow_viz import flow_to_image_torch
import transformers
import os

class LatentMotionTokenizer(nn.Module):
    def __init__(
            self,
            # image_encoder,
            m_former,
            vector_quantizer,
            decoder,
            hidden_state_decoder=None,
            codebook_dim=32,
            commit_loss_w=1.,
            recon_loss_w=1.,
            recon_hidden_loss_w=1.,
            perceptual_loss_w=1.,
            use_abs_recons_loss=False,
            use_optical_flow=False,
            optical_flow_model=None,
            flow_checkpoint_path=None,
            compute_flow_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        codebook_embed_dim = codebook_dim
        decoder_hidden_size = decoder.config.hidden_size
        m_former_hidden_size = m_former.config.hidden_size
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        vit_repo_root = os.path.join(hf_home, "hub", "models--facebook--vit-mae-large")
        vit_snapshots = os.path.join(vit_repo_root, "snapshots")
        vit_local_dir = None
        try:
            if os.path.isdir(vit_snapshots):
                candidates = sorted(
                    [d for d in os.listdir(vit_snapshots) if os.path.isdir(os.path.join(vit_snapshots, d))]
                )
                if candidates:
                    vit_local_dir = os.path.join(vit_snapshots, candidates[-1])
        except Exception:
            vit_local_dir = None
        
        if vit_local_dir and os.path.isdir(vit_local_dir):
            image_encoder = ViTMAEModel.from_pretrained(vit_local_dir, local_files_only=True)
        else:
            # Fallback to HuggingFace hub (respects HF_ENDPOINT / mirrors if set)
            image_encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-large")

        if isinstance(image_encoder, ViTMAEModel):
            image_encoder.config.mask_ratio = 0.0

        self.image_encoder = image_encoder.requires_grad_(False).eval()

        self.m_former = m_former

        self.vector_quantizer = vector_quantizer
        self.vq_down_resampler = nn.Sequential(
            nn.Linear(m_former_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size, codebook_embed_dim)
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, decoder_hidden_size)
        )

        self.decoder = decoder
        self.hidden_state_decoder = hidden_state_decoder

        self.commit_loss_w = commit_loss_w
        self.recon_loss_w = recon_loss_w
        self.recon_hidden_loss_w = recon_hidden_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()
        self.use_abs_recons_loss = use_abs_recons_loss
        self.use_optical_flow = use_optical_flow
        self.compute_flow_kwargs = compute_flow_kwargs
        if self.use_optical_flow:
            assert optical_flow_model is not None, "Optical flow model must be provided if use_optical_flow is True."
            assert flow_checkpoint_path is not None, "Flow checkpoint path must be provided if use_optical_flow is True."
            self.optical_flow_model = optical_flow_model
            if isinstance(optical_flow_model, UniMatch):
                self.optical_flow_model.load_state_dict(torch.load(flow_checkpoint_path, map_location='cpu'), strict=False)
            self.optical_flow_model.requires_grad_(False).eval()
            

    @property
    def device(self):
        return next(self.parameters()).device


    def get_state_dict_to_save(self):
        modules_to_exclude = ['loss_fn_lpips', 'image_encoder']
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict


    @torch.no_grad()
    def decode_image(self, cond_pixel_values, given_motion_token_ids, **kwargs):
        quant = self.vector_quantizer.get_codebook_entry(given_motion_token_ids)
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(cond_input=cond_pixel_values, latent_motion_tokens=latent_motion_tokens_up, **kwargs)
        return  {
            "recons_pixel_values": recons_pixel_values,
        }


    @torch.no_grad()
    def embed(self, cond_pixel_values, target_pixel_values, pool=False, before_vq=False, avg=False):
        quant, *_ = self.tokenize(cond_pixel_values, target_pixel_values, before_vq=before_vq)
        if pool:
            latent_motion_tokens_up = self.vq_up_resampler(quant)
            flat_latent_motion_tokens_up = latent_motion_tokens_up.reshape(latent_motion_tokens_up.shape[0], -1)
            pooled_embeddings = self.decoder.transformer.embeddings.query_pooling_layer(flat_latent_motion_tokens_up)
            return pooled_embeddings
        elif avg:
            return quant.mean(dim=1)
        else:
            return quant.reshape(quant.shape[0], -1)

    def tokenize(self, cond_pixel_values, target_pixel_values, before_vq=False):
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state

        query_num = self.m_former.query_num
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]

        if before_vq:
            return latent_motion_tokens, None, None
        else:
            latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
            quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
            return quant, indices, commit_loss


    def forward(self, cond_pixel_values, target_pixel_values,
                return_recons_only=False, 
                return_motion_token_ids_only=False,
                **kwargs): 

        # Tokenization
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state
        
        if self.use_optical_flow:
            assert self.compute_flow_kwargs is not None, "compute_flow_kwargs must be provided when use_optical_flow is True."
            flow_pixel_values = self.optical_flow_model(
                cond_pixel_values,
                target_pixel_values,
                **self.compute_flow_kwargs
            )['flow_preds'][-1]
        
            flow_pixel_values = rearrange(flow_pixel_values, 'b c h w -> b h w c')
            
            flow_rgb = flow_to_image_torch(flow_pixel_values)
            flow_rgb = rearrange(flow_rgb, 'b h w c -> b c h w')
            flow_hidden_states = self.image_encoder(flow_rgb).last_hidden_state

        query_num = self.m_former.query_num
        if self.use_optical_flow:
            latent_motion_tokens = self.m_former(
                cond_hidden_states=cond_hidden_states,
                target_hidden_states=target_hidden_states,
                flow_hidden_states=flow_hidden_states).last_hidden_state[:, :query_num]
        else:
            # Use the original MFormer without optical flow
            latent_motion_tokens = self.m_former(
                cond_hidden_states=cond_hidden_states,
                target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]
        # latent_motion_tokens = self.m_former(
        #     cond_hidden_states=cond_hidden_states,
        #     target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]

        latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
        quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
        
        # quant, indices, commit_loss = self.tokenize(cond_pixel_values, target_pixel_values)

        if return_motion_token_ids_only:
            return indices # (bs, motion_query_num)

        # Detokenization
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(
            cond_input=cond_pixel_values,
            latent_motion_tokens=latent_motion_tokens_up,
            **kwargs
        )
            
        if return_recons_only:
            return {
                "recons_pixel_values": recons_pixel_values,
                "indices": indices
            }

        if self.hidden_state_decoder is not None:
            recons_hidden_states = self.hidden_state_decoder(
                cond_input = cond_hidden_states,
                latent_motion_tokens=latent_motion_tokens_up
            )

        # Compute loss
        outputs = {
            "loss": torch.zeros_like(commit_loss),
            "commit_loss": commit_loss,
            "recons_loss": torch.zeros_like(commit_loss),
            "recons_hidden_loss": torch.zeros_like(commit_loss),
            "perceptual_loss": torch.zeros_like(commit_loss)
        }

        if self.use_abs_recons_loss:
            recons_loss = torch.abs(recons_pixel_values - target_pixel_values).mean()
        else:
            recons_loss = F.mse_loss(target_pixel_values, recons_pixel_values)
        outputs["recons_loss"] = recons_loss

        if self.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss = self.loss_fn_lpips.forward(
                    target_pixel_values, recons_pixel_values, normalize=True).mean()
        else:
            perceptual_loss = torch.zeros_like(recons_loss)
        outputs["perceptual_loss"] = perceptual_loss

        loss =  self.commit_loss_w * outputs["commit_loss"] + self.recon_loss_w * outputs["recons_loss"] + \
                self.perceptual_loss_w * outputs["perceptual_loss"]
        
        if self.hidden_state_decoder is not None:
            recon_hidden_loss = F.mse_loss(target_hidden_states, recons_hidden_states)
            outputs['recons_hidden_loss'] = recon_hidden_loss
            loss += self.recon_hidden_loss_w * outputs['recons_hidden_loss']

        outputs["loss"] = loss

        # active_code_num = torch.tensor(len(set(indices.long().reshape(-1).cpu().numpy().tolist()))).float().to(loss.device)
        active_code_num = torch.tensor(torch.unique(indices).shape[0]).float().to(loss.device)
        outputs["active_code_num"] = active_code_num

        return outputs

class PairedLatentMotionTokenizer(nn.Module):
    def __init__(
            self,
            image_encoder,
            m_former,
            vector_quantizer,
            decoder,
            hidden_state_decoder=None,
            codebook_dim=32,
            commit_loss_w=1.,
            recon_loss_w=1.,
            recon_hidden_loss_w=1.,
            perceptual_loss_w=1.,
            use_abs_recons_loss=False,
            use_optical_flow=False,
            optical_flow_model=None,
            flow_checkpoint_path=None,
            compute_flow_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        codebook_embed_dim = codebook_dim
        decoder_hidden_size = decoder.config.hidden_size
        m_former_hidden_size = m_former.config.hidden_size

        if isinstance(image_encoder, ViTMAEModel):
            image_encoder.config.mask_ratio = 0.0

        self.image_encoder = image_encoder.requires_grad_(False).eval()

        self.m_former = m_former

        self.vector_quantizer = vector_quantizer
        self.vq_down_resampler = nn.Sequential(
            nn.Linear(m_former_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size, codebook_embed_dim)
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, decoder_hidden_size)
        )

        self.decoder = decoder
        self.hidden_state_decoder = hidden_state_decoder

        self.commit_loss_w = commit_loss_w
        self.recon_loss_w = recon_loss_w
        self.recon_hidden_loss_w = recon_hidden_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()
        self.use_abs_recons_loss = use_abs_recons_loss
        self.use_optical_flow = use_optical_flow
        self.compute_flow_kwargs = compute_flow_kwargs
        if self.use_optical_flow:
            assert optical_flow_model is not None, "Optical flow model must be provided if use_optical_flow is True."
            assert flow_checkpoint_path is not None, "Flow checkpoint path must be provided if use_optical_flow is True."
            self.optical_flow_model = optical_flow_model
            if isinstance(optical_flow_model, UniMatch):
                self.optical_flow_model.load_state_dict(torch.load(flow_checkpoint_path, map_location='cpu'), strict=False)
            self.optical_flow_model.requires_grad_(False).eval()


    @property
    def device(self):
        return next(self.parameters()).device


    def get_state_dict_to_save(self):
        modules_to_exclude = ['loss_fn_lpips', 'image_encoder']
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict


    @torch.no_grad()
    def decode_image(self, cond_pixel_values, given_motion_token_ids, **kwargs):
        quant = self.vector_quantizer.get_codebook_entry(given_motion_token_ids)
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(cond_input=cond_pixel_values, latent_motion_tokens=latent_motion_tokens_up, **kwargs)
        return  {
            "recons_pixel_values": recons_pixel_values,
        }


    @torch.no_grad()
    def embed(self, cond_pixel_values, target_pixel_values, pool=False, before_vq=False, avg=False):
        quant, *_ = self.tokenize(cond_pixel_values, target_pixel_values, before_vq=before_vq)
        if pool:
            latent_motion_tokens_up = self.vq_up_resampler(quant)
            flat_latent_motion_tokens_up = latent_motion_tokens_up.reshape(latent_motion_tokens_up.shape[0], -1)
            pooled_embeddings = self.decoder.transformer.embeddings.query_pooling_layer(flat_latent_motion_tokens_up)
            return pooled_embeddings
        elif avg:
            return quant.mean(dim=1)
        else:
            return quant.reshape(quant.shape[0], -1)

    def tokenize(self, cond_pixel_values, target_pixel_values, before_vq=False):
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state

        query_num = self.m_former.query_num
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]

        if before_vq:
            return latent_motion_tokens, None, None
        else:
            latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
            quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
            return quant, indices, commit_loss


    def forward(self, 
                cond_pixel_values1,  # 第一组条件图像
                target_pixel_values1,  # 第一组目标图像（用于生成latent tokens）
                cond_pixel_values2,  # 第二组条件图像
                target_pixel_values2,  # 第二组目标图像（用于计算重建损失）
                return_recons_only=False,
                return_motion_token_ids_only=False,
                return_latent_motion_embeddings=False,
                **kwargs):

        # ================== 第一阶段：用第一组数据生成latent tokens ==================
        with torch.no_grad():
            # 计算第一组数据的隐藏状态
            cond1_hidden = self.image_encoder(cond_pixel_values1).last_hidden_state  # (b, 197, 1024)
            target1_hidden = self.image_encoder(target_pixel_values1).last_hidden_state # (b, 197, 1024)

        if self.use_optical_flow:
            assert self.compute_flow_kwargs is not None, "compute_flow_kwargs must be provided when use_optical_flow is True."
            flow_pixel_values = self.optical_flow_model(
                cond_pixel_values1,
                target_pixel_values1,
                **self.compute_flow_kwargs
            )["flow_preds"][-1]
            
            flow_pixel_values = rearrange(flow_pixel_values, 'b c h w -> b h w c')
            flow_rgb = flow_to_image_torch(flow_pixel_values)
            flow_rgb = rearrange(flow_rgb, 'b h w c -> b c h w')
            flow_hidden_states = self.image_encoder(flow_rgb).last_hidden_state
        

        # 生成运动token
        query_num = self.m_former.query_num # 8
        
        if self.use_optical_flow:
            latent_motion_tokens = self.m_former(
                cond_hidden_states=cond1_hidden,
                target_hidden_states=target1_hidden,
                flow_hidden_states=flow_hidden_states
            ).last_hidden_state[:, :query_num]
        else:
            # 使用原始的MFormer，不使用光流
            latent_motion_tokens = self.m_former(
                cond_hidden_states=cond1_hidden,
                target_hidden_states=target1_hidden
            ).last_hidden_state[:, :query_num]
        # latent_motion_tokens = self.m_former(
        #     cond_hidden_states=cond1_hidden,
        #     target_hidden_states=target1_hidden
        # ).last_hidden_state[:, :query_num] # (b, 8, 768)

        # 量化处理
        latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
        quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down) # quant: [b 8 32] indices [b 8]

        if return_motion_token_ids_only:
            return indices

        # ================== 第二阶段：用第二组数据计算重建损失 ==================
        # 上采样量化结果
        latent_motion_tokens_up = self.vq_up_resampler(quant) # [b 8 768]
        
        # 使用第二组条件图像解码
        recons_pixel_values = self.decoder(
            cond_input=cond_pixel_values2,  # 关键修改：使用第二组条件图像
            latent_motion_tokens=latent_motion_tokens_up,
            **kwargs
        )

        if return_recons_only:
            return {"recons_pixel_values": recons_pixel_values, "indices": indices}

        # ================== 计算所有损失项 ==================
        outputs = {
            "loss": torch.zeros_like(commit_loss),
            "commit_loss": commit_loss,
            "recons_loss": torch.zeros_like(commit_loss),
            "recons_hidden_loss": torch.zeros_like(commit_loss),
            "perceptual_loss": torch.zeros_like(commit_loss)
        }

        # 重建像素损失（使用第二组目标图像）
        if self.use_abs_recons_loss:
            recons_loss = torch.abs(recons_pixel_values - target_pixel_values2).mean()
        else:
            recons_loss = F.mse_loss(target_pixel_values2, recons_pixel_values)
        outputs["recons_loss"] = recons_loss

        # 感知损失（使用第二组目标图像）
        if self.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss = self.loss_fn_lpips.forward(
                    target_pixel_values2, recons_pixel_values, normalize=True).mean()
        else:
            perceptual_loss = torch.zeros_like(recons_loss)
        outputs["perceptual_loss"] = perceptual_loss

        # 潜在空间重建损失（如果需要）
        if self.hidden_state_decoder is not None:
            # 计算第二组条件图像的隐藏状态
            with torch.no_grad():
                cond2_hidden = self.image_encoder(cond_pixel_values2).last_hidden_state
                target2_hidden = self.image_encoder(target_pixel_values2).last_hidden_state
            
            # 解码隐藏状态
            recons_hidden = self.hidden_state_decoder(
                cond_input=cond2_hidden,
                latent_motion_tokens=latent_motion_tokens_up
            )
            
            # 计算隐藏状态重建损失
            recon_hidden_loss = F.mse_loss(target2_hidden, recons_hidden)
            outputs['recons_hidden_loss'] = recon_hidden_loss

        # 总损失计算
        loss = (self.commit_loss_w * outputs["commit_loss"] + 
                self.recon_loss_w * outputs["recons_loss"] + 
                self.perceptual_loss_w * outputs["perceptual_loss"])
        
        if self.hidden_state_decoder is not None:
            loss += self.recon_hidden_loss_w * outputs['recons_hidden_loss']
        
        outputs["loss"] = loss
        outputs["active_code_num"] = torch.tensor(torch.unique(indices).shape[0]).float().to(loss.device)
        if return_latent_motion_embeddings:
            outputs["latent_motion_embeddings"] = latent_motion_tokens

        return outputs

class LatentMotionTokenizer3D(nn.Module):
    def __init__(
            self,
            # image_encoder,
            m_former,
            m_former3d,  # 用于融合双视角运动的3D Transformer
            vector_quantizer,
            decoder1,   # 视角1的解码器
            decoder2,   # 视角2的解码器
            hidden_state_decoder=None,
            codebook_dim=32,
            commit_loss_w=1.,
            recon_loss_w=1.,
            recon_hidden_loss_w=1.,
            perceptual_loss_w=1.,
            use_abs_recons_loss=False,
            use_optical_flow=False,
            optical_flow_model=None,
            flow_checkpoint_path=None,
            compute_flow_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        codebook_embed_dim = codebook_dim
        # 使用m_former3d的隐藏层大小作为基准
        m_former_hidden_size = m_former3d.config.hidden_size
        decoder_hidden_size = decoder1.config.hidden_size
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        vit_repo_root = os.path.join(hf_home, "hub", "models--facebook--vit-mae-large")
        vit_snapshots = os.path.join(vit_repo_root, "snapshots")
        vit_local_dir = None
        try:
            if os.path.isdir(vit_snapshots):
                candidates = sorted(
                    [d for d in os.listdir(vit_snapshots) if os.path.isdir(os.path.join(vit_snapshots, d))]
                )
                if candidates:
                    vit_local_dir = os.path.join(vit_snapshots, candidates[-1])
        except Exception:
            vit_local_dir = None
        
        if vit_local_dir and os.path.isdir(vit_local_dir):
            image_encoder = ViTMAEModel.from_pretrained(vit_local_dir, local_files_only=True)
        else:
            # Fallback to HuggingFace hub (respects HF_ENDPOINT / mirrors if set)
            image_encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-large")

        if isinstance(image_encoder, ViTMAEModel):
            image_encoder.config.mask_ratio = 0.0

        # 冻结图像编码器
        self.image_encoder = image_encoder.requires_grad_(False).eval()
        
        # 定义2个Transformer模块
        self.m_former = m_former
        self.m_former3d = m_former3d
        
        self.vector_quantizer = vector_quantizer
        
        # 降维和升维模块
        self.vq_down_resampler = nn.Sequential(
            nn.Linear(m_former_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size, codebook_embed_dim)
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, decoder_hidden_size)
        )

        # 两个解码器
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        
        # 可选的隐藏状态解码器
        self.hidden_state_decoder = hidden_state_decoder

        # 损失权重
        self.commit_loss_w = commit_loss_w
        self.recon_loss_w = recon_loss_w
        self.recon_hidden_loss_w = recon_hidden_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        
        # 感知损失函数
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()
        
        # 重建损失类型
        self.use_abs_recons_loss = use_abs_recons_loss
        
        # 光流相关设置
        self.use_optical_flow = use_optical_flow
        self.compute_flow_kwargs = compute_flow_kwargs
        if self.use_optical_flow:
            assert optical_flow_model is not None, "Optical flow model must be provided if use_optical_flow is True."
            assert flow_checkpoint_path is not None, "Flow checkpoint path must be provided if use_optical_flow is True."
            self.optical_flow_model = optical_flow_model
            if isinstance(optical_flow_model, UniMatch):
                self.optical_flow_model.load_state_dict(torch.load(flow_checkpoint_path, map_location='cpu'), strict=False)
            self.optical_flow_model.requires_grad_(False).eval()

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict_to_save(self):
        modules_to_exclude = ['loss_fn_lpips', 'image_encoder']
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict

    @torch.no_grad()
    def decode_image(self, cond_pixel_values, given_motion_token_ids, view=1, **kwargs):
        """
        解码图像（可以选择解码为哪个视角）
        
        参数:
            view: 1 表示视角1，2 表示视角2
        """
        quant = self.vector_quantizer.get_codebook_entry(given_motion_token_ids)
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        
        # 根据视角选择解码器
        if view == 1:
            recons_pixel_values = self.decoder1(cond_input=cond_pixel_values, 
                                               latent_motion_tokens=latent_motion_tokens_up, 
                                               **kwargs)
        else:
            recons_pixel_values = self.decoder2(cond_input=cond_pixel_values, 
                                               latent_motion_tokens=latent_motion_tokens_up, 
                                               **kwargs)
        
        return {
            "recons_pixel_values": recons_pixel_values,
        }

    @torch.no_grad()
    def embed(self, cond_pixel_values, target_pixel_values, view=1, before_vq=False, pool=False, avg=False):
        """
        提取运动特征嵌入（可以选择视角）
        
        参数:
            view: 1 表示视角1, 2 表示视角2
        """
        # 首先提取视角的运动token
        if view == 1:
            latent_motion_tokens = self._get_view_motion_tokens(cond_pixel_values, target_pixel_values, view=1)
        else:
            latent_motion_tokens = self._get_view_motion_tokens(cond_pixel_values, target_pixel_values, view=2)
        
        if before_vq:
            return latent_motion_tokens
        else:
            # 应用量化
            latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
            quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
            
            if pool:
                latent_motion_tokens_up = self.vq_up_resampler(quant)
                flat_latent_motion_tokens_up = latent_motion_tokens_up.reshape(latent_motion_tokens_up.shape[0], -1)
                pooled_embeddings = self.decoder1.transformer.embeddings.query_pooling_layer(flat_latent_motion_tokens_up)
                return pooled_embeddings
            elif avg:
                return quant.mean(dim=1)
            else:
                return quant.reshape(quant.shape[0], -1)

    def tokenize(self, cond_pixel_values1, target_pixel_values1, 
                 cond_pixel_values2, target_pixel_values2, before_vq=False):
        """
        将双视角数据转化为运动token
        """
        # 提取视角1的运动token
        latent_motion_tokens1 = self._get_view_motion_tokens(cond_pixel_values1, target_pixel_values1, view=1)
        
        # 提取视角2的运动token
        latent_motion_tokens2 = self._get_view_motion_tokens(cond_pixel_values2, target_pixel_values2, view=2)
        
        # 融合为3D运动token
        combined_tokens = torch.cat([latent_motion_tokens1, latent_motion_tokens2], dim=1)
        latent_3d_motion_tokens = self.m_former3d(combined_tokens).last_hidden_state[:, :self.m_former3d.query_num]
        
        if before_vq:
            return latent_3d_motion_tokens, None, None
        else:
            latent_3d_motion_down = self.vq_down_resampler(latent_3d_motion_tokens)
            quant, indices, commit_loss = self.vector_quantizer(latent_3d_motion_down)
            return quant, indices, commit_loss

    def _get_view_motion_tokens(self, cond_pixel_values, target_pixel_values, view=1):
        """
        提取单视角的运动token(内部方法)
        """
        with torch.no_grad():
            cond_hidden = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden = self.image_encoder(target_pixel_values).last_hidden_state
        
        if self.use_optical_flow:
            assert self.compute_flow_kwargs is not None, "compute_flow_kwargs must be provided when use_optical_flow is True."
            flow_pixel_values = self.optical_flow_model(
                cond_pixel_values,
                target_pixel_values,
                **self.compute_flow_kwargs
            )["flow_preds"][-1]

            flow_pixel_values = rearrange(flow_pixel_values, 'b c h w -> b h w c')
            flow_rgb = flow_to_image_torch(flow_pixel_values)
            flow_rgb = rearrange(flow_rgb, 'b h w c -> b c h w')
            flow_hidden_states = self.image_encoder(flow_rgb).last_hidden_state

            # 使用带光流的Transformer
            latent_motion_tokens = self.m_former(
                cond_hidden_states=cond_hidden,
                target_hidden_states=target_hidden,
                flow_hidden_states=flow_hidden_states
            ).last_hidden_state[:, :self.m_former.query_num]
        else:
            # 使用标准Transformer
            latent_motion_tokens = self.m_former(
                cond_hidden_states=cond_hidden,
                target_hidden_states=target_hidden
            ).last_hidden_state[:, :self.m_former.query_num]
        return latent_motion_tokens

    def forward(self, 
                cond_pixel_values1,  # 第一组条件图像
                target_pixel_values1,  # 第一组目标图像
                cond_pixel_values2,  # 第二组条件图像
                target_pixel_values2,  # 第二组目标图像
                return_recons_only=False,
                return_motion_token_ids_only=False,
                return_latent_motion_embeddings=False,
                **kwargs):
        
        # ================== 第一阶段：提取各视角运动token ==================
        # 提取视角1的运动token
        latent_motion_tokens1 = self._get_view_motion_tokens(
            cond_pixel_values1, target_pixel_values1, view=1)  # (b, q, d)
        
        # 提取视角2的运动token
        latent_motion_tokens2 = self._get_view_motion_tokens(
            cond_pixel_values2, target_pixel_values2, view=2)  # (b, q, d)
        
        # ================== 第二阶段：融合为3D运动token ==================
        # 拼接两个视角的运动token
        combined_tokens = torch.cat([latent_motion_tokens1, latent_motion_tokens2], dim=1)  # (b, q1+q2, d)
        
        # 通过3D融合Transformer
        latent_3d_motion_tokens = self.m_former3d(combined_tokens).last_hidden_state[:, :self.m_former3d.query_num]  # (b, q3d, d)
        
        # ================== 第三阶段：量化处理 ==================
        latent_3d_motion_down = self.vq_down_resampler(latent_3d_motion_tokens)  # (b, q3d, codebook_dim)
        quant, indices, commit_loss = self.vector_quantizer(latent_3d_motion_down)  # quant: (b, q3d, codebook_dim)
        latent_3d_motion_up = self.vq_up_resampler(quant)  # (b, q3d, d)
        
        if return_motion_token_ids_only:
            return indices

        # ================== 第四阶段：双视角重建 ==================
        # 使用共享的3D运动token重建两个视角
        recons_pixel_values1 = self.decoder1(
            cond_input=cond_pixel_values1,
            latent_motion_tokens=latent_3d_motion_up,
            **kwargs
        )  # (b, c, h, w)
        
        recons_pixel_values2 = self.decoder2(
            cond_input=cond_pixel_values2,
            latent_motion_tokens=latent_3d_motion_up,
            **kwargs
        )  # (b, c, h, w)
        
        if return_recons_only:
            return {
                "recons_pixel_values1": recons_pixel_values1,
                "recons_pixel_values2": recons_pixel_values2,
                "indices": indices
            }

        # ================== 第五阶段：计算损失 ==================
        outputs = {
            "loss": torch.zeros_like(commit_loss),
            "commit_loss": commit_loss,
            "recons_loss1": torch.zeros_like(commit_loss),  # 视角1重建损失
            "recons_loss2": torch.zeros_like(commit_loss),  # 视角2重建损失
            "perceptual_loss1": torch.zeros_like(commit_loss),
            "perceptual_loss2": torch.zeros_like(commit_loss),
            "recons_hidden_loss1": torch.zeros_like(commit_loss),
            "recons_hidden_loss2": torch.zeros_like(commit_loss)
        }

        # 视角1重建损失
        if self.use_abs_recons_loss:
            recons_loss1 = torch.abs(recons_pixel_values1 - target_pixel_values1).mean()
        else:
            recons_loss1 = F.mse_loss(target_pixel_values1, recons_pixel_values1)
        outputs["recons_loss1"] = recons_loss1
        
        # 视角2重建损失
        if self.use_abs_recons_loss:
            recons_loss2 = torch.abs(recons_pixel_values2 - target_pixel_values2).mean()
        else:
            recons_loss2 = F.mse_loss(target_pixel_values2, recons_pixel_values2)
        outputs["recons_loss2"] = recons_loss2

        # 感知损失（双视角）
        if self.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss1 = self.loss_fn_lpips.forward(
                    target_pixel_values1, recons_pixel_values1, normalize=True).mean()
                perceptual_loss2 = self.loss_fn_lpips.forward(
                    target_pixel_values2, recons_pixel_values2, normalize=True).mean()
        else:
            perceptual_loss1 = torch.zeros_like(recons_loss1)
            perceptual_loss2 = torch.zeros_like(recons_loss2)
        outputs["perceptual_loss1"] = perceptual_loss1
        outputs["perceptual_loss2"] = perceptual_loss2

        # 隐藏状态重建损失（可选）
        if self.hidden_state_decoder is not None:
            with torch.no_grad():
                cond1_hidden = self.image_encoder(cond_pixel_values1).last_hidden_state
                target1_hidden = self.image_encoder(target_pixel_values1).last_hidden_state
                cond2_hidden = self.image_encoder(cond_pixel_values2).last_hidden_state
                target2_hidden = self.image_encoder(target_pixel_values2).last_hidden_state
            
            # 解码隐藏状态（双视角）
            recons_hidden1 = self.hidden_state_decoder(
                cond_input=cond1_hidden,
                latent_motion_tokens=latent_3d_motion_up
            )
            recons_hidden2 = self.hidden_state_decoder(
                cond_input=cond2_hidden,
                latent_motion_tokens=latent_3d_motion_up
            )
            
            # 计算双视角隐藏状态重建损失
            recon_hidden_loss1 = F.mse_loss(target1_hidden, recons_hidden1)
            recon_hidden_loss2 = F.mse_loss(target2_hidden, recons_hidden2)
            recon_hidden_loss = (recon_hidden_loss1 + recon_hidden_loss2) / 2
            outputs['recons_hidden_loss'] = recon_hidden_loss

        # ================== 第六阶段：总损失计算 ==================
        # 基础损失
        loss = (self.commit_loss_w * outputs["commit_loss"] +
                self.recon_loss_w * (outputs["recons_loss1"] + outputs["recons_loss2"]) +
                self.perceptual_loss_w * (outputs["perceptual_loss1"] + outputs["perceptual_loss2"]))
        
        # 加上隐藏状态损失（如果存在）
        if self.hidden_state_decoder is not None:
            loss += self.recon_hidden_loss_w * outputs['recons_hidden_loss']
        
        outputs["loss"] = loss
        outputs["active_code_num"] = torch.tensor(torch.unique(indices).shape[0]).float().to(loss.device)
        
        if return_latent_motion_embeddings:
            outputs["latent_3d_motion_embeddings"] = latent_3d_motion_tokens
        
        return outputs