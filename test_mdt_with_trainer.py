#!/usr/bin/env python3
"""
使用PyTorch Lightning Trainer进行MDT模型验证测试的脚本
此脚本会：
1. 加载训练好的模型
2. 使用trainer.test()或trainer.validate()进行测试
3. 比较latent action的一致性
4. 保存测试结果
"""

import os
import sys
import torch
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import logging
from typing import Dict, Any
import argparse

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class MDTTestRunner:
    """MDT模型测试运行器"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        output_dir: str = "./test_outputs",
        device: str = "auto",
        limit_test_batches: int = 10,
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.output_dir = output_dir
        self.device = self._setup_device(device)
        self.limit_test_batches = limit_test_batches
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化组件
        self.model = None
        self.datamodule = None
        self.trainer = None
        
    def _setup_device(self, device):
        """设置设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_checkpoint_and_config(self):
        """加载模型checkpoint和配置"""
        try:
            print(f"正在加载checkpoint: {self.checkpoint_path}")
            
            # 检查checkpoint是否存在
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint文件不存在: {self.checkpoint_path}")
            
            # 尝试从checkpoint中提取配置
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            if 'hyper_parameters' in checkpoint:
                # PyTorch Lightning checkpoint包含超参数
                print("检测到PyTorch Lightning checkpoint")
                
                # 动态导入模型类
                try:
                    from mdt.models.mdt_3d_latent_action import MDT3dLatentActionAgent
                    
                    # 直接从checkpoint加载模型
                    self.model = MDT3dLatentActionAgent.load_from_checkpoint(
                        self.checkpoint_path,
                        map_location='cpu',  # 先加载到CPU
                        strict=False
                    )
                    
                    # 移动模型到指定设备
                    self.model = self.model.to(self.device)
                    print(f"模型已移动到设备: {self.device}")
                    
                    # 确保模型处于评估模式
                    self.model.eval()
                    
                    print("成功加载模型")
                    self._print_model_summary()
                    
                except ImportError as e:
                    print(f"导入模型类失败: {e}")
                    print("请确保mdt包在Python路径中")
                    return False
                    
            else:
                print("检测到普通state_dict checkpoint")
                if self.config_path is None:
                    print("普通checkpoint需要配置文件，请提供config_path")
                    return False
                
                return self._load_with_config()
            
            return True
            
        except Exception as e:
            print(f"加载checkpoint失败: {e}")
            return False
    
    def _load_with_config(self):
        """使用配置文件加载模型"""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            # 加载配置
            cfg = OmegaConf.load(self.config_path)
            
            # 使用hydra实例化模型
            from mdt.models.mdt_3d_latent_action import MDT3dLatentActionAgent
            
            # 从配置中获取模型参数
            model_cfg = cfg.get('model', {})
            self.model = hydra.utils.instantiate(model_cfg)
            
            # 加载权重
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint, strict=False)
            
            print("使用配置文件成功加载模型")
            return True
            
        except Exception as e:
            print(f"使用配置文件加载失败: {e}")
            return False
    
    def _print_model_summary(self):
        """打印模型摘要"""
        if self.model is None:
            return
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"""
模型摘要:
- 总参数: {total_params:,}
- 可训练参数: {trainable_params:,}
- 冻结参数: {total_params - trainable_params:,}
- Latent Motion预测: {getattr(self.model, 'latent_motion_pred', False)}
- 使用预训练编码器: {getattr(self.model, 'use_pretrained_encoder', False)}
""")
    
    def setup_datamodule(self):
        """设置数据模块"""
        try:
            print("正在设置数据模块...")
            
            # 方法1: 使用独立的datamodule配置文件
            try:
                print("使用独立的datamodule配置文件...")
                config_path = "/home/hlwang/mdt_policy/conf/datamodule_standalone.yaml"
                
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"独立配置文件不存在: {config_path}")
                
                # 加载配置
                config = OmegaConf.load(config_path)
                
                # 实例化datamodule
                self.datamodule = hydra.utils.instantiate(config)
                
                # 设置数据模块
                self.datamodule.setup("fit")
                print("使用独立配置成功设置datamodule")
                print(f"数据路径: {config.root_data_dir}")
                print(f"模态列表: {self.datamodule.modalities}")
                return True
                
            except Exception as standalone_e:
                print(f"独立配置方法失败: {standalone_e}")
                
                # 方法2: 尝试使用完整Hydra配置（备用）
                try:
                    print("尝试备用的Hydra配置方法...")
                    from hydra import initialize, compose
                    with initialize(config_path="/home/hlwang/mdt_policy/conf"):
                        cfg = compose(config_name="config_latent")
                        self.datamodule = hydra.utils.instantiate(cfg.datamodule)
                        self.datamodule.setup("fit")
                        print("使用完整Hydra配置成功设置datamodule")
                        return True
                        
                except Exception as hydra_e:
                    print(f"Hydra方法也失败: {hydra_e}")
                    return False
            
        except Exception as e:
            print(f"设置datamodule失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return False
    
    def setup_trainer(self):
        """设置trainer"""
        try:
            # 设置日志记录器
            csv_logger = CSVLogger(
                save_dir=self.output_dir,
                name="mdt_test_logs"
            )
            
            # 配置trainer
            trainer_kwargs = {
                "logger": csv_logger,
                "enable_checkpointing": False,
                "enable_progress_bar": True,
                "enable_model_summary": False,
                "limit_test_batches": self.limit_test_batches,
                "limit_val_batches": self.limit_test_batches,
            }
            
            # 设置设备相关参数
            if self.device == "cuda":
                trainer_kwargs.update({
                    "accelerator": "gpu",
                    "devices": 1,
                })
            else:
                trainer_kwargs.update({
                    "accelerator": "cpu",
                })
            
            self.trainer = Trainer(**trainer_kwargs)
            print("Trainer设置完成")
            return True
            
        except Exception as e:
            print(f"设置trainer失败: {e}")
            return False
    
    def run_validation_test(self):
        """运行验证测试"""
        try:
            print("开始运行验证测试...")
            
            # 确保模型在正确的设备上并处于评估模式
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 运行验证
            val_results = self.trainer.validate(
                model=self.model,
                datamodule=self.datamodule,
                verbose=True
            )
            
            print("验证测试完成")
            print(f"验证结果: {val_results}")
            
            return val_results
            
        except Exception as e:
            print(f"运行验证测试失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return None
    
    def run_test(self):
        """运行测试"""
        try:
            print("开始运行测试...")
            
            # 确保模型处于评估模式
            self.model.eval()
            
            # 运行测试
            test_results = self.trainer.test(
                model=self.model,
                datamodule=self.datamodule,
                verbose=True
            )
            
            print("测试完成")
            print(f"测试结果: {test_results}")
            
            return test_results
            
        except Exception as e:
            print(f"运行测试失败: {e}")
            return None
    
    def run_custom_latent_action_test(self):
        """运行自定义的latent action测试"""
        try:
            print("开始运行自定义latent action测试...")
            
            if not (hasattr(self.model, 'latent_motion_pred') and self.model.latent_motion_pred):
                print("模型未启用latent motion预测，跳过此测试")
                return None
            
            # 确保模型在正确的设备上并处于评估模式
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 获取验证数据加载器
            val_dataloader = self.datamodule.val_dataloader()
            
            similarities = []
            latent_action_losses = []
            indices_cross_entropy_losses = []
            batch_count = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    if batch_idx >= self.limit_test_batches:
                        break
                    
                    try:
                        # 移动数据到设备
                        batch = self._move_to_device(batch)
                        
                        # 处理每个modality
                        for modality_scope, dataset_batch in batch.items():
                            self.model.modality_scope = modality_scope
                            
                            # 生成ground truth和predicted latent actions
                            similarity = self._test_latent_action_consistency(dataset_batch)
                            
                            if similarity is not None:
                                similarities.append(similarity)
                                print(f"批次 {batch_idx}, Modality {modality_scope}: 嵌入相似度 = {similarity:.4f}")
                            
                            # 计算generated和ground truth latent action之间的loss
                            latent_loss = self._calculate_latent_action_loss(dataset_batch)
                            if latent_loss is not None:
                                latent_action_losses.append(latent_loss)
                                print(f"批次 {batch_idx}, Modality {modality_scope}: Latent Action Loss = {latent_loss:.4f}")
                            
                            # 计算indices之间的cross entropy loss（近似方法）
                            indices_ce_loss = self._calculate_indices_cross_entropy_loss(dataset_batch)
                            if indices_ce_loss is not None:
                                indices_cross_entropy_losses.append(indices_ce_loss)
                                print(f"批次 {batch_idx}, Modality {modality_scope}: 近似Cross Entropy Loss = {indices_ce_loss:.4f}")
                            
                            # 计算真实自回归cross entropy loss（更准确的方法）
                            try:
                                true_ar_loss = self._calculate_autoregressive_with_true_logits(dataset_batch)
                                if true_ar_loss is not None:
                                    print(f"批次 {batch_idx}, Modality {modality_scope}: 真实自回归Cross Entropy Loss = {true_ar_loss:.4f}")
                            except Exception as e:
                                print(f"计算真实自回归cross entropy时出错: {e}")
                            
                            break  # 只测试第一个modality
                        
                        batch_count += 1
                        
                    except Exception as e:
                        print(f"处理批次 {batch_idx} 时出错: {e}")
                        import traceback
                        print(f"详细错误: {traceback.format_exc()}")
                        continue
            
            # 计算统计结果
            if similarities or latent_action_losses or indices_cross_entropy_losses:
                results = {
                    'mean_similarity': np.mean(similarities) if similarities else None,
                    'std_similarity': np.std(similarities) if similarities else None,
                    'min_similarity': np.min(similarities) if similarities else None,
                    'max_similarity': np.max(similarities) if similarities else None,
                    'total_samples': len(similarities),
                    'mean_latent_action_loss': np.mean(latent_action_losses) if latent_action_losses else None,
                    'std_latent_action_loss': np.std(latent_action_losses) if latent_action_losses else None,
                    'min_latent_action_loss': np.min(latent_action_losses) if latent_action_losses else None,
                    'max_latent_action_loss': np.max(latent_action_losses) if latent_action_losses else None,
                    'total_loss_samples': len(latent_action_losses),
                    'mean_indices_ce_loss': np.mean(indices_cross_entropy_losses) if indices_cross_entropy_losses else None,
                    'std_indices_ce_loss': np.std(indices_cross_entropy_losses) if indices_cross_entropy_losses else None,
                    'min_indices_ce_loss': np.min(indices_cross_entropy_losses) if indices_cross_entropy_losses else None,
                    'max_indices_ce_loss': np.max(indices_cross_entropy_losses) if indices_cross_entropy_losses else None,
                    'total_indices_ce_samples': len(indices_cross_entropy_losses)
                }
                
                print(f"\nLatent Action测试结果:")
                if similarities:
                    print(f"- 平均嵌入相似度: {results['mean_similarity']:.4f}")
                    print(f"- 相似度标准差: {results['std_similarity']:.4f}")
                    print(f"- 相似度范围: {results['min_similarity']:.4f} - {results['max_similarity']:.4f}")
                    print(f"- 相似度样本数: {results['total_samples']}")
                
                if latent_action_losses:
                    print(f"\nLatent Action Loss结果:")
                    print(f"- 平均loss: {results['mean_latent_action_loss']:.4f}")
                    print(f"- loss标准差: {results['std_latent_action_loss']:.4f}")
                    print(f"- loss范围: {results['min_latent_action_loss']:.4f} - {results['max_latent_action_loss']:.4f}")
                    print(f"- loss样本数: {results['total_loss_samples']}")
                
                if indices_cross_entropy_losses:
                    print(f"\nIndices Cross Entropy Loss结果:")
                    print(f"- 平均CE loss: {results['mean_indices_ce_loss']:.4f}")
                    print(f"- CE loss标准差: {results['std_indices_ce_loss']:.4f}")
                    print(f"- CE loss范围: {results['min_indices_ce_loss']:.4f} - {results['max_indices_ce_loss']:.4f}")
                    print(f"- CE loss样本数: {results['total_indices_ce_samples']}")
                
                # 保存结果
                self._save_latent_action_results(results, similarities, latent_action_losses, indices_cross_entropy_losses)
                
                return results
            else:
                print("未能生成任何有效的相似度测量或loss计算")
                return None
                
        except Exception as e:
            print(f"运行自定义latent action测试失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _move_to_device(self, batch):
        """将批次数据移动到设备"""
        def move_item(item):
            if isinstance(item, torch.Tensor):
                # 移动到设备并确保数据类型一致
                item = item.to(self.device)
                # 如果设备是CPU且张量是half精度，转换为float32
                if self.device == "cpu" and item.dtype == torch.float16:
                    item = item.float()
                return item
            elif isinstance(item, dict):
                return {k: move_item(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [move_item(x) for x in item]
            else:
                return item
        
        return move_item(batch)
    
    def _test_latent_action_consistency(self, dataset_batch):
        """测试latent action的一致性"""
        try:
            # 检查必要的组件
            if not hasattr(self.model, 'pretrained_vq') or self.model.pretrained_vq is None:
                return None
            
            # 计算ground truth latent motion embeddings
            gt_embeddings = self._generate_gt_latent_motion(dataset_batch)
            if gt_embeddings is None:
                return None
            
            # 计算predicted latent motion embeddings
            pred_embeddings = self._generate_pred_latent_motion(dataset_batch, gt_embeddings)
            if pred_embeddings is None:
                return None
            
            # 计算相似度
            similarity = self._calculate_embedding_similarity(gt_embeddings, pred_embeddings)
            return similarity
            
        except Exception as e:
            print(f"测试latent action一致性时出错: {e}")
            return None
    
    def _generate_gt_latent_motion(self, dataset_batch):
        """生成ground truth latent motion embeddings"""
        try:
            # 这里复用模型中的逻辑，遵循3D motion tokenization pipeline
            # 提取RGB观察并处理
            t = dataset_batch['rgb_obs']['rgb_static'].shape[1]
            rgb_static = dataset_batch['rgb_obs']['rgb_static'][:, :-1] if t > 1 else dataset_batch['rgb_obs']['rgb_static']
            rgb_gripper = dataset_batch['rgb_obs']['rgb_gripper'][:, :-1] if t > 1 else dataset_batch['rgb_obs']['rgb_gripper']
            
            # 获取目标图像（gen_static/gen_gripper）
            if 'gen_static' in dataset_batch["rgb_obs"] and 'gen_gripper' in dataset_batch["rgb_obs"]:
                gen_static = dataset_batch["rgb_obs"]['gen_static']        # (B, T, C, H, W)
                gen_gripper = dataset_batch["rgb_obs"]['gen_gripper']      # (B, T, C, H, W)
                # 调整尺寸: 112*112-> 224*224
                gen_static = self.model.interpolate_img(gen_static, size=(224, 224))
                gen_gripper = self.model.interpolate_img(gen_gripper, size=(224, 224))
            else:
                # 备选方案：使用当前帧作为目标（无运动）
                gen_static = rgb_static
                gen_gripper = rgb_gripper
            
            rgb_gripper = self.model.interpolate_img(rgb_gripper, size=(224, 224))
            B, T = rgb_static.shape[:2]
            
            # 处理每个时间步以生成ground truth motion embeddings，遵循LatentMotionTokenizer3D pipeline
            gt_latent_motion_embeddings = []
            for timestep in range(T):
                # 步骤1: 使用预训练图像编码器编码图像
                cond_static_encoded = self.model.pretrained_image_encoder(rgb_static[:, timestep])      # 输出带last_hidden_state: (B, num_patches, hidden_dim)
                target_static_encoded = self.model.pretrained_image_encoder(gen_static[:, timestep])    # 输出带last_hidden_state: (B, num_patches, hidden_dim)
                cond_gripper_encoded = self.model.pretrained_image_encoder(rgb_gripper[:, timestep])    # 输出带last_hidden_state: (B, num_patches, hidden_dim)
                target_gripper_encoded = self.model.pretrained_image_encoder(gen_gripper[:, timestep])  # 输出带last_hidden_state: (B, num_patches, hidden_dim)
                
                # 步骤2: 使用m_former为每个视点提取motion tokens
                # 视点1 (静态相机): condition=rgb_static, target=gen_static
                motion_tokens_view1 = self.model.pretrained_m_former(
                    cond_hidden_states=cond_static_encoded.last_hidden_state,
                    target_hidden_states=target_static_encoded.last_hidden_state
                ).last_hidden_state[:, :self.model.pretrained_m_former.query_num]  # (B, query_num, hidden_dim)
                
                # 视点2 (抓手相机): condition=rgb_gripper, target=gen_gripper  
                motion_tokens_view2 = self.model.pretrained_m_former(
                    cond_hidden_states=cond_gripper_encoded.last_hidden_state,
                    target_hidden_states=target_gripper_encoded.last_hidden_state
                ).last_hidden_state[:, :self.model.pretrained_m_former.query_num]  # (B, query_num, hidden_dim)
                
                # 步骤3: 使用m_former3d融合两个视点
                combined_tokens = torch.cat([motion_tokens_view1, motion_tokens_view2], dim=1)  # (B, query_num*2, hidden_dim)
                fused_3d_motion_tokens = self.model.pretrained_m_former3d(combined_tokens).last_hidden_state[:, :self.model.pretrained_m_former3d.query_num]  # (B, query_3d, hidden_dim)
                
                # 步骤4: 使用VQ进行下采样和量化
                motion_tokens_down = self.model.pretrained_vq_down_resampler(fused_3d_motion_tokens)  # (B, query_3d, codebook_dim)
                
                # 量化 - 获取量化后的embedding而不是indices
                quantized_embeddings, indices, _ = self.model.pretrained_vq(motion_tokens_down) # quant: (B, query_3d, codebook_dim), indices: (B, query_3d)
                gt_latent_motion_embeddings.append(quantized_embeddings)
            
            # 堆叠所有时间步: (B, T, query_3d, codebook_dim)  
            gt_latent_motion_embeddings = torch.stack(gt_latent_motion_embeddings, dim=1)
            
            # 调整形状以匹配per_latent_motion_len
            if self.model.per_latent_motion_len != gt_latent_motion_embeddings.shape[2]:
                if self.model.per_latent_motion_len > gt_latent_motion_embeddings.shape[2]:
                    # 重复最后一个token维度以匹配per_latent_motion_len
                    repeat_factor = self.model.per_latent_motion_len // gt_latent_motion_embeddings.shape[2]
                    remainder = self.model.per_latent_motion_len % gt_latent_motion_embeddings.shape[2]
                    
                    repeated_embeddings = gt_latent_motion_embeddings.repeat(1, 1, repeat_factor, 1)
                    if remainder > 0:
                        repeated_embeddings = torch.cat([repeated_embeddings, gt_latent_motion_embeddings[:, :, :remainder]], dim=2)
                    gt_latent_motion_embeddings = repeated_embeddings
                else:
                    # 如果per_latent_motion_len较小则截断
                    gt_latent_motion_embeddings = gt_latent_motion_embeddings[:, :, :self.model.per_latent_motion_len]
            
            return gt_latent_motion_embeddings
            
        except Exception as e:
            print(f"生成ground truth时出错: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _generate_pred_latent_motion(self, dataset_batch, gt_embeddings):
        """生成自回归predicted latent motion embeddings（真正的生成能力测试）"""
        try:
            combined_features, combined_mask = self.model.compute_latent_goal_embeddings_and_mask(dataset_batch)
            
            # ===== 关键：使用自回归生成模式（测试真正的生成能力）=====
            # 推理模式，不提供ground truth，让模型完全自回归生成
            motion_results = self.model.motion_transformer(
                perceptual_features=combined_features,
                latent_motion_ids=None,  # 不提供ground truth，让模型自回归生成
                attention_mask=combined_mask,
                train=False,  # 推理模式，自回归生成
                seq_len=gt_embeddings.shape[1] if gt_embeddings is not None else 1  # 指定要生成的序列长度
            )
            
            # 自回归生成模式下返回的是 'latent_motion_id_preds'（自回归生成的indices）
            predicted_indices = motion_results.get('latent_motion_id_preds')
            
            if predicted_indices is not None:
                print(f"自回归生成的latent motion indices shape: {predicted_indices.shape}")
                print("注意：这是真正的自回归生成，测试的是生成能力而非teacher forcing")
                
                # 将自回归生成的indices转换为embeddings，使用VQ的codebook
                # predicted_indices shape: (B, seq_len, per_latent_motion_len)
                B, seq_len, per_latent_motion_len = predicted_indices.shape
                
                # Flatten indices for embedding lookup
                flat_indices = predicted_indices.view(-1)  # (B * seq_len * per_latent_motion_len,)
                
                # 从VQ codebook中获取embeddings
                codebook = self.model.pretrained_vq.embedding.weight  # (codebook_size, embedding_dim)
                predicted_embeddings = codebook[flat_indices]  # (B * seq_len * per_latent_motion_len, embedding_dim)
                
                # Reshape back to original dimensions
                predicted_embeddings = predicted_embeddings.view(B, seq_len, per_latent_motion_len, -1)  # (B, seq_len, per_latent_motion_len, embedding_dim)
                
                print(f"转换后的自回归predicted embeddings shape: {predicted_embeddings.shape}")
                return predicted_embeddings
            else:
                print("motion_transformer返回的结果中没有找到latent_motion_id_preds")
                print(f"可用的键: {list(motion_results.keys())}")
                return None
            
        except Exception as e:
            print(f"生成自回归predicted latent motion时出错: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _calculate_embedding_similarity(self, gt_embeddings, pred_embeddings):
        """计算embeddings的余弦相似度"""
        try:
            if gt_embeddings is None or pred_embeddings is None:
                return None
            
            # 确保在同一设备上
            gt_embeddings = gt_embeddings.to(pred_embeddings.device)
            
            # Flatten the embeddings for comparison
            # gt_embeddings: (B, T, query_3d, codebook_dim) or (B, seq_len, per_latent_motion_len, codebook_dim)
            # pred_embeddings: (B, seq_len, per_latent_motion_len, embedding_dim)
            
            # 取最小的维度进行比较
            min_batch = min(gt_embeddings.shape[0], pred_embeddings.shape[0])
            min_seq = min(gt_embeddings.shape[1], pred_embeddings.shape[1])
            min_len = min(gt_embeddings.shape[2], pred_embeddings.shape[2])
            
            gt_flat = gt_embeddings[:min_batch, :min_seq, :min_len].flatten(0, 2)  # (min_batch * min_seq * min_len, embedding_dim)
            pred_flat = pred_embeddings[:min_batch, :min_seq, :min_len].flatten(0, 2)  # (min_batch * min_seq * min_len, embedding_dim)
            
            # 确保embedding维度一致
            min_embedding_dim = min(gt_flat.shape[-1], pred_flat.shape[-1])
            gt_flat = gt_flat[..., :min_embedding_dim]
            pred_flat = pred_flat[..., :min_embedding_dim]
            
            # 计算余弦相似度
            import torch.nn.functional as F
            gt_normalized = F.normalize(gt_flat, p=2, dim=-1)
            pred_normalized = F.normalize(pred_flat, p=2, dim=-1)
            
            # 计算逐元素余弦相似度
            cosine_similarities = torch.sum(gt_normalized * pred_normalized, dim=-1)  # (min_batch * min_seq * min_len,)
            
            # 计算平均相似度
            mean_similarity = torch.mean(cosine_similarities).item()
            
            print(f"GT embeddings shape: {gt_embeddings.shape}")
            print(f"Pred embeddings shape: {pred_embeddings.shape}")
            print(f"Flattened GT shape: {gt_flat.shape}")
            print(f"Flattened Pred shape: {pred_flat.shape}")
            print(f"Cosine similarities shape: {cosine_similarities.shape}")
            print(f"Mean cosine similarity: {mean_similarity:.4f}")
            
            return mean_similarity
                
        except Exception as e:
            print(f"计算embedding相似度时出错: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _calculate_latent_action_loss(self, dataset_batch):
        """计算generated latent action和ground truth latent action之间的loss"""
        try:
            # 检查必要的组件
            if not hasattr(self.model, 'motion_transformer') or self.model.motion_transformer is None:
                print("模型没有motion_transformer组件")
                return None
            
            # 生成ground truth latent motion indices
            gt_latent_motion_indices = self._generate_gt_latent_motion_indices_for_loss(dataset_batch)
            if gt_latent_motion_indices is None:
                return None
            
            # 计算combined features和mask
            combined_features, combined_mask = self.model.compute_latent_goal_embeddings_and_mask(dataset_batch)
            
            # 使用motion transformer在训练模式下计算loss（使用ground truth作为target）
            motion_results = self.model.motion_transformer(
                perceptual_features=combined_features,
                latent_motion_ids=gt_latent_motion_indices,
                attention_mask=combined_mask,
                train=True  # 训练模式，会计算cross-entropy loss
            )
            
            # 提取loss
            latent_action_loss = motion_results.get('loss')
            if latent_action_loss is not None:
                return latent_action_loss.item()
            else:
                print("motion_transformer没有返回loss")
                return None
                
        except Exception as e:
            print(f"计算latent action loss时出错: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _generate_gt_latent_motion_indices_for_loss(self, dataset_batch):
        """为loss计算生成ground truth latent motion indices"""
        try:
            # 这里复用模型中的逻辑，遵循3D motion tokenization pipeline
            t = dataset_batch['rgb_obs']['rgb_static'].shape[1]
            rgb_static = dataset_batch['rgb_obs']['rgb_static'][:, :-1] if t > 1 else dataset_batch['rgb_obs']['rgb_static']
            rgb_gripper = dataset_batch['rgb_obs']['rgb_gripper'][:, :-1] if t > 1 else dataset_batch['rgb_obs']['rgb_gripper']
            
            # 获取目标图像（gen_static/gen_gripper）
            if 'gen_static' in dataset_batch["rgb_obs"] and 'gen_gripper' in dataset_batch["rgb_obs"]:
                gen_static = dataset_batch["rgb_obs"]['gen_static']
                gen_gripper = dataset_batch["rgb_obs"]['gen_gripper']
                gen_static = self.model.interpolate_img(gen_static, size=(224, 224))
                gen_gripper = self.model.interpolate_img(gen_gripper, size=(224, 224))
            else:
                gen_static = rgb_static
                gen_gripper = rgb_gripper
            
            rgb_gripper = self.model.interpolate_img(rgb_gripper, size=(224, 224))
            B, T = rgb_static.shape[:2]
            
            # 处理每个时间步生成ground truth motion indices
            gt_latent_motion_indices = []
            for timestep in range(T):
                # 使用预训练图像编码器
                cond_static_encoded = self.model.pretrained_image_encoder(rgb_static[:, timestep])
                target_static_encoded = self.model.pretrained_image_encoder(gen_static[:, timestep])
                cond_gripper_encoded = self.model.pretrained_image_encoder(rgb_gripper[:, timestep])
                target_gripper_encoded = self.model.pretrained_image_encoder(gen_gripper[:, timestep])
                
                # 使用m_former提取motion tokens
                motion_tokens_view1 = self.model.pretrained_m_former(
                    cond_hidden_states=cond_static_encoded.last_hidden_state,
                    target_hidden_states=target_static_encoded.last_hidden_state
                ).last_hidden_state[:, :self.model.pretrained_m_former.query_num]
                
                motion_tokens_view2 = self.model.pretrained_m_former(
                    cond_hidden_states=cond_gripper_encoded.last_hidden_state,
                    target_hidden_states=target_gripper_encoded.last_hidden_state
                ).last_hidden_state[:, :self.model.pretrained_m_former.query_num]
                
                # 使用m_former3d融合
                combined_tokens = torch.cat([motion_tokens_view1, motion_tokens_view2], dim=1)
                fused_3d_motion_tokens = self.model.pretrained_m_former3d(combined_tokens).last_hidden_state[:, :self.model.pretrained_m_former3d.query_num]
                
                # VQ下采样和量化
                motion_tokens_down = self.model.pretrained_vq_down_resampler(fused_3d_motion_tokens)
                _, indices, _ = self.model.pretrained_vq(motion_tokens_down)
                gt_latent_motion_indices.append(indices)
            
            # 堆叠所有时间步
            gt_latent_motion_indices = torch.stack(gt_latent_motion_indices, dim=1)
            
            # 调整形状以匹配per_latent_motion_len
            if self.model.per_latent_motion_len != gt_latent_motion_indices.shape[-1]:
                if self.model.per_latent_motion_len > gt_latent_motion_indices.shape[-1]:
                    repeat_factor = self.model.per_latent_motion_len // gt_latent_motion_indices.shape[-1]
                    remainder = self.model.per_latent_motion_len % gt_latent_motion_indices.shape[-1]
                    
                    repeated_indices = gt_latent_motion_indices.repeat(1, 1, repeat_factor)
                    if remainder > 0:
                        repeated_indices = torch.cat([repeated_indices, gt_latent_motion_indices[:, :, :remainder]], dim=-1)
                    gt_latent_motion_indices = repeated_indices
                else:
                    gt_latent_motion_indices = gt_latent_motion_indices[:, :, :self.model.per_latent_motion_len]
            
            return gt_latent_motion_indices
            
        except Exception as e:
            print(f"生成ground truth indices时出错: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _calculate_indices_cross_entropy_loss(self, dataset_batch):
        """计算autoregressive生成的latent action与ground truth之间的cross entropy loss"""
        try:
            # 检查必要的组件
            if not hasattr(self.model, 'motion_transformer') or self.model.motion_transformer is None:
                print("模型没有motion_transformer组件")
                return None
            
            # 生成ground truth indices
            gt_indices = self._generate_gt_latent_motion_indices_for_loss(dataset_batch)
            if gt_indices is None:
                return None
            
            # 计算combined features和mask
            combined_features, combined_mask = self.model.compute_latent_goal_embeddings_and_mask(dataset_batch)
            
            # ===== 方法1：自回归生成后计算准确率和伪cross entropy =====
            # 推理模式，让模型自回归生成latent action
            motion_results = self.model.motion_transformer(
                perceptual_features=combined_features,
                latent_motion_ids=None,  # 不提供ground truth，让模型自回归生成
                attention_mask=combined_mask,
                train=False,  # 推理模式，自回归生成
                seq_len=gt_indices.shape[1]  # 指定要生成的序列长度
            )
            
            # 获取自回归生成的indices
            generated_indices = motion_results.get('latent_motion_id_preds')
            if generated_indices is None:
                print("motion_transformer没有返回latent_motion_id_preds")
                return None
            
            print(f"自回归生成的indices shape: {generated_indices.shape}")
            print(f"Ground truth indices shape: {gt_indices.shape}")
            
            # 确保维度匹配
            min_batch = min(gt_indices.shape[0], generated_indices.shape[0])
            min_seq = min(gt_indices.shape[1], generated_indices.shape[1])
            min_len = min(gt_indices.shape[2], generated_indices.shape[2])
            
            gt_indices_trimmed = gt_indices[:min_batch, :min_seq, :min_len]
            gen_indices_trimmed = generated_indices[:min_batch, :min_seq, :min_len]
            
            # 计算生成准确率
            correct_predictions = (gen_indices_trimmed == gt_indices_trimmed).float()
            accuracy = correct_predictions.mean().item()
            
            # ===== 方法2：使用自回归生成过程中的真实logits =====
            # 为了获得真实的cross entropy，我们需要重新生成并捕获每一步的logits
            # 这需要修改motion_transformer来支持返回每一步的logits
            
            # 当前的近似方法：基于准确率计算伪cross entropy
            # 如果准确率高，cross entropy应该低；如果准确率低，cross entropy应该高
            # 使用负log似然的近似：-log(accuracy) 但这只是一个粗略估计
            
            vocab_size = self.model.latent_motion_codebook_size
            
            # 计算每个token的cross entropy（基于是否正确预测）
            # 正确预测的token: 低loss; 错误预测的token: 高loss
            token_losses = []
            for b in range(min_batch):
                for s in range(min_seq):
                    for l in range(min_len):
                        if gt_indices_trimmed[b, s, l] == gen_indices_trimmed[b, s, l]:
                            # 正确预测，假设模型给出了高概率
                            token_losses.append(-torch.log(torch.tensor(0.9)).item())  # 低loss
                        else:
                            # 错误预测，假设模型给出了低概率
                            token_losses.append(-torch.log(torch.tensor(1.0/vocab_size)).item())  # 高loss
            
            mean_cross_entropy = sum(token_losses) / len(token_losses) if token_losses else float('inf')
            
            print(f"自回归生成准确率: {accuracy:.4f}")
            print(f"基于准确率的近似cross entropy: {mean_cross_entropy:.4f}")
            print(f"注意：这是基于自回归生成结果的近似cross entropy，测试生成能力而非teacher forcing")
            
            return mean_cross_entropy
            
        except Exception as e:
            print(f"计算autoregressive cross entropy loss时出错: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _calculate_autoregressive_with_true_logits(self, dataset_batch):
        """更准确的方法：手动实现自回归生成过程，获取每一步的真实logits"""
        try:
            # 这个方法需要重新实现自回归生成过程，在每一步都记录logits
            # 然后计算真正的cross entropy
            
            # 生成ground truth indices
            gt_indices = self._generate_gt_latent_motion_indices_for_loss(dataset_batch)
            if gt_indices is None:
                return None
            
            # 计算combined features和mask
            combined_features, combined_mask = self.model.compute_latent_goal_embeddings_and_mask(dataset_batch)
            
            # 获取motion transformer的核心组件
            motion_transformer = self.model.motion_transformer
            batch_size, seq_len, per_len = gt_indices.shape
            vocab_size = self.model.latent_motion_codebook_size
            total_tokens = seq_len * per_len
            
            # 准备条件输入
            cond_embeddings = motion_transformer.input_projection(combined_features)
            cond_embeddings_normed = motion_transformer.embed_ln(cond_embeddings)
            
            # Start token
            start_token_emb = motion_transformer.start_token.weight.view(1, 1, -1).repeat(batch_size, 1, 1)
            start_token_emb_normed = motion_transformer.embed_ln(start_token_emb)
            
            # 初始化sequence
            current_sequence = torch.cat([cond_embeddings_normed, start_token_emb_normed], dim=1)
            
            all_logits = []
            generated_tokens = []
            
            # 自回归生成每个token
            for step in range(total_tokens):
                # 创建attention mask
                cond_tokens = combined_features.shape[1]
                cond_mask = torch.ones(batch_size, cond_tokens, device=combined_features.device)
                start_mask = torch.ones(batch_size, 1, device=combined_features.device)
                latent_mask = torch.ones(batch_size, current_sequence.shape[1] - cond_tokens - 1, device=combined_features.device)
                current_mask = torch.cat([cond_mask, start_mask, latent_mask], dim=1)
                
                # Transformer前向传播
                with torch.no_grad():
                    outputs = motion_transformer.transformer(
                        inputs_embeds=current_sequence,
                        attention_mask=current_mask,
                    )
                
                # 获取最后一个位置的logits
                hidden_states = outputs.last_hidden_state
                last_hidden = hidden_states[:, -1]
                logits = motion_transformer.pred_latent_motion_head(last_hidden)
                
                all_logits.append(logits)
                
                # 采样下一个token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                generated_tokens.append(next_token)
                
                # 添加到序列中
                next_token_emb = motion_transformer.embed_latent_motion(next_token).unsqueeze(1)
                next_token_emb_normed = motion_transformer.embed_ln(next_token_emb)
                current_sequence = torch.cat([current_sequence, next_token_emb_normed], dim=1)
            
            # 计算真正的cross entropy
            all_logits = torch.stack(all_logits, dim=1)  # (batch_size, total_tokens, vocab_size)
            gt_flat = gt_indices.view(batch_size, -1)  # (batch_size, total_tokens)
            
            # 确保token数量匹配
            min_tokens = min(all_logits.shape[1], gt_flat.shape[1])
            logits_trimmed = all_logits[:, :min_tokens]  # (batch_size, min_tokens, vocab_size)
            targets_trimmed = gt_flat[:, :min_tokens]  # (batch_size, min_tokens)
            
            # 计算cross entropy
            import torch.nn.functional as F
            logits_flat = logits_trimmed.view(-1, vocab_size)
            targets_flat = targets_trimmed.view(-1)
            
            cross_entropy = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
            
            # 计算准确率
            generated_flat = torch.stack(generated_tokens, dim=1)[:, :min_tokens]
            accuracy = (generated_flat == targets_trimmed).float().mean().item()
            
            print(f"真实自回归cross entropy: {cross_entropy.item():.4f}")
            print(f"自回归生成准确率: {accuracy:.4f}")
            print(f"这是真正基于自回归生成logits的cross entropy loss")
            
            return cross_entropy.item()
            
        except Exception as e:
            print(f"计算真实自回归cross entropy时出错: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _save_latent_action_results(self, results, similarities, latent_action_losses=None, indices_cross_entropy_losses=None):
        """保存latent action测试结果"""
        try:
            results_file = os.path.join(self.output_dir, "latent_action_test_results.txt")
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("MDT Latent Action测试结果\n")
                f.write("=" * 40 + "\n")
                
                for key, value in results.items():
                    if value is not None:
                        f.write(f"{key}: {value}\n")
                
                if similarities:
                    f.write(f"\n详细嵌入相似度数据:\n")
                    for i, sim in enumerate(similarities):
                        f.write(f"样本 {i}: {sim:.4f}\n")
                
                if latent_action_losses:
                    f.write(f"\n详细Latent Action Loss数据:\n")
                    for i, loss in enumerate(latent_action_losses):
                        f.write(f"样本 {i}: {loss:.4f}\n")
                
                if indices_cross_entropy_losses:
                    f.write(f"\n详细Indices Cross Entropy Loss数据:\n")
                    for i, loss in enumerate(indices_cross_entropy_losses):
                        f.write(f"样本 {i}: {loss:.4f}\n")
            
            print(f"结果已保存到: {results_file}")
            
        except Exception as e:
            print(f"保存结果时出错: {e}")
    
    def run_full_test(self):
        """运行完整测试"""
        print("=" * 60)
        print("开始MDT模型完整测试")
        print("=" * 60)
        
        # 1. 加载模型和配置
        if not self.load_checkpoint_and_config():
            print("加载失败，退出测试")
            return False
        
        # 2. 设置数据模块
        if not self.setup_datamodule():
            print("设置datamodule失败，退出测试")
            return False
        
        # 3. 设置trainer
        if not self.setup_trainer():
            print("设置trainer失败，退出测试")
            return False
        
        print("所有组件设置完成，开始测试...")
        
        # 4. 运行验证测试
        print("\n" + "="*40)
        print("运行标准验证测试")
        print("="*40)
        val_results = self.run_validation_test()
        
        # 5. 运行自定义latent action测试
        print("\n" + "="*40)
        print("运行Latent Action一致性测试")
        print("="*40)
        latent_results = self.run_custom_latent_action_test()
        
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
        print(f"结果文件保存在: {self.output_dir}")
        
        return True


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MDT模型测试脚本")
    
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="模型checkpoint文件路径"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (如果使用普通state_dict checkpoint则必需)"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./mdt_test_outputs",
        help="输出目录"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="使用的设备"
    )
    
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=10,
        help="限制测试的批次数量"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查checkpoint文件
    if not os.path.exists(args.checkpoint):
        print(f"错误: Checkpoint文件不存在: {args.checkpoint}")
        
        # 尝试查找可能的文件
        search_dirs = [
            "/home/hlwang/mdt_policy/checkpoints",
            "/home/hlwang/mdt_policy/outputs",
            "/home/hlwang/results"
        ]
        
        print("\n正在搜索可能的checkpoint文件...")
        found_files = []
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith('.ckpt'):
                            found_files.append(os.path.join(root, file))
        
        if found_files:
            print("发现的checkpoint文件:")
            for file in found_files[:10]:  # 只显示前10个
                print(f"  {file}")
        else:
            print("未找到任何.ckpt文件")
        
        return
    
    # 检查配置文件（如果提供）
    if args.config and not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    # 创建测试运行器
    test_runner = MDTTestRunner(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        limit_test_batches=args.limit_batches,
    )
    
    # 运行测试
    success = test_runner.run_full_test()
    
    if success:
        print(f"\n测试成功完成! 结果保存在: {args.output_dir}")
    else:
        print("\n测试失败")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main()