#!/usr/bin/env python3
"""
简化的MDT模型验证测试脚本
功能：
1. 加载训练好的MDT模型
2. 在不进行训练的情况下运行验证
3. 比较预测的latent action与ground truth的一致性
4. 可选择使用预训练decoder生成图片
"""

import os
import sys
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent / "mdt_policy"
if project_root.exists():
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class SimpleMDTTester:
    """简化的MDT模型测试器"""
    
    def __init__(
        self,
        model_ckpt_path: str,
        config_path: str = None,
        device: str = "auto",
        num_test_batches: int = 5,
    ):
        self.model_ckpt_path = model_ckpt_path
        self.config_path = config_path
        self.device = self._setup_device(device)
        self.num_test_batches = num_test_batches
        
        # 加载模型和配置
        self.model = None
        self.datamodule = None
        self.results = {
            'similarities': [],
            'losses': [],
            'batch_info': []
        }
    
    def _setup_device(self, device):
        """设置设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model_from_checkpoint(self):
        """从checkpoint加载模型"""
        try:
            print(f"正在从 {self.model_ckpt_path} 加载模型...")
            
            # 尝试直接加载checkpoint
            checkpoint = torch.load(self.model_ckpt_path, map_location=self.device)
            
            # 检查checkpoint结构
            if 'hyper_parameters' in checkpoint:
                # 这是PyTorch Lightning checkpoint
                from mdt.models.mdt_3d_latent_action import MDT3dLatentActionAgent
                
                # 从checkpoint加载模型
                self.model = MDT3dLatentActionAgent.load_from_checkpoint(
                    self.model_ckpt_path,
                    map_location=self.device,
                    strict=False
                )
                print("成功加载PyTorch Lightning模型")
            else:
                # 这是普通的state_dict
                print("检测到普通state_dict，需要配置文件来初始化模型")
                if self.config_path is None:
                    raise ValueError("需要提供配置文件路径来初始化模型")
                self._load_model_with_config(checkpoint)
            
            self.model.eval()
            self.model.to(self.device)
            
            # 打印模型信息
            self._print_model_info()
            
            return True
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def _load_model_with_config(self, state_dict):
        """使用配置文件加载模型"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        # 加载配置
        cfg = OmegaConf.load(self.config_path)
        
        # 初始化模型
        from mdt.models.mdt_3d_latent_action import MDT3dLatentActionAgent
        self.model = hydra.utils.instantiate(cfg.model)
        
        # 加载权重
        self.model.load_state_dict(state_dict, strict=False)
        print("使用配置文件成功加载模型")
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"""
模型信息:
- 总参数数量: {total_params:,}
- 可训练参数: {trainable_params:,}
- 冻结参数: {total_params - trainable_params:,}
- 设备: {self.device}
- 是否启用latent motion预测: {getattr(self.model, 'latent_motion_pred', False)}
""")
    
    def setup_datamodule(self, datamodule_config=None):
        """设置数据模块"""
        try:
            if datamodule_config is None and self.config_path:
                # 从配置文件加载datamodule配置
                cfg = OmegaConf.load(self.config_path)
                if 'datamodule' in cfg:
                    self.datamodule = hydra.utils.instantiate(cfg.datamodule)
                else:
                    print("配置文件中未找到datamodule配置，将使用默认设置")
                    return False
            elif datamodule_config:
                self.datamodule = hydra.utils.instantiate(datamodule_config)
            else:
                print("无法设置datamodule，缺少配置")
                return False
            
            self.datamodule.setup("fit")  # 设置为fit模式以获取验证数据
            print("Datamodule设置成功")
            return True
            
        except Exception as e:
            print(f"设置datamodule失败: {e}")
            return False
    
    @torch.no_grad()
    def run_validation_test(self):
        """运行验证测试"""
        if self.model is None:
            print("请先加载模型")
            return False
        
        if self.datamodule is None:
            print("请先设置datamodule")
            return False
        
        print(f"开始运行验证测试，处理 {self.num_test_batches} 个批次...")
        
        # 获取验证数据加载器
        val_dataloader = self.datamodule.val_dataloader()
        
        total_similarity = 0.0
        processed_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="处理验证批次")):
            if batch_idx >= self.num_test_batches:
                break
            
            try:
                # 移动数据到设备
                batch = self._move_to_device(batch)
                
                # 处理单个批次
                batch_results = self._process_single_batch(batch, batch_idx)
                
                if batch_results:
                    self.results['similarities'].extend(batch_results['similarities'])
                    self.results['losses'].append(batch_results.get('loss', 0.0))
                    self.results['batch_info'].append({
                        'batch_idx': batch_idx,
                        'avg_similarity': np.mean(batch_results['similarities'])
                    })
                    
                    total_similarity += np.mean(batch_results['similarities'])
                    processed_batches += 1
                    
                    print(f"批次 {batch_idx}: 平均相似度 = {np.mean(batch_results['similarities']):.4f}")
                
            except Exception as e:
                print(f"处理批次 {batch_idx} 时出错: {e}")
                continue
        
        # 计算总体统计
        if processed_batches > 0:
            avg_similarity = total_similarity / processed_batches
            print(f"\n验证完成!")
            print(f"平均余弦相似度: {avg_similarity:.4f}")
            print(f"成功处理的批次数: {processed_batches}/{self.num_test_batches}")
            
            self._save_results()
            return True
        else:
            print("没有成功处理任何批次")
            return False
    
    def _move_to_device(self, batch):
        """将批次数据移动到指定设备"""
        def move_item(item):
            if isinstance(item, torch.Tensor):
                return item.to(self.device)
            elif isinstance(item, dict):
                return {k: move_item(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [move_item(x) for x in item]
            else:
                return item
        
        return move_item(batch)
    
    @torch.no_grad()
    def _process_single_batch(self, batch, batch_idx):
        """处理单个批次"""
        results = {'similarities': [], 'loss': 0.0}
        
        for modality_scope, dataset_batch in batch.items():
            self.model.modality_scope = modality_scope
            
            try:
                # 计算输入embeddings
                perceptual_emb, latent_goal, image_latent_goal = self.model.compute_input_embeddings(dataset_batch)
                
                # 如果启用了latent motion预测
                if self.model.latent_motion_pred and self.model.use_pretrained_encoder:
                    # 生成ground truth latent motion indices
                    gt_latent_motion = self._generate_ground_truth_latent_motion(dataset_batch)
                    
                    if gt_latent_motion is not None:
                        # 生成预测的latent motion
                        pred_latent_motion = self._generate_predicted_latent_motion(dataset_batch, gt_latent_motion)
                        
                        if pred_latent_motion is not None:
                            # 计算相似度
                            similarity = self._calculate_similarity(gt_latent_motion, pred_latent_motion)
                            results['similarities'].append(similarity)
                
                # 运行标准验证步骤
                val_output = self.model.validation_step({modality_scope: dataset_batch}, batch_idx)
                
                break  # 只处理第一个modality
                
            except Exception as e:
                print(f"处理modality {modality_scope} 时出错: {e}")
                continue
        
        return results if results['similarities'] else None
    
    def _generate_ground_truth_latent_motion(self, dataset_batch):
        """生成ground truth latent motion indices"""
        try:
            # 提取RGB观察
            t = dataset_batch['rgb_obs']['rgb_static'].shape[1]
            rgb_static = dataset_batch['rgb_obs']['rgb_static'][:, :-1] if t > 1 else dataset_batch['rgb_obs']['rgb_static']
            rgb_gripper = dataset_batch['rgb_obs']['rgb_gripper'][:, :-1] if t > 1 else dataset_batch['rgb_obs']['rgb_gripper']
            
            # 调整gripper图像大小
            rgb_gripper = self.model.interpolate_img(rgb_gripper, size=(224, 224))
            
            B, T = rgb_static.shape[:2]
            
            # 展平以便批处理
            rgb_static_flat = rgb_static.view(B * T, *rgb_static.shape[2:])
            rgb_gripper_flat = rgb_gripper.view(B * T, *rgb_gripper.shape[2:])
            
            # 1. 使用预训练ViT-MAE进行图像编码
            static_features = self.model.pretrained_image_encoder(rgb_static_flat).last_hidden_state
            gripper_features = self.model.pretrained_image_encoder(rgb_gripper_flat).last_hidden_state
            
            # 2. MFormer处理
            static_obs_features = self.model.pretrained_m_former(static_features).last_hidden_state
            gripper_obs_features = self.model.pretrained_m_former(gripper_features).last_hidden_state
            
            # 重新整形并准备3D融合
            static_obs_features = static_obs_features.view(B, T, *static_obs_features.shape[1:])
            gripper_obs_features = gripper_obs_features.view(B, T, *gripper_obs_features.shape[1:])
            
            # 3. 使用MFormer3D进行3D融合
            combined_features = torch.cat([static_obs_features, gripper_obs_features], dim=2)
            combined_features_flat = combined_features.view(B, -1, combined_features.shape[-1])
            fused_features = self.model.pretrained_m_former3d(combined_features_flat).last_hidden_state
            
            # 4. VQ下采样
            downsampled_features = self.model.pretrained_vq_down_resampler(fused_features)
            
            # 5. 向量量化
            vq_output = self.model.pretrained_vq(downsampled_features)
            latent_motion_indices = vq_output[1]  # 获取indices
            
            return latent_motion_indices
            
        except Exception as e:
            print(f"生成ground truth latent motion时出错: {e}")
            return None
    
    def _generate_predicted_latent_motion(self, dataset_batch, gt_latent_motion):
        """生成预测的latent motion"""
        try:
            # 计算组合特征和mask
            combined_features, combined_mask = self.model.compute_latent_goal_embeddings_and_mask(dataset_batch)
            
            # 使用motion transformer生成预测
            motion_results = self.model.motion_transformer(
                perceptual_features=combined_features,
                latent_motion_ids=None,  # 推理模式，不提供ground truth
                attention_mask=combined_mask,
                train=False
            )
            
            return motion_results.get('latent_motion_preds')
            
        except Exception as e:
            print(f"生成预测latent motion时出错: {e}")
            return None
    
    def _calculate_similarity(self, gt_latent, pred_latent):
        """计算相似度"""
        try:
            if gt_latent is None or pred_latent is None:
                return 0.0
            
            # 转换为相同类型和设备
            gt_latent = gt_latent.float()
            pred_latent = pred_latent.float()
            
            # 展平进行比较
            gt_flat = gt_latent.view(-1)
            pred_flat = pred_latent.view(-1)
            
            # 确保长度相同
            min_len = min(len(gt_flat), len(pred_flat))
            gt_flat = gt_flat[:min_len]
            pred_flat = pred_flat[:min_len]
            
            # 计算余弦相似度
            if min_len > 0:
                cosine_sim = F.cosine_similarity(
                    gt_flat.unsqueeze(0), 
                    pred_flat.unsqueeze(0)
                ).item()
                return cosine_sim
            else:
                return 0.0
                
        except Exception as e:
            print(f"计算相似度时出错: {e}")
            return 0.0
    
    def _save_results(self):
        """保存结果"""
        output_dir = "./mdt_test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数值结果
        results_file = os.path.join(output_dir, "validation_results.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("MDT Latent Action验证结果\n")
            f.write("=" * 40 + "\n")
            
            if self.results['similarities']:
                similarities = self.results['similarities']
                f.write(f"总样本数: {len(similarities)}\n")
                f.write(f"平均余弦相似度: {np.mean(similarities):.4f}\n")
                f.write(f"标准差: {np.std(similarities):.4f}\n")
                f.write(f"最小值: {np.min(similarities):.4f}\n")
                f.write(f"最大值: {np.max(similarities):.4f}\n")
                f.write(f"中位数: {np.median(similarities):.4f}\n")
            
            f.write(f"\n批次详细信息:\n")
            for batch_info in self.results['batch_info']:
                f.write(f"批次 {batch_info['batch_idx']}: 平均相似度 = {batch_info['avg_similarity']:.4f}\n")
        
        # 保存相似度分布图
        if self.results['similarities']:
            plt.figure(figsize=(10, 6))
            plt.hist(self.results['similarities'], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('余弦相似度')
            plt.ylabel('频次')
            plt.title('Latent Action预测相似度分布')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "similarity_distribution.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"结果已保存到: {output_dir}")


def main():
    """主函数"""
    
    # 配置参数 - 请根据你的实际路径修改
    MODEL_CKPT_PATH = "/home/hlwang/mdt_policy/checkpoints/your_model.ckpt"
    CONFIG_PATH = "/home/hlwang/mdt_policy/conf/config_latent.yaml"
    
    # 检查文件是否存在
    if not os.path.exists(MODEL_CKPT_PATH):
        print(f"模型checkpoint文件不存在: {MODEL_CKPT_PATH}")
        print("请更新MODEL_CKPT_PATH为你的模型文件路径")
        
        # 尝试查找可能的checkpoint文件
        possible_dirs = [
            "/home/hlwang/mdt_policy/checkpoints",
            "/home/hlwang/mdt_policy/outputs",
            "/home/hlwang/results"
        ]
        
        print("\n正在搜索可能的checkpoint文件...")
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                ckpt_files = [f for f in os.listdir(dir_path) if f.endswith('.ckpt')]
                if ckpt_files:
                    print(f"在 {dir_path} 中发现的checkpoint文件:")
                    for ckpt in ckpt_files[:5]:  # 只显示前5个
                        print(f"  - {os.path.join(dir_path, ckpt)}")
        return
    
    if not os.path.exists(CONFIG_PATH):
        print(f"配置文件不存在: {CONFIG_PATH}")
        print("请更新CONFIG_PATH为你的配置文件路径")
        return
    
    # 初始化测试器
    tester = SimpleMDTTester(
        model_ckpt_path=MODEL_CKPT_PATH,
        config_path=CONFIG_PATH,
        device="auto",
        num_test_batches=5,  # 测试5个批次
    )
    
    # 加载模型
    if not tester.load_model_from_checkpoint():
        print("加载模型失败，退出测试")
        return
    
    # 设置数据模块
    if not tester.setup_datamodule():
        print("设置datamodule失败，退出测试")
        return
    
    # 运行验证测试
    success = tester.run_validation_test()
    
    if success:
        print("\n测试完成! 请查看 ./mdt_test_results/ 目录下的结果文件")
    else:
        print("测试失败")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main()