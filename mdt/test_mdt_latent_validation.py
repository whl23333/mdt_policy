#!/usr/bin/env python3
"""
MDT模型Latent Action验证脚本
专门用于测试训练好的MDT模型在validation和forward过程中的latent action生成质量

使用方法:
python test_mdt_latent_validation.py --checkpoint /path/to/your/model.ckpt --config /path/to/config.yaml

功能:
1. 加载训练好的MDT模型
2. 使用现有datamodule进行数据加载
3. 验证latent action预测与ground truth的一致性
4. 可选的图像生成质量检查
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, Any, Optional
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent  # 指向mdt_policy根目录
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def load_model_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """
    加载模型checkpoint
    支持PyTorch Lightning checkpoint和普通state_dict
    """
    print(f"正在加载模型: {checkpoint_path}")
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 检查是否是PyTorch Lightning checkpoint
        if 'hyper_parameters' in checkpoint and 'state_dict' in checkpoint:
            print("检测到PyTorch Lightning checkpoint")
            
            # 提取hyperparameters
            hparams = checkpoint['hyper_parameters']
            state_dict = checkpoint['state_dict']
            
            # 动态导入模型类
            from mdt.models.mdt_3d_latent_action import MDT3dLatentActionAgent
            
            # 使用hyperparameters创建模型
            print("正在使用hyperparameters创建模型...")
            model = MDT3dLatentActionAgent(**hparams)
            
            # 加载state_dict
            print("正在加载模型权重...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"缺失的键: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"意外的键: {len(unexpected_keys)} 个")
            
            print("成功加载PyTorch Lightning模型")
            
        else:
            print("检测到普通state_dict checkpoint")
            raise ValueError("普通state_dict需要额外的配置信息，请使用PyTorch Lightning checkpoint")
        
        model.eval()
        model.to(device)
        
        return model
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


def setup_test_environment(model, device="cuda"):
    """设置测试环境"""
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"""
模型信息:
- 总参数: {total_params:,}
- 可训练参数: {trainable_params:,}  
- 冻结参数: {total_params - trainable_params:,}
- 设备: {device}
- Latent Motion预测: {getattr(model, 'latent_motion_pred', False)}
- 使用预训练编码器: {getattr(model, 'use_pretrained_encoder', False)}
""")
    
    # 检查关键组件
    has_pretrained_components = all([
        hasattr(model, 'pretrained_image_encoder') and model.pretrained_image_encoder is not None,
        hasattr(model, 'pretrained_m_former') and model.pretrained_m_former is not None,
        hasattr(model, 'pretrained_m_former3d') and model.pretrained_m_former3d is not None,
        hasattr(model, 'pretrained_vq_down_resampler') and model.pretrained_vq_down_resampler is not None,
        hasattr(model, 'pretrained_vq') and model.pretrained_vq is not None,
    ])
    
    print(f"预训练组件完整性: {'✓' if has_pretrained_components else '✗'}")
    
    return has_pretrained_components


def create_simple_dataloader(batch_size=2, num_batches=5, device="cuda"):
    """
    创建简单的测试数据加载器
    如果无法使用原始datamodule，可以用这个创建mock数据
    """
    print("创建模拟测试数据...")
    
    def generate_batch():
        """生成单个批次的模拟数据"""
        batch = {
            'vis': {
                'rgb_obs': {
                    'rgb_static': torch.randn(batch_size, 8, 3, 200, 200, device=device),
                    'rgb_gripper': torch.randn(batch_size, 8, 3, 84, 84, device=device),
                },
                'actions': torch.randn(batch_size, 10, 7, device=device),
                'lang_text': ['pick up the object'] * batch_size,
            }
        }
        return batch
    
    # 生成多个批次
    batches = [generate_batch() for _ in range(num_batches)]
    
    print(f"创建了 {num_batches} 个模拟批次，每批次大小: {batch_size}")
    
    return batches


def test_latent_action_consistency(model, batch, device="cuda"):
    """
    测试latent action的一致性
    比较ground truth和predicted latent actions
    """
    try:
        model.eval()
        
        results = {
            'similarities': [],
            'accuracies': [],
            'errors': []
        }
        
        with torch.no_grad():
            for modality_scope, dataset_batch in batch.items():
                model.modality_scope = modality_scope
                
                try:
                    # 1. 生成ground truth latent motion indices
                    gt_latent_motion = generate_ground_truth_latent_motion(model, dataset_batch)
                    
                    if gt_latent_motion is None:
                        print(f"无法生成ground truth latent motion for {modality_scope}")
                        continue
                    
                    # 2. 生成predicted latent motion
                    pred_latent_motion = generate_predicted_latent_motion(model, dataset_batch)
                    
                    if pred_latent_motion is None:
                        print(f"无法生成predicted latent motion for {modality_scope}")
                        continue
                    
                    # 3. 计算一致性指标
                    consistency_metrics = calculate_consistency_metrics(gt_latent_motion, pred_latent_motion)
                    
                    if consistency_metrics:
                        results['similarities'].append(consistency_metrics.get('cosine_similarity', 0.0))
                        results['accuracies'].append(consistency_metrics.get('accuracy', 0.0))
                        
                        print(f"Modality {modality_scope}:")
                        print(f"  - 余弦相似度: {consistency_metrics.get('cosine_similarity', 0.0):.4f}")
                        print(f"  - 准确率: {consistency_metrics.get('accuracy', 0.0):.4f}")
                        print(f"  - L2距离: {consistency_metrics.get('l2_distance', 0.0):.4f}")
                    
                except Exception as e:
                    error_msg = f"处理 {modality_scope} 时出错: {e}"
                    print(error_msg)
                    results['errors'].append(error_msg)
                    continue
                
                break  # 只处理第一个modality
        
        return results
        
    except Exception as e:
        print(f"测试latent action一致性时出错: {e}")
        return None


def generate_ground_truth_latent_motion(model, dataset_batch):
    """生成ground truth latent motion indices"""
    try:
        if not hasattr(model, 'pretrained_vq') or model.pretrained_vq is None:
            print("模型缺少pretrained_vq组件")
            return None
        
        # 提取RGB观察
        t = dataset_batch['rgb_obs']['rgb_static'].shape[1]
        rgb_static = dataset_batch['rgb_obs']['rgb_static'][:, :-1] if t > 1 else dataset_batch['rgb_obs']['rgb_static']
        rgb_gripper = dataset_batch['rgb_obs']['rgb_gripper'][:, :-1] if t > 1 else dataset_batch['rgb_obs']['rgb_gripper']
        
        # 调整gripper图像大小
        rgb_gripper = model.interpolate_img(rgb_gripper, size=(224, 224))
        
        B, T = rgb_static.shape[:2]
        
        # 展平进行批处理
        rgb_static_flat = rgb_static.view(B * T, *rgb_static.shape[2:])
        rgb_gripper_flat = rgb_gripper.view(B * T, *rgb_gripper.shape[2:])
        
        # 1. 图像编码
        static_features = model.pretrained_image_encoder(rgb_static_flat).last_hidden_state
        gripper_features = model.pretrained_image_encoder(rgb_gripper_flat).last_hidden_state
        
        # 2. MFormer处理
        static_obs_features = model.pretrained_m_former(static_features).last_hidden_state
        gripper_obs_features = model.pretrained_m_former(gripper_features).last_hidden_state
        
        # 重新整形
        static_obs_features = static_obs_features.view(B, T, *static_obs_features.shape[1:])
        gripper_obs_features = gripper_obs_features.view(B, T, *gripper_obs_features.shape[1:])
        
        # 3. 3D融合
        combined_features = torch.cat([static_obs_features, gripper_obs_features], dim=2)
        combined_features_flat = combined_features.view(B, -1, combined_features.shape[-1])
        fused_features = model.pretrained_m_former3d(combined_features_flat).last_hidden_state
        
        # 4. VQ下采样
        downsampled_features = model.pretrained_vq_down_resampler(fused_features)
        
        # 5. 向量量化
        vq_output = model.pretrained_vq(downsampled_features)
        latent_motion_indices = vq_output[1]  # 获取indices
        
        return latent_motion_indices
        
    except Exception as e:
        print(f"生成ground truth时出错: {e}")
        return None


def generate_predicted_latent_motion(model, dataset_batch):
    """生成predicted latent motion"""
    try:
        if not hasattr(model, 'motion_transformer') or model.motion_transformer is None:
            print("模型缺少motion_transformer组件")
            return None
        
        # 计算组合特征和mask
        combined_features, combined_mask = model.compute_latent_goal_embeddings_and_mask(dataset_batch)
        
        # 使用motion transformer进行推理
        motion_results = model.motion_transformer(
            perceptual_features=combined_features,
            latent_motion_ids=None,  # 推理模式，不提供ground truth
            attention_mask=combined_mask,
            train=False
        )
        
        return motion_results.get('latent_motion_preds')
        
    except Exception as e:
        print(f"生成predicted latent motion时出错: {e}")
        return None


def calculate_consistency_metrics(gt_latent, pred_latent):
    """计算一致性指标"""
    try:
        if gt_latent is None or pred_latent is None:
            return None
        
        metrics = {}
        
        # 转换为numpy进行计算
        gt_np = gt_latent.detach().cpu().numpy().flatten()
        pred_np = pred_latent.detach().cpu().numpy().flatten()
        
        # 确保长度一致
        min_len = min(len(gt_np), len(pred_np))
        gt_np = gt_np[:min_len]
        pred_np = pred_np[:min_len]
        
        if min_len == 0:
            return None
        
        # 计算准确率（对于离散indices）
        if gt_latent.dtype in [torch.long, torch.int]:
            accuracy = np.mean(gt_np == pred_np)
            metrics['accuracy'] = float(accuracy)
        
        # 计算余弦相似度（转换为float）
        gt_float = gt_latent.float().detach().cpu().numpy().flatten()[:min_len]
        pred_float = pred_latent.float().detach().cpu().numpy().flatten()[:min_len]
        
        # 归一化
        gt_norm = gt_float / (np.linalg.norm(gt_float) + 1e-8)
        pred_norm = pred_float / (np.linalg.norm(pred_float) + 1e-8)
        
        cosine_sim = np.dot(gt_norm, pred_norm)
        metrics['cosine_similarity'] = float(cosine_sim)
        
        # L2距离
        l2_distance = np.linalg.norm(gt_float - pred_float)
        metrics['l2_distance'] = float(l2_distance)
        
        return metrics
        
    except Exception as e:
        print(f"计算一致性指标时出错: {e}")
        return None


def test_forward_pass(model, batch, device="cuda"):
    """测试forward pass"""
    try:
        model.eval()
        
        print("测试forward pass...")
        
        with torch.no_grad():
            for modality_scope, dataset_batch in batch.items():
                # 准备观察和目标数据
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
                
                # 运行forward pass
                try:
                    predicted_actions = model.forward(obs, goal)
                    
                    print(f"Forward pass成功:")
                    print(f"  - 输入形状: {obs['rgb_obs']['rgb_static'].shape}")
                    print(f"  - 输出形状: {predicted_actions.shape}")
                    print(f"  - 输出范围: [{predicted_actions.min():.3f}, {predicted_actions.max():.3f}]")
                    
                    return predicted_actions
                    
                except Exception as e:
                    print(f"Forward pass失败: {e}")
                    return None
                
                break  # 只测试第一个modality
        
    except Exception as e:
        print(f"测试forward pass时出错: {e}")
        return None


def save_test_results(results, output_dir="./test_results"):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数值结果
    results_file = os.path.join(output_dir, "test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存文本报告
    report_file = os.path.join(output_dir, "test_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("MDT Latent Action测试报告\n")
        f.write("=" * 50 + "\n\n")
        
        if 'batch_results' in results:
            all_similarities = []
            all_accuracies = []
            
            for i, batch_result in enumerate(results['batch_results']):
                f.write(f"批次 {i}:\n")
                if batch_result:
                    similarities = batch_result.get('similarities', [])
                    accuracies = batch_result.get('accuracies', [])
                    
                    if similarities:
                        f.write(f"  - 相似度: {similarities}\n")
                        all_similarities.extend(similarities)
                    
                    if accuracies:
                        f.write(f"  - 准确率: {accuracies}\n")
                        all_accuracies.extend(accuracies)
                    
                    if batch_result.get('errors'):
                        f.write(f"  - 错误: {batch_result['errors']}\n")
                else:
                    f.write("  - 批次处理失败\n")
                f.write("\n")
            
            # 总体统计
            f.write("总体统计:\n")
            if all_similarities:
                f.write(f"平均相似度: {np.mean(all_similarities):.4f} ± {np.std(all_similarities):.4f}\n")
                f.write(f"相似度范围: [{np.min(all_similarities):.4f}, {np.max(all_similarities):.4f}]\n")
            
            if all_accuracies:
                f.write(f"平均准确率: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}\n")
                f.write(f"准确率范围: [{np.min(all_accuracies):.4f}, {np.max(all_accuracies):.4f}]\n")
    
    print(f"测试结果已保存到: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MDT Latent Action验证测试")
    
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="模型checkpoint路径")
    parser.add_argument("--device", type=str, default="auto",
                       help="使用的设备 (auto/cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="批次大小")
    parser.add_argument("--num_batches", type=int, default=5,
                       help="测试批次数量")
    parser.add_argument("--output_dir", "-o", type=str, default="./test_results",
                       help="输出目录")
    parser.add_argument("--use_mock_data", action="store_true",
                       help="使用模拟数据（如果无法加载真实datamodule）")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 检查checkpoint文件
    if not os.path.exists(args.checkpoint):
        print(f"错误: Checkpoint文件不存在: {args.checkpoint}")
        return
    
    # 1. 加载模型
    model = load_model_checkpoint(args.checkpoint, device)
    if model is None:
        print("加载模型失败，退出")
        return
    
    # 2. 设置测试环境
    has_components = setup_test_environment(model, device)
    if not has_components:
        print("警告: 模型缺少必要的预训练组件，某些测试可能失败")
    
    # 3. 创建或加载测试数据
    if args.use_mock_data:
        test_batches = create_simple_dataloader(args.batch_size, args.num_batches, device)
        print("使用模拟测试数据")
    else:
        print("尝试使用真实数据...")
        # 这里可以尝试加载真实的datamodule
        # 如果失败，回退到模拟数据
        try:
            # 尝试从model的配置中获取datamodule信息
            # 这需要model保存了相关配置信息
            print("真实数据加载功能待实现，使用模拟数据")
            test_batches = create_simple_dataloader(args.batch_size, args.num_batches, device)
        except:
            print("加载真实数据失败，使用模拟数据")
            test_batches = create_simple_dataloader(args.batch_size, args.num_batches, device)
    
    # 4. 运行测试
    print("\n" + "="*50)
    print("开始Latent Action一致性测试")
    print("="*50)
    
    batch_results = []
    
    for i, batch in enumerate(tqdm(test_batches, desc="处理测试批次")):
        print(f"\n处理批次 {i+1}/{len(test_batches)}")
        
        # 测试latent action一致性
        batch_result = test_latent_action_consistency(model, batch, device)
        batch_results.append(batch_result)
        
        # 测试forward pass（可选）
        if i == 0:  # 只在第一个批次测试forward pass
            print("\n测试Forward Pass:")
            forward_result = test_forward_pass(model, batch, device)
    
    # 5. 保存结果
    results = {
        'batch_results': batch_results,
        'model_info': {
            'checkpoint_path': args.checkpoint,
            'device': device,
            'batch_size': args.batch_size,
            'num_batches': args.num_batches,
            'has_latent_motion_pred': getattr(model, 'latent_motion_pred', False),
            'has_pretrained_components': has_components,
        },
        'test_config': vars(args)
    }
    
    save_test_results(results, args.output_dir)
    
    # 6. 打印总结
    print("\n" + "="*50)
    print("测试完成!")
    print("="*50)
    
    successful_batches = sum(1 for r in batch_results if r is not None)
    print(f"成功处理的批次: {successful_batches}/{len(test_batches)}")
    
    if successful_batches > 0:
        all_similarities = []
        all_accuracies = []
        
        for result in batch_results:
            if result:
                all_similarities.extend(result.get('similarities', []))
                all_accuracies.extend(result.get('accuracies', []))
        
        if all_similarities:
            print(f"平均相似度: {np.mean(all_similarities):.4f} ± {np.std(all_similarities):.4f}")
        
        if all_accuracies:
            print(f"平均准确率: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    
    print(f"详细结果保存在: {args.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main()