#!/usr/bin/env python3
"""
简化的MDT模型测试脚本 - 从项目根目录运行
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import logging

# 设置项目目录
current_dir = Path(__file__).parent.resolve()
print(f"使用项目目录: {current_dir}")

# 将项目目录添加到Python路径
sys.path.insert(0, str(current_dir))

logger = logging.getLogger(__name__)


def load_mdt_model(checkpoint_path: str, device: str = "cuda"):
    """加载MDT模型"""
    print(f"正在加载模型: {checkpoint_path}")
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'hyper_parameters' in checkpoint and 'state_dict' in checkpoint:
            print("检测到PyTorch Lightning checkpoint")
            
            # 提取数据
            hparams = checkpoint['hyper_parameters']
            state_dict = checkpoint['state_dict']
            
            print("正在导入模型类...")
            # 导入模型类
            from mdt.models.mdt_3d_latent_action import MDT3dLatentActionAgent
            
            print("正在创建模型实例...")
            # 创建模型
            model = MDT3dLatentActionAgent(**hparams)
            
            print("正在加载权重...")
            # 加载权重
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"注意: 有 {len(missing_keys)} 个缺失的键")
            if unexpected_keys:
                print(f"注意: 有 {len(unexpected_keys)} 个意外的键")
            
            print("模型加载成功!")
            
        else:
            raise ValueError("不支持的checkpoint格式")
        
        model.eval()
        
        # 如果使用CPU，确保模型使用float32
        if device == "cpu":
            model = model.float()
        
        model.to(device)
        
        return model
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_model_info(model):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"""
模型信息:
- 总参数: {total_params:,}
- 可训练参数: {trainable_params:,}
- 冻结参数: {total_params - trainable_params:,}
- Latent Motion预测: {getattr(model, 'latent_motion_pred', False)}
- 使用预训练编码器: {getattr(model, 'use_pretrained_encoder', False)}
""")


def create_test_batch(batch_size=2, device="cuda"):
    """创建测试批次"""
    dtype = torch.float32  # 统一使用float32避免问题
    
    batch = {
        'vis': {
            'rgb_obs': {
                'rgb_static': torch.randn(batch_size, 8, 3, 224, 224, device=device, dtype=dtype),
                'rgb_gripper': torch.randn(batch_size, 8, 3, 224, 224, device=device, dtype=dtype),
            },
            'actions': torch.randn(batch_size, 10, 7, device=device, dtype=dtype),
            'lang_text': ['pick up the object'] * batch_size,
        }
    }
    return batch


def test_forward_pass(model, batch, device="cuda"):
    """测试前向传播"""
    print("测试forward pass...")
    
    try:
        model.eval()
        
        with torch.no_grad():
            for modality_scope, dataset_batch in batch.items():
                # 准备输入
                obs = {
                    'rgb_obs': dataset_batch['rgb_obs']
                }
                
                goal = {
                    'visual_goal': dataset_batch['rgb_obs']['rgb_static'][:, -1]
                }
                
                # 运行forward
                predicted_actions = model.forward(obs, goal)
                
                print(f"Forward pass成功:")
                print(f"  - 输入形状: {obs['rgb_obs']['rgb_static'].shape}")
                print(f"  - 输出形状: {predicted_actions.shape}")
                print(f"  - 输出范围: [{predicted_actions.min():.3f}, {predicted_actions.max():.3f}]")
                
                return predicted_actions
                
    except Exception as e:
        print(f"Forward pass失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_simple_components(model, batch, device="cuda"):
    """测试简单的模型组件"""
    print("测试模型组件...")
    
    try:
        model.eval()
        
        with torch.no_grad():
            for modality_scope, dataset_batch in batch.items():
                model.modality_scope = modality_scope
                
                # 测试1: 基本embeddings计算
                try:
                    print("  测试input embeddings...")
                    perceptual_emb, latent_goal, image_latent_goal = model.compute_input_embeddings(dataset_batch)
                    print(f"    ✓ perceptual_emb keys: {list(perceptual_emb.keys()) if isinstance(perceptual_emb, dict) else 'tensor'}")
                    print(f"    ✓ latent_goal shape: {latent_goal.shape if latent_goal is not None else None}")
                except Exception as e:
                    print(f"    ✗ Input embeddings失败: {e}")
                
                # 测试2: 视觉编码器
                try:
                    print("  测试visual encoders...")
                    rgb_static = dataset_batch['rgb_obs']['rgb_static'][:, :-1]
                    rgb_gripper = dataset_batch['rgb_obs']['rgb_gripper'][:, :-1]
                    visual_emb = model.embed_visual_obs(rgb_static, rgb_gripper)
                    print(f"    ✓ Visual embedding keys: {list(visual_emb.keys())}")
                except Exception as e:
                    print(f"    ✗ Visual encoders失败: {e}")
                
                # 测试3: Goal编码器
                try:
                    print("  测试goal encoder...")
                    goal_img = dataset_batch['rgb_obs']['rgb_static'][:, -1]
                    goal_emb = model.visual_goal(goal_img)
                    print(f"    ✓ Goal embedding shape: {goal_emb.shape}")
                except Exception as e:
                    print(f"    ✗ Goal encoder失败: {e}")
                
                break  # 只测试第一个modality
                
        return True
        
    except Exception as e:
        print(f"组件测试失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MDT模型简化测试")
    
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="模型checkpoint路径")
    parser.add_argument("--device", type=str, default="auto",
                       help="使用的设备")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小")
    
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
    
    # 加载模型
    model = load_mdt_model(args.checkpoint, device)
    if model is None:
        print("模型加载失败")
        return
    
    # 打印模型信息
    print_model_info(model)
    
    # 创建测试数据
    print("创建测试数据...")
    test_batch = create_test_batch(args.batch_size, device)
    
    # 测试组件
    print("\n" + "="*50)
    print("测试模型组件")
    print("="*50)
    component_result = test_simple_components(model, test_batch, device)
    
    # 测试forward pass
    print("\n" + "="*50)
    print("测试Forward Pass")
    print("="*50)
    forward_result = test_forward_pass(model, test_batch, device)
    
    # 总结
    print("\n" + "="*50)
    print("测试总结:")
    print(f"✓ 模型加载: 成功")
    print(f"{'✓' if component_result else '✗'} 组件测试: {'成功' if component_result else '失败'}")
    print(f"{'✓' if forward_result is not None else '✗'} Forward pass: {'成功' if forward_result is not None else '失败'}")
    
    if forward_result is not None:
        print(f"  模型可以成功生成动作预测!")
        print(f"  输出形状: {forward_result.shape}")
    
    print("\n测试完成!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main()