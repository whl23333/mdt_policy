#!/usr/bin/env python3
"""
MDT测试配置助手
帮助检查环境和查找必要的文件
"""

import os
import sys
import glob
from pathlib import Path

def check_environment():
    """检查Python环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 检查关键包
    packages = ['torch', 'numpy', 'matplotlib', 'tqdm']
    for pkg in packages:
        try:
            exec(f"import {pkg}")
            print(f"✓ {pkg}: 已安装")
        except ImportError:
            print(f"✗ {pkg}: 未安装")
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA: 可用 (设备数量: {torch.cuda.device_count()})")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("✗ CUDA: 不可用")
    except:
        print("✗ PyTorch: 无法检查CUDA状态")


def find_checkpoints(search_dirs=None):
    """查找checkpoint文件"""
    print("\n" + "=" * 60)
    print("查找Checkpoint文件")
    print("=" * 60)
    
    if search_dirs is None:
        search_dirs = [
            "/home/hlwang/mdt_policy/checkpoints",
            "/home/hlwang/mdt_policy/outputs",
            "/home/hlwang/results",
            "/home/hlwang",
        ]
    
    found_files = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"\n搜索目录: {search_dir}")
            
            # 查找.ckpt文件
            ckpt_pattern = os.path.join(search_dir, "**/*.ckpt")
            ckpt_files = glob.glob(ckpt_pattern, recursive=True)
            
            if ckpt_files:
                print(f"  发现 {len(ckpt_files)} 个checkpoint文件:")
                for i, ckpt in enumerate(ckpt_files[:5]):  # 只显示前5个
                    file_size = os.path.getsize(ckpt) / (1024*1024)  # MB
                    print(f"    {i+1}. {ckpt} ({file_size:.1f} MB)")
                    found_files.append(ckpt)
                
                if len(ckpt_files) > 5:
                    print(f"    ... 还有 {len(ckpt_files) - 5} 个文件")
            else:
                print("  未找到checkpoint文件")
        else:
            print(f"目录不存在: {search_dir}")
    
    return found_files


def find_configs(search_dirs=None):
    """查找配置文件"""
    print("\n" + "=" * 60)
    print("查找配置文件")
    print("=" * 60)
    
    if search_dirs is None:
        search_dirs = [
            "/home/hlwang/mdt_policy/conf",
            "/home/hlwang/mdt_policy/config",
            "/home/hlwang/mdt_policy",
        ]
    
    found_configs = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"\n搜索目录: {search_dir}")
            
            # 查找yaml配置文件
            yaml_pattern = os.path.join(search_dir, "**/*.yaml")
            yaml_files = glob.glob(yaml_pattern, recursive=True)
            
            if yaml_files:
                print(f"  发现 {len(yaml_files)} 个YAML配置文件:")
                for i, config in enumerate(yaml_files[:5]):
                    print(f"    {i+1}. {config}")
                    found_configs.append(config)
                
                if len(yaml_files) > 5:
                    print(f"    ... 还有 {len(yaml_files) - 5} 个文件")
            else:
                print("  未找到YAML配置文件")
        else:
            print(f"目录不存在: {search_dir}")
    
    return found_configs


def check_mdt_code():
    """检查MDT代码是否可访问"""
    print("\n" + "=" * 60)
    print("检查MDT代码")
    print("=" * 60)
    
    # 检查主要的MDT文件
    mdt_files = [
        "/home/hlwang/mdt_policy/mdt/models/mdt_3d_latent_action.py",
        "/home/hlwang/mdt_policy/mdt/__init__.py",
        "/home/hlwang/mdt_policy/mdt/models/__init__.py",
    ]
    
    for file_path in mdt_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
    
    # 尝试导入MDT模块
    mdt_path = "/home/hlwang/mdt_policy"
    if mdt_path not in sys.path:
        sys.path.insert(0, mdt_path)
    
    try:
        from mdt.models.mdt_3d_latent_action import MDT3dLatentActionAgent
        print("✓ 成功导入 MDT3dLatentActionAgent")
        return True
    except ImportError as e:
        print(f"✗ 导入MDT模块失败: {e}")
        return False


def generate_test_commands(checkpoint_files, config_files):
    """生成测试命令"""
    print("\n" + "=" * 60)
    print("推荐的测试命令")
    print("=" * 60)
    
    if not checkpoint_files:
        print("没有找到checkpoint文件，无法生成命令")
        return
    
    # 选择最新的checkpoint
    latest_ckpt = max(checkpoint_files, key=lambda x: os.path.getmtime(x))
    print(f"推荐使用最新的checkpoint: {latest_ckpt}")
    
    print(f"\n1. 基础测试命令 (使用模拟数据):")
    print(f"cd /home/hlwang")
    print(f"python test_mdt_latent_validation.py \\")
    print(f"  --checkpoint \"{latest_ckpt}\" \\")
    print(f"  --device cuda \\")
    print(f"  --batch_size 2 \\")
    print(f"  --num_batches 3 \\")
    print(f"  --use_mock_data")
    
    if config_files:
        # 寻找最相关的配置文件
        relevant_configs = [c for c in config_files if 'config' in os.path.basename(c).lower()]
        if relevant_configs:
            config_file = relevant_configs[0]
            print(f"\n2. 使用真实配置的测试命令:")
            print(f"cd /home/hlwang")
            print(f"python test_mdt_with_trainer.py \\")
            print(f"  --checkpoint \"{latest_ckpt}\" \\")
            print(f"  --config \"{config_file}\" \\")
            print(f"  --output_dir ./trainer_test_results \\")
            print(f"  --limit_batches 5")
    
    print(f"\n3. 调试模式 (输出详细信息):")
    print(f"cd /home/hlwang")
    print(f"python test_mdt_latent_validation.py \\")
    print(f"  --checkpoint \"{latest_ckpt}\" \\")
    print(f"  --device cpu \\")
    print(f"  --batch_size 1 \\")
    print(f"  --num_batches 1 \\")
    print(f"  --use_mock_data 2>&1 | tee debug_log.txt")


def create_setup_script():
    """创建环境设置脚本"""
    setup_script = """#!/bin/bash
# MDT测试环境设置脚本

echo "设置MDT测试环境..."

# 设置Python路径
export PYTHONPATH="/home/hlwang/mdt_policy:$PYTHONPATH"
echo "PYTHONPATH已设置为: $PYTHONPATH"

# 创建测试目录
mkdir -p /home/hlwang/test_results
mkdir -p /home/hlwang/trainer_test_results

# 设置CUDA相关环境变量
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES已设置为: $CUDA_VISIBLE_DEVICES"

echo "环境设置完成!"
echo "现在可以运行测试命令了。"
"""
    
    script_path = "/home/hlwang/setup_mdt_test_env.sh"
    with open(script_path, 'w') as f:
        f.write(setup_script)
    
    os.chmod(script_path, 0o755)
    print(f"\n环境设置脚本已保存到: {script_path}")
    print("运行以下命令来设置环境:")
    print(f"source {script_path}")


def main():
    """主函数"""
    print("MDT测试配置助手")
    print("=" * 60)
    
    # 1. 检查环境
    check_environment()
    
    # 2. 查找文件
    checkpoint_files = find_checkpoints()
    config_files = find_configs()
    
    # 3. 检查代码
    mdt_available = check_mdt_code()
    
    # 4. 生成命令
    if checkpoint_files and mdt_available:
        generate_test_commands(checkpoint_files, config_files)
    else:
        print("\n" + "=" * 60)
        print("问题诊断")
        print("=" * 60)
        
        if not checkpoint_files:
            print("- 未找到checkpoint文件")
            print("  请确保你有已训练的模型文件")
        
        if not mdt_available:
            print("- MDT代码不可访问")
            print("  请检查代码路径和PYTHONPATH设置")
    
    # 5. 创建设置脚本
    create_setup_script()
    
    print("\n" + "=" * 60)
    print("配置检查完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()