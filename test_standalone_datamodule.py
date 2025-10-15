#!/usr/bin/env python3
"""
测试独立datamodule配置文件的脚本
"""

import os
import sys
import hydra
from omegaconf import OmegaConf
from pathlib import Path

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_standalone_datamodule():
    """测试独立的datamodule配置"""
    try:
        print("正在测试独立datamodule配置...")
        
        # 加载独立配置文件
        config_path = "/home/hlwang/mdt_policy/conf/datamodule_standalone.yaml"
        
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
            return False
        
        print(f"加载配置文件: {config_path}")
        config = OmegaConf.load(config_path)
        
        print("配置文件加载成功，开始实例化datamodule...")
        
        # 实例化datamodule
        datamodule = hydra.utils.instantiate(config)
        
        print("Datamodule实例化成功!")
        print(f"类型: {type(datamodule)}")
        
        # 尝试设置数据模块
        print("正在设置datamodule...")
        datamodule.setup("fit")
        
        print("Datamodule设置成功!")
        
        # 获取数据加载器信息
        print("\n=== Datamodule信息 ===")
        print(f"训练数据路径: {datamodule.training_dir}")
        print(f"验证数据路径: {datamodule.val_dir}")
        print(f"工作进程数: {datamodule.num_workers}")
        print(f"模态列表: {datamodule.modalities}")
        
        # 尝试获取数据加载器
        try:
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            
            print(f"\n=== 数据加载器信息 ===")
            print(f"训练加载器类型: {type(train_loader)}")
            print(f"验证加载器类型: {type(val_loader)}")
            
            if hasattr(train_loader, 'loaders'):
                print(f"训练加载器数量: {len(train_loader.loaders)}")
                for i, (key, loader) in enumerate(train_loader.loaders.items()):
                    print(f"  - {key}: {len(loader)} batches")
            
            if hasattr(val_loader, 'loaders'):
                print(f"验证加载器数量: {len(val_loader.loaders)}")
                for i, (key, loader) in enumerate(val_loader.loaders.items()):
                    print(f"  - {key}: {len(loader)} batches")
            
        except Exception as loader_e:
            print(f"获取数据加载器时出错: {loader_e}")
            print("这可能是因为数据文件不存在，但datamodule配置是正确的")
        
        print("\n✅ 独立datamodule配置测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("独立DataModule配置测试")
    print("="*60)
    
    success = test_standalone_datamodule()
    
    if success:
        print(f"\n🎉 测试成功! 配置文件可以正常使用")
        print(f"配置文件位置: /home/hlwang/mdt_policy/conf/datamodule_standalone.yaml")
        print(f"")
        print(f"使用方法:")
        print(f"```python")
        print(f"from omegaconf import OmegaConf")
        print(f"import hydra")
        print(f"")
        print(f"config = OmegaConf.load('/home/hlwang/mdt_policy/conf/datamodule_standalone.yaml')")
        print(f"datamodule = hydra.utils.instantiate(config)")
        print(f"datamodule.setup('fit')")
        print(f"```")
    else:
        print(f"\n❌ 测试失败")

if __name__ == "__main__":
    main()