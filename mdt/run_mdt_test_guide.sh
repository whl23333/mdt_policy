#!/bin/bash
# MDT模型测试脚本运行指南

echo "==================================================="
echo "MDT Latent Action验证测试运行指南"
echo "==================================================="

# 设置基本路径
MODEL_DIR="/home/hlwang/mdt_policy"
CHECKPOINT_DIR="/home/hlwang/mdt_policy/checkpoints"
CONFIG_DIR="/home/hlwang/mdt_policy/conf"

echo ""
echo "1. 检查Python环境和依赖..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python3 -c "import sys; print(f'Python路径: {sys.executable}')"

echo ""
echo "2. 查找可用的checkpoint文件..."
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "发现的checkpoint文件:"
    find "$CHECKPOINT_DIR" -name "*.ckpt" -type f | head -5
else
    echo "checkpoint目录不存在，搜索其他位置..."
    find /home/hlwang -name "*.ckpt" -type f 2>/dev/null | head -5
fi

echo ""
echo "3. 查找配置文件..."
if [ -f "$CONFIG_DIR/config_latent.yaml" ]; then
    echo "找到配置文件: $CONFIG_DIR/config_latent.yaml"
else
    echo "搜索配置文件..."
    find "$CONFIG_DIR" -name "*.yaml" -type f 2>/dev/null | head -3
fi

echo ""
echo "==================================================="
echo "运行测试的方法:"
echo "==================================================="

echo ""
echo "方法1: 使用简化测试脚本 (推荐)"
echo "----------------------------------------"
echo "cd /home/hlwang"
echo "python test_mdt_latent_validation.py \\"
echo "  --checkpoint /path/to/your/model.ckpt \\"
echo "  --device cuda \\"
echo "  --batch_size 2 \\"
echo "  --num_batches 5 \\"
echo "  --output_dir ./test_results \\"
echo "  --use_mock_data"

echo ""
echo "方法2: 使用PyTorch Lightning Trainer"
echo "----------------------------------------"
echo "cd /home/hlwang"
echo "python test_mdt_with_trainer.py \\"
echo "  --checkpoint /path/to/your/model.ckpt \\"
echo "  --config /path/to/config.yaml \\"
echo "  --output_dir ./trainer_test_results \\"
echo "  --limit_batches 10"

echo ""
echo "方法3: 在Jupyter Notebook中交互式测试"
echo "----------------------------------------"
echo "# 创建notebook进行交互式测试"
echo "jupyter notebook"

echo ""
echo "==================================================="
echo "注意事项:"
echo "==================================================="
echo "1. 确保模型checkpoint文件路径正确"
echo "2. 如果使用方法2，需要提供对应的配置文件"
echo "3. 第一次运行建议使用 --use_mock_data 选项"
echo "4. 如果遇到导入错误，检查PYTHONPATH设置"
echo "5. GPU内存不足时可以减少batch_size"

echo ""
echo "==================================================="
echo "示例命令 (请根据实际路径修改):"
echo "==================================================="

# 查找最新的checkpoint文件
LATEST_CKPT=$(find /home/hlwang -name "*.ckpt" -type f 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "使用发现的checkpoint文件:"
    echo "python test_mdt_latent_validation.py \\"
    echo "  --checkpoint \"$LATEST_CKPT\" \\"
    echo "  --device cuda \\"
    echo "  --batch_size 2 \\"
    echo "  --num_batches 3 \\"
    echo "  --use_mock_data"
else
    echo "未找到checkpoint文件，请手动指定:"
    echo "python test_mdt_latent_validation.py \\"
    echo "  --checkpoint /path/to/your/model.ckpt \\"
    echo "  --device cuda \\"
    echo "  --batch_size 2 \\"
    echo "  --num_batches 3 \\"
    echo "  --use_mock_data"
fi

echo ""
echo "==================================================="
echo "故障排除:"
echo "==================================================="
echo "1. 如果遇到 'Import error':"
echo "   export PYTHONPATH=/home/hlwang/mdt_policy:\$PYTHONPATH"
echo ""
echo "2. 如果CUDA内存不足:"
echo "   使用 --device cpu 或减少 --batch_size"
echo ""
echo "3. 如果模型加载失败:"
echo "   检查checkpoint文件是否完整，尝试不同的加载方式"
echo ""
echo "4. 查看详细错误信息:"
echo "   在命令后添加 2>&1 | tee test_log.txt"

echo ""
echo "==================================================="
echo "预期输出:"
echo "==================================================="
echo "- 模型信息 (参数数量、组件状态等)"
echo "- 每个批次的相似度和准确率"
echo "- 平均性能指标"
echo "- 测试结果文件保存在指定目录"
echo ""
echo "运行完成后，检查输出目录中的:"
echo "- test_results.json: 数值结果"
echo "- test_report.txt: 详细报告"
echo "- similarity_distribution.png: 相似度分布图(如果生成)"