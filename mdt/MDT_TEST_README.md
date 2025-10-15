# MDT模型Latent Action验证测试指南

## 概述

本测试套件专门用于验证训练好的MDT模型在生成latent action时的质量和一致性。主要功能包括：

1. **Latent Action一致性测试**: 比较模型预测的latent action与ground truth的相似度
2. **Forward Pass验证**: 测试模型的推理能力
3. **可选的图像生成质量检查**: 如果有预训练decoder，可以生成图像进行质量评估

## 文件说明

### 核心测试脚本

1. **`test_mdt_latent_validation.py`** (推荐)
   - 简化的测试脚本，适合快速验证
   - 支持模拟数据和真实数据
   - 输出详细的相似度和准确率指标

2. **`test_mdt_with_trainer.py`**
   - 使用PyTorch Lightning Trainer的完整测试
   - 需要完整的配置文件
   - 更接近训练时的环境

3. **`test_mdt_simple.py`**
   - 最基础的测试脚本
   - 适合调试和理解测试流程

### 辅助工具

4. **`mdt_test_helper.py`**
   - 配置助手，帮助查找checkpoint和配置文件
   - 环境检查和问题诊断

5. **`run_mdt_test_guide.sh`**
   - 完整的运行指南脚本
   - 包含示例命令和故障排除

## 快速开始

### 步骤1: 环境准备

```bash
# 运行配置助手
cd /home/hlwang
python mdt_test_helper.py
```

这将检查你的环境并查找可用的checkpoint文件。

### 步骤2: 基础测试 (推荐)

```bash
# 使用模拟数据进行快速测试
python test_mdt_latent_validation.py \
  --checkpoint /path/to/your/model.ckpt \
  --device cuda \
  --batch_size 2 \
  --num_batches 3 \
  --use_mock_data
```

### 步骤3: 完整测试 (如果有配置文件)

```bash
# 使用真实数据和配置
python test_mdt_with_trainer.py \
  --checkpoint /path/to/your/model.ckpt \
  --config /path/to/config.yaml \
  --output_dir ./test_results \
  --limit_batches 5
```

## 参数说明

### 通用参数

- `--checkpoint` / `-c`: 模型checkpoint文件路径 (必需)
- `--device`: 使用的设备 (`auto`/`cuda`/`cpu`)
- `--batch_size`: 批次大小 (建议从2开始)
- `--num_batches`: 测试批次数量
- `--output_dir` / `-o`: 结果输出目录

### 特殊参数

- `--use_mock_data`: 使用模拟数据 (推荐用于初次测试)
- `--config`: 配置文件路径 (某些脚本需要)
- `--limit_batches`: 限制验证批次数量

## 输出说明

### 控制台输出

测试运行时会显示：
- 模型信息 (参数数量、组件状态)
- 每个批次的处理结果
- 相似度和准确率指标
- 平均性能统计

### 输出文件

测试完成后会在输出目录生成：
- `test_results.json`: 数值结果
- `test_report.txt`: 详细文本报告  
- `similarity_distribution.png`: 相似度分布图 (如果生成)
- CSV日志文件 (使用trainer脚本时)

## 性能指标说明

### 相似度指标

- **余弦相似度**: 衡量预测和真实latent action的方向相似性
  - 范围: [-1, 1]，越接近1越好
  - 适用于连续值比较

- **准确率**: 对于离散latent action indices的精确匹配率
  - 范围: [0, 1]，越接近1越好
  - 适用于离散值比较

- **L2距离**: 欧几里得距离，衡量数值差异
  - 越小越好
  - 适用于连续值比较

### 典型结果范围

- **良好性能**: 余弦相似度 > 0.8，准确率 > 0.7
- **一般性能**: 余弦相似度 0.5-0.8，准确率 0.4-0.7
- **需要改进**: 余弦相似度 < 0.5，准确率 < 0.4

## 故障排除

### 常见问题

1. **导入错误 (Import Error)**
   ```bash
   # 设置Python路径
   export PYTHONPATH="/home/hlwang/mdt_policy:$PYTHONPATH"
   ```

2. **CUDA内存不足**
   ```bash
   # 使用CPU或减少批次大小
   python test_mdt_latent_validation.py --device cpu --batch_size 1
   ```

3. **模型加载失败**
   - 检查checkpoint文件是否完整
   - 确认checkpoint是PyTorch Lightning格式
   - 尝试使用不同的加载方式

4. **缺少预训练组件**
   - 检查模型是否包含必要的预训练模块
   - 确认checkpoint包含所有组件的权重

### 调试模式

```bash
# 详细日志输出
python test_mdt_latent_validation.py \
  --checkpoint /path/to/model.ckpt \
  --device cpu \
  --batch_size 1 \
  --num_batches 1 \
  --use_mock_data 2>&1 | tee debug.log
```

## 高级用法

### 自定义测试数据

如果需要使用特定的测试数据，可以修改脚本中的`create_simple_dataloader`函数，或者实现自己的数据加载逻辑。

### 添加新的评估指标

可以在`calculate_consistency_metrics`函数中添加新的评估指标，例如：
- BLEU分数 (对于序列数据)
- 结构相似性指标
- 自定义领域特定指标

### 批量测试

```bash
# 测试多个checkpoints
for ckpt in /path/to/checkpoints/*.ckpt; do
    echo "Testing $ckpt"
    python test_mdt_latent_validation.py \
      --checkpoint "$ckpt" \
      --output_dir "./results/$(basename $ckpt .ckpt)" \
      --use_mock_data
done
```

## 注意事项

1. **内存管理**: 大批次可能导致GPU内存不足，建议从小批次开始
2. **时间成本**: 完整测试可能需要较长时间，可以先用少量批次验证
3. **结果解释**: 相似度指标需要结合具体任务和数据特点来解释
4. **版本兼容**: 确保测试环境与训练环境的PyTorch版本兼容

## 联系支持

如果遇到问题，请：
1. 首先运行`python mdt_test_helper.py`进行环境诊断
2. 检查checkpoint文件和配置文件是否正确
3. 查看详细的错误日志
4. 尝试使用更简单的测试设置