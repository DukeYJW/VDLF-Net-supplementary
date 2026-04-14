# Table 2 Few-Shot Learning 实验实现

本项目提供了实现论文中Table 2的few-shot learning实验的完整方案和代码框架。

## 📁 文件结构

```
experiment_2/
├── README.md                          # 本文件
├── few_shot_experiment_design.md      # 详细的实验设计方案
├── IMPLEMENTATION_GUIDE.md            # 逐步实现指南
├── experiment_framework.py            # VDLF-Net基础实现框架
└── Variational_Deep_Learning_...pdf  # 参考论文PDF
```

## 🎯 实验目标

在Mini-ImageNet数据集上评估VDLF-Net的few-shot学习能力，并与以下基线方法对比：

- **MAML** (Model-Agnostic Meta-Learning)
- **Prototypical Networks**
- **Matching Networks**

**评估指标**：5-way 1-shot 和 5-way 5-shot 分类准确率（mean ± 95% confidence interval）

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n fewshot python=3.8
conda activate fewshot

# 安装依赖
pip install torch torchvision numpy scipy scikit-learn tqdm tensorboard
```

### 2. 数据集准备

下载Mini-ImageNet数据集并按照标准划分准备：
- 训练集：64个类别
- 验证集：16个类别  
- 测试集：20个类别

### 3. 运行实验

参考 `IMPLEMENTATION_GUIDE.md` 中的详细步骤：
1. 实现数据集加载器
2. 实现基线方法
3. 训练VDLF-Net模型
4. 评估所有方法
5. 生成结果表格

## 📊 预期输出

实验完成后，将生成Table 2格式的结果：

| Model | 5-way Accuracy |
|-------|----------------|
|       | 1-shot (%)    | 5-shot (%)    |
|-------|----------------|---------------|
| MAML | XX.XX ± X.XX | XX.XX ± X.XX |
| Prototypical Networks | XX.XX ± X.XX | XX.XX ± X.XX |
| Matching Networks | XX.XX ± X.XX | XX.XX ± X.XX |
| **VDLF-Net** | **XX.XX ± X.XX** | **XX.XX ± X.XX** |

## 📖 文档说明

### `few_shot_experiment_design.md`
包含完整的实验设计方案，包括：
- 数据集准备细节
- 基线方法实现要点
- VDLF-Net架构配置
- 训练和评估流程
- 统计报告方法

### `IMPLEMENTATION_GUIDE.md`
提供逐步实现指南，包括：
- 代码实现示例
- 数据集加载脚本
- 基线方法实现
- 训练和评估脚本
- 结果汇总方法

### `experiment_framework.py`
提供了VDLF-Net的基础实现框架，包括：
- 配置类（FewShotConfig）
- Episode采样器（EpisodeSampler）
- VDLF-Net模型架构
- 损失函数计算
- 评估函数
- 训练循环

## 🔧 核心组件

### VDLF-Net架构

- **Backbone**: ResNet-50（截断到layer4）
- **多尺度特征**: 2×2和1×1 adaptive average pooling
- **VAE**: 编码器-解码器结构，潜在维度128
- **融合模块**: Gating网络生成自适应权重
- **Few-shot推理**: 变分采样支持集，确定性查询集

### 关键超参数

- `latent_dim`: 128
- `num_samples` (T): 10（支持集采样数）
- `temperature` (τ): 10.0
- `alpha`: 0.01（变分正则化系数；与 `table2_few_shot_experiment.ipynb` 及论文 Table 2 一致）

## ⚠️ 注意事项

1. **计算资源**: Few-shot训练需要大量GPU时间，预计每个配置需要1-2天
2. **可复现性**: 固定随机种子，记录所有超参数
3. **公平对比**: 确保所有方法使用相同的特征提取器和数据划分
4. **多次运行**: 建议每个配置运行3-5次取平均

## 📝 实验检查清单

- [ ] 下载并预处理Mini-ImageNet数据集
- [ ] 实现数据集加载器
- [ ] 实现Prototypical Networks
- [ ] 实现Matching Networks
- [ ] 实现MAML（或使用开源库）
- [ ] 实现VDLF-Net完整架构
- [ ] 训练1-shot模型
- [ ] 训练5-shot模型
- [ ] 评估所有基线方法
- [ ] 生成结果表格
- [ ] 多次运行验证稳定性

## 🔗 参考资源

### 数据集
- Mini-ImageNet: https://github.com/yaoyao-liu/mini-imagenet-tools

### 基线方法实现
- MAML: https://github.com/cbfinn/maml
- Prototypical Networks: https://github.com/jakesnell/prototypical-networks
- Matching Networks: https://github.com/gitabcworld/MatchingNetworks
- Learn2Learn框架: https://github.com/learnables/learn2learn

## 📧 问题反馈

如有问题或建议，请参考论文或联系作者。

## 📄 许可证

本项目代码仅供研究使用。
