# 科学实验设计优化方案

## 基于论文的关键发现

### 1. 架构要点（从论文Section 2）
- **Multi-scale features**: 从intermediate layers提取 {Fk}K k=1，不是同一层的不同pooling
- **VAE latent dimension**: dz (需要确定具体值)
- **Loss balance**: α 通过grid search或cross-validation选择
- **Feature normalization**: batch mean centering + L2 normalization

### 2. 训练设置（论文Section 3.4）
- Optimizer: AdamW
- Learning rate: η (需要确定)
- Batch size: B (需要确定)
- Weight decay: w (需要确定)
- Epochs: e (需要确定)
- Alpha: α (grid search)

## 优化策略

### A. 架构改进
1. **真正的multi-scale features**: 从ResNet的不同stage提取
2. **标准化分类头**: ResNet-50 Enhanced使用与VDLF-Net相同的分类头
3. **优化VAE架构**: 确保latent space有足够表达能力

### B. 超参数优化
1. **学习率调度**: CosineAnnealingLR或StepLR
2. **Alpha调优**: 网格搜索 [0.01, 0.05, 0.1, 0.2]
3. **Batch size**: 128 (CIFAR-100标准)
4. **Weight decay**: 1e-4 (标准值)
5. **Latent dimension**: 128 (平衡表达能力和计算)

### C. 训练策略
1. **学习率预热**: 前几个epoch线性warmup
2. **早停机制**: 防止过拟合
3. **模型保存**: 保存最佳验证性能的模型
4. **数据增强**: 按照论文描述的标准增强

### D. 公平对比
1. **ResNet-50 Standard**: 单层分类头（基线）
2. **ResNet-50 Enhanced**: 多层分类头（公平对比）
3. **VDLF-Net**: 完整架构
4. **参数量报告**: 透明化所有模型的参数量

## 具体实现要点

### Multi-scale Feature Extraction
```python
# 从ResNet的不同stage提取特征
# Stage 1: layer2 output (512 channels)
# Stage 2: layer3 output (1024 channels)  
# Stage 3: layer4 output (2048 channels)
# 然后统一到相同维度进行融合
```

### Alpha调优策略
- 初始值: 0.1
- 搜索范围: [0.01, 0.05, 0.1, 0.2]
- 评估指标: Validation F1-score
- 选择标准: 平衡task loss和VAE loss

### 学习率策略
- 初始LR: 0.001 (baselines), 0.0005 (VDLF-Net)
- 调度器: CosineAnnealingLR (T_max=epochs)
- Warmup: 5 epochs线性warmup
