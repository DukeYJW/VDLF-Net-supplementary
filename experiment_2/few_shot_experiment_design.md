# Table 2 Few-Shot Learning 实验设计方案

## 1. 实验目标

在Mini-ImageNet数据集上评估VDLF-Net的few-shot学习能力，并与以下基线方法对比：
- MAML (Model-Agnostic Meta-Learning)
- Prototypical Networks
- Matching Networks

评估指标：5-way 1-shot 和 5-way 5-shot 分类准确率（mean ± 95% confidence interval）

## 2. 数据集准备

### 2.1 Mini-ImageNet数据集
- **来源**：从ImageNet中选取的100个类别，每个类别600张图像
- **数据划分**：
  - 训练集：64个类别（用于episodic meta-training）
  - 验证集：16个类别（用于超参数调优）
  - 测试集：20个类别（用于最终评估，训练时不可见）

### 2.2 数据预处理
- **图像尺寸**：84×84（标准few-shot学习设置）
- **归一化**：ImageNet的channel-wise均值和标准差
- **数据增强**（训练时）：
  - Random crop (padding=8)
  - Random horizontal flip (p=0.5)
  - Random rotation (角度范围：-10°到+10°)
- **评估时**：仅中心裁剪和归一化

### 2.3 Episode采样
- **N-way K-shot设置**：
  - N=5（5个类别）
  - K ∈ {1, 5}（每个类别的支持样本数）
  - Q=15（每个类别的查询样本数，标准设置）
- **测试episodes数量**：600个随机采样的episodes
- **Episode采样策略**：
  - 从测试集的20个类别中随机选择N个类别
  - 每个类别随机选择K个支持样本和Q个查询样本
  - 确保每个episode中的类别和样本不重复

## 3. 基线方法实现

### 3.1 MAML (Model-Agnostic Meta-Learning)
**核心思想**：学习一个可以快速适应新任务的初始化参数

**实现要点**：
- 使用4层CNN作为特征提取器（与原始MAML论文一致）
- 内循环（inner loop）：每个episode内进行K步梯度更新
- 外循环（outer loop）：基于查询集损失更新元参数
- 学习率：内循环α=0.01，外循环β=0.001
- 内循环更新步数：5步（5-shot）或1步（1-shot）

**训练设置**：
- 优化器：Adam
- Batch size：4个episodes/batch
- 训练episodes：60,000个
- 评估：600个测试episodes

### 3.2 Prototypical Networks
**核心思想**：在embedding空间中计算类别原型，通过最近邻分类

**实现要点**：
- 特征提取器：ResNet-50（与VDLF-Net保持一致，便于公平对比）
- 原型计算：每个类别的支持样本embedding的平均值
- 距离度量：欧氏距离（原始论文）或余弦相似度（可选）
- 分类：基于到原型的距离进行softmax分类

**训练设置**：
- 优化器：Adam
- 学习率：0.001
- Batch size：16个episodes/batch
- 训练episodes：40,000个
- 评估：600个测试episodes

### 3.3 Matching Networks
**核心思想**：使用注意力机制匹配查询样本和支持样本

**实现要点**：
- 特征提取器：ResNet-50（与VDLF-Net保持一致）
- 注意力机制：使用余弦相似度计算注意力权重
- 全上下文嵌入（Full Context Embeddings）：支持集和查询集都使用LSTM进行上下文编码
- 分类：基于加权的支持集embedding进行预测

**训练设置**：
- 优化器：Adam
- 学习率：0.001
- Batch size：16个episodes/batch
- 训练episodes：40,000个
- 评估：600个测试episodes

## 4. VDLF-Net实现细节

### 4.1 架构配置
- **Backbone**：ResNet-50（在global average pooling之前截断）
- **多尺度特征提取**：从layer4提取K=2个尺度的特征
  - 尺度1：2×2 adaptive average pooling
  - 尺度2：1×1 adaptive average pooling
- **VAE配置**：
  - 编码器：2层MLP，输出μ和log(σ²)
  - 解码器：2层MLP，重构初始融合特征
  - 潜在维度：128
- **融合模块**：
  - Gating网络Θ：2层MLP，输入为潜在变量z，输出K维权重
  - 归一化：L2归一化，投影到单位超球面

### 4.2 Few-Shot推理流程
1. **支持集处理**：
   - 对每个支持样本，采样T=10个潜在变量z^(t)
   - 每个z^(t)生成对应的融合权重w^(t)
   - 计算T个归一化特征F_norm^(t)
   - 类别原型：P_c = (1/KT) Σ_i Σ_t F_norm^(t)(x_{c,i})

2. **查询集处理**：
   - 使用确定性embedding（不采样）
   - 计算F_norm(x_q)

3. **预测**：
   - 计算查询样本与每个类别原型的余弦相似度
   - 使用温度缩放τ=10的softmax进行分类

### 4.3 训练设置
- **优化器**：AdamW
- **学习率**：0.001
- **Weight decay**：0.0001
- **Batch size**：16个episodes/batch
- **训练episodes**：40,000个
- **损失函数平衡系数α**：0.01（与 Table 2 notebook 一致）
- **学习率调度**：CosineAnnealingLR，η_min = 0.1 × initial LR

### 4.4 关键超参数
- **FAAM采样数T**：10（支持集）
- **温度参数τ**：10（余弦softmax）
- **变分正则化系数α**：0.01

## 5. 训练流程

### 5.1 预训练阶段（可选）
- 在训练集的64个类别上进行标准监督学习预训练
- 使用交叉熵损失 + 变分正则化项
- 预训练epochs：50
- 这有助于初始化特征提取器

### 5.2 Episodic Meta-Training
1. **Episode采样**：
   - 从训练集的64个类别中随机采样N=5个类别
   - 每个类别采样K个支持样本和Q=15个查询样本

2. **前向传播**：
   - 支持集：使用T次采样计算原型
   - 查询集：计算确定性embedding
   - 计算episodic交叉熵损失L_CE^epi

3. **变分损失计算**：
   - 对所有样本（支持+查询）计算重构损失L_Recon
   - 计算KL散度损失L_KL
   - 总损失：L_total = L_CE^epi + α(L_Recon + L_KL)

4. **反向传播和优化**：
   - 端到端训练所有模块
   - 使用AdamW优化器更新参数

### 5.3 验证和早停
- 每1000个episodes在验证集上评估
- 使用验证集上的准确率进行早停
- 保存最佳模型用于测试

## 6. 评估流程

### 6.1 测试Episode采样
- 从测试集的20个类别中采样600个episodes
- 每个episode：N=5个类别，K个支持样本，Q=15个查询样本
- 确保episodes之间的随机性和独立性

### 6.2 推理过程
1. **支持集处理**：
   - 对每个支持样本采样T=10个潜在变量
   - 计算类别原型（平均所有采样）

2. **查询集预测**：
   - 对每个查询样本计算确定性embedding
   - 计算与所有原型的余弦相似度
   - 预测类别：argmax_c sim(F_norm(x_q), P_c)

3. **准确率计算**：
   - 每个episode的准确率：正确预测数 / 查询样本总数
   - 600个episodes的平均准确率

### 6.3 统计报告
- **均值**：600个episodes的平均准确率
- **95%置信区间**：
  ```
  CI = mean ± 1.96 * (std / sqrt(600))
  ```
  其中std是600个episode准确率的标准差

## 7. 实验实施检查清单

### 7.1 数据准备
- [ ] 下载Mini-ImageNet数据集
- [ ] 按照标准划分准备train/val/test集
- [ ] 实现episode采样器
- [ ] 实现数据增强pipeline

### 7.2 基线方法实现
- [ ] 实现MAML（或使用开源实现）
- [ ] 实现Prototypical Networks
- [ ] 实现Matching Networks
- [ ] 确保所有基线使用相同的特征提取器（ResNet-50）

### 7.3 VDLF-Net实现
- [ ] 实现ResNet-50 backbone和多尺度特征提取
- [ ] 实现VAE编码器和解码器
- [ ] 实现FAAM模块（gating网络和融合）
- [ ] 实现prototypical head
- [ ] 实现episodic训练循环

### 7.4 训练和评估
- [ ] 实现episodic训练循环
- [ ] 实现测试评估循环（600个episodes）
- [ ] 实现统计报告（均值和置信区间）
- [ ] 设置日志和checkpoint保存

### 7.5 结果验证
- [ ] 确保所有方法在相同的数据划分上评估
- [ ] 确保使用相同的随机种子（用于可复现性）
- [ ] 多次运行取平均（至少3次独立运行）
- [ ] 记录所有超参数设置

## 8. 预期结果格式

Table 2的最终格式应为：

| Model | 5-way Accuracy |
|-------|----------------|
|       | 1-shot (%)    | 5-shot (%)    |
|-------|----------------|---------------|
| MAML | XX.XX ± X.XX | XX.XX ± X.XX |
| Prototypical Networks | XX.XX ± X.XX | XX.XX ± X.XX |
| Matching Networks | XX.XX ± X.XX | XX.XX ± X.XX |
| **VDLF-Net** | **XX.XX ± X.XX** | **XX.XX ± X.XX** |

## 9. 注意事项

1. **公平对比**：确保所有基线方法使用相同的特征提取器（ResNet-50）和相同的训练/测试数据划分

2. **可复现性**：
   - 固定随机种子
   - 记录所有超参数
   - 保存模型checkpoints

3. **计算资源**：
   - Few-shot训练需要大量episodes，预计训练时间较长
   - 建议使用GPU加速
   - 考虑使用分布式训练（如果资源允许）

4. **调试建议**：
   - 先在较小的数据集（如CIFAR-FS）上验证实现
   - 使用较少的episodes进行快速迭代
   - 可视化特征空间和原型分布

5. **结果分析**：
   - 分析1-shot和5-shot的性能差异
   - 分析不同方法的失败案例
   - 可视化t-SNE特征分布（可选）

## 10. 参考实现

建议参考以下开源实现：
- **MAML**: https://github.com/cbfinn/maml
- **Prototypical Networks**: https://github.com/jakesnell/prototypical-networks
- **Matching Networks**: https://github.com/gitabcworld/MatchingNetworks
- **Few-shot学习框架**: https://github.com/learnables/learn2learn

注意：需要根据论文的具体要求修改这些实现，确保与VDLF-Net的实验设置一致。
