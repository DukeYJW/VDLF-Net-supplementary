# Table 2 Few-Shot实验实现指南

## 文件说明

1. **few_shot_experiment_design.md**: 详细的实验设计方案
2. **experiment_framework.py**: VDLF-Net的基础实现框架
3. **IMPLEMENTATION_GUIDE.md**: 本文件，实现步骤指南

## 实现步骤

### 第一步：环境准备

```bash
# 创建虚拟环境
conda create -n fewshot python=3.8
conda activate fewshot

# 安装依赖
pip install torch torchvision
pip install numpy scipy
pip install scikit-learn
pip install tqdm tensorboard
```

### 第二步：数据集准备

#### 2.1 下载Mini-ImageNet

Mini-ImageNet数据集可以从以下来源获取：
- 官方链接（如果可用）
- Few-shot学习论文的GitHub仓库
- 常用下载链接：https://github.com/yaoyao-liu/mini-imagenet-tools

#### 2.2 数据预处理脚本

创建 `data_loader.py`:

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json

class MiniImageNetDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: 数据根目录
            split: 'train', 'val', 或 'test'
            transform: 数据增强变换
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 加载split文件（包含类别和图像路径）
        split_file = os.path.join(data_dir, f'{split}.json')
        with open(split_file, 'r') as f:
            self.data = json.load(f)
        
        # 构建图像路径和标签列表
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        for class_name, image_list in self.data.items():
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
            
            class_idx = self.class_to_idx[class_name]
            for img_name in image_list:
                img_path = os.path.join(data_dir, 'images', img_name)
                self.images.append(img_path)
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

#### 2.3 数据增强

```python
from torchvision import transforms

def get_train_transform():
    return transforms.Compose([
        transforms.Resize(92),  # 稍大于84，用于random crop
        transforms.RandomCrop(84, padding=8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
```

### 第三步：实现基线方法

#### 3.1 Prototypical Networks

创建 `baselines/prototypical_networks.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, x):
        features = self.backbone(x)
        # 全局平均池化
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        # L2归一化
        features = F.normalize(features, p=2, dim=1)
        return features
    
    def compute_prototypes(self, support_features, support_labels, n_way, k_shot):
        """计算类别原型"""
        support_features = support_features.view(n_way, k_shot, -1)
        prototypes = support_features.mean(dim=1)  # [N, feat_dim]
        return prototypes
    
    def predict(self, query_features, prototypes):
        """基于欧氏距离预测"""
        # 计算欧氏距离
        distances = torch.cdist(query_features.unsqueeze(0), 
                                prototypes.unsqueeze(0)).squeeze(0)
        # 转换为相似度（负距离）
        logits = -distances
        return logits
```

#### 3.2 Matching Networks

创建 `baselines/matching_networks.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingNetworks(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Full Context Embeddings (简化版，使用LSTM)
        self.lstm = nn.LSTM(backbone.output_dim, 128, batch_first=True)
    
    def forward(self, x):
        features = self.backbone(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        return features
    
    def attention_match(self, query_features, support_features):
        """注意力匹配"""
        # 计算余弦相似度
        similarity = F.cosine_similarity(
            query_features.unsqueeze(1),  # [Q, 1, feat_dim]
            support_features.unsqueeze(0),  # [1, S, feat_dim]
            dim=2
        )  # [Q, S]
        
        # Softmax归一化
        attention_weights = F.softmax(similarity, dim=1)
        
        # 加权求和
        matched_features = torch.bmm(
            attention_weights.unsqueeze(1),  # [Q, 1, S]
            support_features.unsqueeze(0).expand(
                query_features.size(0), -1, -1
            )  # [Q, S, feat_dim]
        ).squeeze(1)  # [Q, feat_dim]
        
        return matched_features
```

#### 3.3 MAML

MAML实现较复杂，建议参考：
- https://github.com/cbfinn/maml
- https://github.com/learnables/learn2learn

或者使用learn2learn库：

```python
import learn2learn as l2l
from learn2learn.algorithms import MAML

# 使用learn2learn库实现MAML
model = l2l.vision.models.ResNet12(output_size=64, hidden_size=1600)
maml = MAML(model, lr=0.01)
```

### 第四步：训练脚本

创建 `train.py`:

```python
import torch
from experiment_framework import (
    FewShotConfig, EpisodeSampler, VDLFNet, 
    train_vdlf_net, evaluate_model
)
from data_loader import MiniImageNetDataset, get_train_transform, get_test_transform

def main():
    # 配置
    config_1shot = FewShotConfig(k_shot=1)
    config_5shot = FewShotConfig(k_shot=5)
    
    # 加载数据集
    train_dataset = MiniImageNetDataset(
        'data/mini-imagenet', 
        split='train',
        transform=get_train_transform()
    )
    val_dataset = MiniImageNetDataset(
        'data/mini-imagenet',
        split='val',
        transform=get_test_transform()
    )
    test_dataset = MiniImageNetDataset(
        'data/mini-imagenet',
        split='test',
        transform=get_test_transform()
    )
    
    # 创建采样器
    train_sampler_1shot = EpisodeSampler(
        train_dataset, n_way=5, k_shot=1, q_query=15
    )
    train_sampler_5shot = EpisodeSampler(
        train_dataset, n_way=5, k_shot=5, q_query=15
    )
    test_sampler_1shot = EpisodeSampler(
        test_dataset, n_way=5, k_shot=1, q_query=15
    )
    test_sampler_5shot = EpisodeSampler(
        test_dataset, n_way=5, k_shot=5, q_query=15
    )
    
    # 训练1-shot模型
    print("Training 1-shot model...")
    model_1shot = VDLFNet(config_1shot)
    model_1shot = train_vdlf_net(
        model_1shot, train_sampler_1shot, 
        EpisodeSampler(val_dataset, 5, 1, 15), config_1shot
    )
    
    # 评估1-shot
    print("Evaluating 1-shot model...")
    acc_1shot, ci_1shot, _ = evaluate_model(
        model_1shot, test_sampler_1shot, config_1shot
    )
    print(f"1-shot Accuracy: {acc_1shot:.2f}% ± {ci_1shot:.2f}%")
    
    # 训练5-shot模型
    print("Training 5-shot model...")
    model_5shot = VDLFNet(config_5shot)
    model_5shot = train_vdlf_net(
        model_5shot, train_sampler_5shot,
        EpisodeSampler(val_dataset, 5, 5, 15), config_5shot
    )
    
    # 评估5-shot
    print("Evaluating 5-shot model...")
    acc_5shot, ci_5shot, _ = evaluate_model(
        model_5shot, test_sampler_5shot, config_5shot
    )
    print(f"5-shot Accuracy: {acc_5shot:.2f}% ± {ci_5shot:.2f}%")

if __name__ == "__main__":
    main()
```

### 第五步：基线方法评估脚本

创建 `evaluate_baselines.py`:

```python
import torch
from baselines.prototypical_networks import PrototypicalNetworks
from baselines.matching_networks import MatchingNetworks
from experiment_framework import FewShotConfig, EpisodeSampler, evaluate_model
from data_loader import MiniImageNetDataset, get_test_transform

def evaluate_baseline(model, sampler, config, model_name):
    """评估基线方法"""
    acc, ci, _ = evaluate_model(model, sampler, config)
    print(f"{model_name} - Accuracy: {acc:.2f}% ± {ci:.2f}%")
    return acc, ci

def main():
    # 加载测试集
    test_dataset = MiniImageNetDataset(
        'data/mini-imagenet',
        split='test',
        transform=get_test_transform()
    )
    
    config_1shot = FewShotConfig(k_shot=1)
    config_5shot = FewShotConfig(k_shot=5)
    
    test_sampler_1shot = EpisodeSampler(test_dataset, 5, 1, 15)
    test_sampler_5shot = EpisodeSampler(test_dataset, 5, 5, 15)
    
    # 创建backbone（ResNet-50）
    import torchvision.models as models
    backbone = models.resnet50(pretrained=True)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
    
    results = {}
    
    # Prototypical Networks
    print("Evaluating Prototypical Networks...")
    proto_net_1shot = PrototypicalNetworks(backbone)
    proto_net_5shot = PrototypicalNetworks(backbone)
    
    acc_1shot, ci_1shot = evaluate_baseline(
        proto_net_1shot, test_sampler_1shot, config_1shot, 
        "Prototypical Networks (1-shot)"
    )
    acc_5shot, ci_5shot = evaluate_baseline(
        proto_net_5shot, test_sampler_5shot, config_5shot,
        "Prototypical Networks (5-shot)"
    )
    
    results['Prototypical Networks'] = {
        '1-shot': (acc_1shot, ci_1shot),
        '5-shot': (acc_5shot, ci_5shot)
    }
    
    # Matching Networks
    print("Evaluating Matching Networks...")
    match_net_1shot = MatchingNetworks(backbone)
    match_net_5shot = MatchingNetworks(backbone)
    
    acc_1shot, ci_1shot = evaluate_baseline(
        match_net_1shot, test_sampler_1shot, config_1shot,
        "Matching Networks (1-shot)"
    )
    acc_5shot, ci_5shot = evaluate_baseline(
        match_net_5shot, test_sampler_5shot, config_5shot,
        "Matching Networks (5-shot)"
    )
    
    results['Matching Networks'] = {
        '1-shot': (acc_1shot, ci_1shot),
        '5-shot': (acc_5shot, ci_5shot)
    }
    
    # MAML（需要单独实现或使用库）
    # ...
    
    return results

if __name__ == "__main__":
    results = main()
```

### 第六步：结果汇总脚本

创建 `generate_table.py`:

```python
import json

def generate_table(results_dict):
    """生成LaTeX表格"""
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Few-shot classification accuracy (\\%; mean $\\pm$ 95\\% confidence interval) on Mini-ImageNet.}")
    print("\\label{tab:fewshot_results}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{5-way Accuracy} \\\\")
    print(" & 1-shot (\\%) \\hspace{2em} 5-shot (\\%) \\\\")
    print("\\midrule")
    
    for model_name, metrics in results_dict.items():
        acc_1shot, ci_1shot = metrics['1-shot']
        acc_5shot, ci_5shot = metrics['5-shot']
        
        if model_name == 'VDLF-Net':
            print(f"\\textbf{{{model_name}}} & \\textbf{{{acc_1shot:.2f} $\\pm$ {ci_1shot:.2f}}} \\hspace{{2em}} \\textbf{{{acc_5shot:.2f} $\\pm$ {ci_5shot:.2f}}} \\\\")
        else:
            print(f"{model_name} & {acc_1shot:.2f} $\\pm$ {ci_1shot:.2f} \\hspace{{2em}} {acc_5shot:.2f} $\\pm$ {ci_5shot:.2f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    # 从JSON文件加载结果
    with open('results.json', 'r') as f:
        results = json.load(f)
    
    generate_table(results)
```

## 实验运行流程

1. **准备数据**：下载并预处理Mini-ImageNet数据集
2. **实现基线**：实现Prototypical Networks、Matching Networks和MAML
3. **训练VDLF-Net**：运行`train.py`训练1-shot和5-shot模型
4. **评估基线**：运行`evaluate_baselines.py`评估所有基线方法
5. **汇总结果**：运行`generate_table.py`生成LaTeX表格

## 注意事项

1. **计算资源**：Few-shot训练需要大量GPU时间，建议使用多GPU或云服务器
2. **超参数调优**：在验证集上调整超参数（α, τ, T等）
3. **多次运行**：建议每个配置运行3-5次，取平均结果
4. **结果保存**：保存所有实验结果和模型checkpoints

## 预期时间

- 数据集准备：1-2小时
- 基线方法实现：2-3天
- VDLF-Net训练（1-shot）：1-2天
- VDLF-Net训练（5-shot）：1-2天
- 基线方法评估：1-2天
- **总计**：约1-2周（取决于计算资源）

## 调试建议

1. 先在CIFAR-FS（更小的数据集）上验证实现
2. 使用较少的episodes进行快速迭代
3. 可视化特征空间（t-SNE）检查学习效果
4. 检查损失函数是否正常下降
5. 验证episode采样是否正确
