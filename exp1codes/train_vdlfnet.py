#!/usr/bin/env python3
"""
VDLF-Net 独立训练脚本
基于 table1_experiment.ipynb 提取，仅包含 VDLF-Net 训练部分。
支持 VDLF_LR、VDLF_WEIGHT_DECAY、VDLF_ALPHA、VDLF_LATENT_DIM 四个命令行参数。
每次运行后将参数和主要结果追加到 CSV 文件。
"""

import argparse
import csv
import os
import time
import warnings
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# 模型定义
# ============================================================================

class VAEEncoder(nn.Module):
    """VAE Encoder for encoding fused features into latent space"""
    def __init__(self, input_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder for reconstructing fused features from latent code"""
    def __init__(self, latent_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        recon = self.fc3(z)
        return recon


class GatingNetwork(nn.Module):
    """Gating network that generates fusion weights from latent code"""
    def __init__(self, latent_dim, num_scales):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, num_scales)
        self.relu = nn.ReLU()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        weights = torch.softmax(self.fc2(z), dim=1)
        return weights


class VDLFNet(nn.Module):
    """
    Variational-Deep Learning Fusion Network for CIFAR-100 classification
    """
    def __init__(self, backbone='resnet50', latent_dim=128, num_classes=100, alpha=0.1):
        super(VDLFNet, self).__init__()
        self.alpha = alpha

        resnet = resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.pool1 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.num_scales = 2
        self.initial_fusion_dim = 2048

        self.feat1_proj = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.vae_encoder = VAEEncoder(self.initial_fusion_dim, latent_dim)
        self.vae_decoder = VAEDecoder(latent_dim, self.initial_fusion_dim)
        self.gating_net = GatingNetwork(latent_dim, self.num_scales)
        self.epsilon = 1e-8

        self.classifier = nn.Sequential(
            nn.Linear(self.initial_fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_loss_components=False):
        features = self.backbone(x)
        feat1 = self.pool1(features).flatten(1)
        feat2 = self.pool2(features).flatten(1)
        feat1_proj = self.feat1_proj(feat1)
        multi_scale_features = [feat1_proj, feat2]
        F_fused_0 = torch.stack(multi_scale_features, dim=0).mean(dim=0)
        mu, logvar = self.vae_encoder(F_fused_0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        weights = self.gating_net(z)
        stacked_features = torch.stack(multi_scale_features, dim=1)
        weights_expanded = weights.unsqueeze(-1)
        F_fused = (stacked_features * weights_expanded).sum(dim=1)
        F_fused_mean = F_fused.mean(dim=0, keepdim=True)
        F_fused_centered = F_fused - F_fused_mean
        F_norm = F_fused_centered / (F_fused_centered.norm(dim=1, keepdim=True) + self.epsilon)
        logits = self.classifier(F_norm)

        if return_loss_components:
            F_fused_0_recon = self.vae_decoder(z)
            recon_loss = nn.functional.mse_loss(F_fused_0_recon, F_fused_0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            return logits, recon_loss, kl_loss

        return logits


# ============================================================================
# 工具函数
# ============================================================================

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, test_loader, device):
    """Evaluate model and return Accuracy, Macro Precision, Macro Recall, Macro F1"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    return {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }


def _get_kl_anneal_weight(epoch, kl_anneal_epochs):
    """KL退火权重：前kl_anneal_epochs个epoch线性从0增至1，之后为1。kl_anneal_epochs=0表示不退火。"""
    if kl_anneal_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / kl_anneal_epochs)


def train_vdlfnet(model, train_loader, test_loader, epochs, lr, weight_decay, device, alpha=0.1, kl_anneal_epochs=0):
    """Train VDLF-Net with unified objective: L_total = L_task + alpha * (L_Recon + kl_weight * L_KL)
    kl_anneal_epochs: KL退火周期，0表示不退火；>0时前N个epoch线性增加KL权重
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )
    best_acc = 0.0
    best_model_state = None
    start_time = time.time()

    for epoch in range(epochs):
        kl_weight = _get_kl_anneal_weight(epoch, kl_anneal_epochs)
        model.train()
        running_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f'VDLF-Net Epoch {epoch+1}/{epochs}', leave=True, dynamic_ncols=True)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits, recon_loss, kl_loss = model(images, return_loss_components=True)
            task_loss = nn.functional.cross_entropy(logits, labels)
            total_loss = task_loss + alpha * (recon_loss + kl_weight * kl_loss)
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            num_batches += 1
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'task': f'{task_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{(kl_weight * kl_loss.item()):.4f}',
                'lr': f'{current_lr:.5f}'
            })
        scheduler.step()
        metrics = evaluate_model(model, test_loader, device)
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{epochs} - Test Acc: {metrics["accuracy"]:.2f}%, F1: {metrics["f1"]:.2f}%')
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    total_time = time.time() - start_time
    print(f"\nVDLF-Net Training Completed! Total Time: {timedelta(seconds=int(total_time))}")
    return evaluate_model(model, test_loader, device), total_time


def append_results_to_csv(csv_path, vdlf_lr, vdlf_weight_decay, vdlf_alpha, vdlf_latent_dim, kl_anneal_epochs, metrics, train_time):
    """将参数和结果追加到CSV文件"""
    file_exists = os.path.exists(csv_path)
    row = {
        'VDLF_LR': vdlf_lr,
        'VDLF_WEIGHT_DECAY': vdlf_weight_decay,
        'VDLF_ALPHA': vdlf_alpha,
        'VDLF_LATENT_DIM': vdlf_latent_dim,
        'KL_ANNEAL_EPOCHS': kl_anneal_epochs,
        'Accuracy': f"{metrics['accuracy']:.4f}",
        'Precision': f"{metrics['precision']:.4f}",
        'Recall': f"{metrics['recall']:.4f}",
        'F1': f"{metrics['f1']:.4f}",
        'TrainTime_sec': f"{train_time:.2f}"
    }
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"结果已追加到 {csv_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='VDLF-Net 训练脚本')
    parser.add_argument('--VDLF_LR', type=float, default=0.0015, help='学习率')
    parser.add_argument('--VDLF_WEIGHT_DECAY', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--VDLF_ALPHA', type=float, default=0.02, help='VAE损失权重，控制分类与重建损失的平衡')
    parser.add_argument('--VDLF_LATENT_DIM', type=int, default=128, help='VAE潜在空间维度')
    parser.add_argument('--KL_ANNEAL_EPOCHS', type=int, default=0, help='KL退火周期，0=不退火；>0则前N个epoch线性增加KL权重至1')
    parser.add_argument('--EPOCHS', type=int, default=100, help='训练轮数')
    parser.add_argument('--BATCH_SIZE', type=int, default=128, help='批次大小')
    parser.add_argument('--RANDOM_SEED', type=int, default=42, help='随机种子')
    parser.add_argument('--DATA_ROOT', type=str, default='data', help='CIFAR-100 数据根目录')
    parser.add_argument('--OUTPUT_CSV', type=str, default='vdlf_results.csv', help='结果输出CSV路径')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(args.RANDOM_SEED)
    np.random.seed(args.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_dataset = torchvision.datasets.CIFAR100(
        root=args.DATA_ROOT, train=True, download=False, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=args.DATA_ROOT, train=False, download=False, transform=test_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=8)

    print(f"\n{'='*60}")
    print("VDLF-Net 超参数:")
    print(f"  LR: {args.VDLF_LR}, Weight Decay: {args.VDLF_WEIGHT_DECAY}")
    print(f"  Alpha: {args.VDLF_ALPHA}, Latent Dim: {args.VDLF_LATENT_DIM}")
    print(f"  KL Anneal Epochs: {args.KL_ANNEAL_EPOCHS} (0=不退火)")
    print(f"{'='*60}\n")

    model = VDLFNet(
        backbone='resnet50',
        latent_dim=args.VDLF_LATENT_DIM,
        num_classes=100,
        alpha=args.VDLF_ALPHA
    )
    print(f"参数量: {count_parameters(model):,}")

    metrics, train_time = train_vdlfnet(
        model, train_loader, test_loader,
        args.EPOCHS, args.VDLF_LR, args.VDLF_WEIGHT_DECAY, device,
        alpha=args.VDLF_ALPHA, kl_anneal_epochs=args.KL_ANNEAL_EPOCHS
    )

    print(f"\n最终结果: Acc={metrics['accuracy']:.2f}%, Precision={metrics['precision']:.2f}%, "
          f"Recall={metrics['recall']:.2f}%, F1={metrics['f1']:.2f}%")

    append_results_to_csv(
        args.OUTPUT_CSV,
        args.VDLF_LR, args.VDLF_WEIGHT_DECAY, args.VDLF_ALPHA, args.VDLF_LATENT_DIM,
        args.KL_ANNEAL_EPOCHS, metrics, train_time
    )


if __name__ == '__main__':
    main()
