#!/usr/bin/env python3
"""
Table 1 对比实验脚本
基于 table1_experiment.ipynb，对比不同 EPOCHS 和 BATCH_SIZE 下三种方法（VGG-16、ResNet-50 Enhanced、VDLF-Net）的表现。
支持 --MAX_EPOCHS 和 --RECORD_INTERVAL：训练至最大 epoch，每隔 N 个 epoch 记录一次结果，避免重复训练浪费计算。
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
from torchvision.models import vgg16, resnet50
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# 超参数常量（与 notebook 一致）
# ============================================================================
RANDOM_SEED = 42
NUM_WORKERS = 2
VGG_LR = 0.001
VGG_WEIGHT_DECAY = 0.0001
VGG_SCHEDULER_TYPE = 'cosine'
RESNET_LR = 0.002
RESNET_WEIGHT_DECAY = 0.00015
VDLF_LR = 0.002
VDLF_WEIGHT_DECAY = 0.00015
VDLF_ALPHA = 0.01
VDLF_LATENT_DIM = 128
VDLF_KL_ANNEAL_EPOCHS = 0

# ============================================================================
# VAE / VDLF 模型定义
# ============================================================================
class VAEEncoder(nn.Module):
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
        return self.fc_mu(x), self.fc_logvar(x)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        return self.fc3(z)


class GatingNetwork(nn.Module):
    def __init__(self, latent_dim, num_scales):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, num_scales)
        self.relu = nn.ReLU()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        return torch.softmax(self.fc2(z), dim=1)


class VDLFNet(nn.Module):
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
            nn.Linear(8192, 2048), nn.BatchNorm1d(2048), nn.ReLU()
        )
        self.vae_encoder = VAEEncoder(self.initial_fusion_dim, latent_dim)
        self.vae_decoder = VAEDecoder(latent_dim, self.initial_fusion_dim)
        self.gating_net = GatingNetwork(latent_dim, self.num_scales)
        self.epsilon = 1e-8
        self.classifier = nn.Sequential(
            nn.Linear(self.initial_fusion_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_loss_components=False):
        features = self.backbone(x)
        feat1 = self.feat1_proj(self.pool1(features).flatten(1))
        feat2 = self.pool2(features).flatten(1)
        multi_scale_features = [feat1, feat2]
        F_fused_0 = torch.stack(multi_scale_features, dim=0).mean(dim=0)
        mu, logvar = self.vae_encoder(F_fused_0)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        weights = self.gating_net(z)
        stacked = torch.stack(multi_scale_features, dim=1)
        F_fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        F_norm = (F_fused - F_fused.mean(0, keepdim=True)) / (F_fused.norm(dim=1, keepdim=True) + self.epsilon)
        logits = self.classifier(F_norm)
        if return_loss_components:
            recon_loss = nn.functional.mse_loss(self.vae_decoder(z), F_fused_0)
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
            return logits, recon_loss, kl_loss
        return logits


# ============================================================================
# 工具函数
# ============================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, test_loader, device, model_name=""):
    model.eval()
    if model_name == 'VGG-16':
        model.features.train()
        model.classifier.eval()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return {'accuracy': accuracy, 'precision': precision * 100, 'recall': recall * 100, 'f1': f1 * 100}


def build_vgg16_cifar100(device, train_loader):
    """VGG-16 适配 CIFAR-100：1x1 特征需替换 avgpool 并重建 classifier"""
    model = vgg16(pretrained=False).to(device)
    with torch.no_grad():
        sample, _ = next(iter(train_loader))
        sample = sample.to(device)
        model.features.eval()
        feat = model.features(sample)
        if feat.shape[2] == 1 and feat.shape[3] == 1:
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
            feat_new = model.avgpool(model.features(sample))
            features_size = feat_new.flatten(1).size(1)
        else:
            features_size = model.avgpool(feat).flatten(1).size(1)
        layers = list(model.classifier.children())
        layers[0] = nn.Linear(features_size, 4096).to(device)
        nn.init.kaiming_normal_(layers[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(layers[0].bias, 0)
        layers[6] = nn.Linear(4096, 100).to(device)
        nn.init.kaiming_normal_(layers[6].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(layers[6].bias, 0)
        model.classifier = nn.Sequential(*layers)
    return model


def train_model(model, train_loader, test_loader, epochs, lr, weight_decay, device, model_name, scheduler_type='cosine', record_epochs=None):
    """训练模型。若 record_epochs 非空，在指定 epoch 记录当前模型指标并返回 checkpoint_metrics"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if model_name == 'VGG-16':
        for p in model.features.parameters():
            p.requires_grad = True
        optimizer = optim.AdamW([
            {'params': model.features.parameters(), 'lr': lr * 0.01, 'weight_decay': weight_decay},
            {'params': model.classifier.parameters(), 'lr': lr, 'weight_decay': weight_decay}
        ])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    best_acc, best_state = 0.0, None
    checkpoint_metrics = {}
    for epoch in range(epochs):
        model.train()
        if model_name == 'VGG-16':
            model.features.train()
            model.classifier.train()
        for images, labels in tqdm(train_loader, desc=f'{model_name} {epoch+1}/{epochs}', leave=True, dynamic_ncols=True):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            if model_name == 'VGG-16':
                torch.nn.utils.clip_grad_norm_(model.features.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        metrics = evaluate_model(model, test_loader, device, model_name)
        if record_epochs and (epoch + 1) in record_epochs:
            checkpoint_metrics[epoch + 1] = metrics.copy()
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1}/{epochs} Acc: {metrics["accuracy"]:.2f}%, F1: {metrics["f1"]:.2f}%')
    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = evaluate_model(model, test_loader, device, model_name)
    if record_epochs:
        return final_metrics, checkpoint_metrics
    return final_metrics, {}


def _get_kl_anneal_weight(epoch, kl_anneal_epochs):
    if kl_anneal_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / kl_anneal_epochs)


def train_vdlfnet(model, train_loader, test_loader, epochs, lr, weight_decay, device, alpha, kl_anneal_epochs=0, record_epochs=None):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    best_acc, best_state = 0.0, None
    checkpoint_metrics = {}
    for epoch in range(epochs):
        kl_weight = _get_kl_anneal_weight(epoch, kl_anneal_epochs)
        model.train()
        for images, labels in tqdm(train_loader, desc=f'VDLF-Net {epoch+1}/{epochs}', leave=True, dynamic_ncols=True):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, recon, kl = model(images, return_loss_components=True)
            task = nn.functional.cross_entropy(logits, labels)
            loss = task + alpha * (recon + kl_weight * kl)
            loss.backward()
            optimizer.step()
        scheduler.step()
        metrics = evaluate_model(model, test_loader, device, 'VDLF-Net')
        if record_epochs and (epoch + 1) in record_epochs:
            checkpoint_metrics[epoch + 1] = metrics.copy()
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1}/{epochs} Acc: {metrics["accuracy"]:.2f}%, F1: {metrics["f1"]:.2f}%')
    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = evaluate_model(model, test_loader, device, 'VDLF-Net')
    if record_epochs:
        return final_metrics, checkpoint_metrics
    return final_metrics, {}


def append_results_csv(csv_path, epochs, batch_size, vgg_m, resnet_m, vdlf_m):
    mean_diff = (vdlf_m['accuracy'] - resnet_m['accuracy'] + vdlf_m['precision'] - resnet_m['precision'] +
                 vdlf_m['recall'] - resnet_m['recall'] + vdlf_m['f1'] - resnet_m['f1']) / 4.0
    row = {
        'ResNet_VDLF_Mean_Diff': f"{mean_diff:.4f}",
        'EPOCHS': epochs, 'BATCH_SIZE': batch_size,
        'VGG16_Accuracy': f"{vgg_m['accuracy']:.4f}", 'VGG16_Precision': f"{vgg_m['precision']:.4f}",
        'VGG16_Recall': f"{vgg_m['recall']:.4f}", 'VGG16_F1': f"{vgg_m['f1']:.4f}",
        'ResNet_Accuracy': f"{resnet_m['accuracy']:.4f}", 'ResNet_Precision': f"{resnet_m['precision']:.4f}",
        'ResNet_Recall': f"{resnet_m['recall']:.4f}", 'ResNet_F1': f"{resnet_m['f1']:.4f}",
        'VDLF_Accuracy': f"{vdlf_m['accuracy']:.4f}", 'VDLF_Precision': f"{vdlf_m['precision']:.4f}",
        'VDLF_Recall': f"{vdlf_m['recall']:.4f}", 'VDLF_F1': f"{vdlf_m['f1']:.4f}",
    }
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print(f"结果已追加到 {csv_path}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Table 1 三种方法对比实验')
    parser.add_argument('--MAX_EPOCHS', type=int, default=150, help='最大训练轮数，训练一次，每隔 RECORD_INTERVAL 记录')
    parser.add_argument('--RECORD_INTERVAL', type=int, default=20, help='每隔多少 epoch 记录一次结果')
    parser.add_argument('--BATCH_SIZE', type=int, default=128, help='批次大小')
    parser.add_argument('--DATA_ROOT', type=str, default='data', help='CIFAR-100 数据根目录')
    parser.add_argument('--OUTPUT_CSV', type=str, default='table1_compare_results.csv', help='结果 CSV 路径')
    parser.add_argument('--RANDOM_SEED', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    # 记录点：20, 40, 60, ... 以及最后的 MAX_EPOCHS
    record_epochs = sorted(set(list(range(args.RECORD_INTERVAL, args.MAX_EPOCHS + 1, args.RECORD_INTERVAL)) + [args.MAX_EPOCHS]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    torch.manual_seed(args.RANDOM_SEED)
    np.random.seed(args.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(), transforms.Normalize(mean, std)
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = torchvision.datasets.CIFAR100(root=args.DATA_ROOT, train=True, download=False, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR100(root=args.DATA_ROOT, train=False, download=False, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"\n{'='*60}")
    print(f"MAX_EPOCHS={args.MAX_EPOCHS}, RECORD_INTERVAL={args.RECORD_INTERVAL}, BATCH_SIZE={args.BATCH_SIZE}")
    print(f"记录点: {record_epochs}")
    print(f"{'='*60}\n")

    # 1. VGG-16
    print("🚀 训练 VGG-16")
    vgg16_model = build_vgg16_cifar100(device, train_loader)
    vgg_start = time.time()
    vgg_metrics, vgg_checkpoints = train_model(vgg16_model, train_loader, test_loader, args.MAX_EPOCHS, VGG_LR, VGG_WEIGHT_DECAY, device, 'VGG-16', VGG_SCHEDULER_TYPE, record_epochs=record_epochs)
    print(f"VGG-16 完成: Acc={vgg_metrics['accuracy']:.2f}%, 用时 {timedelta(seconds=int(time.time()-vgg_start))}\n")

    # 2. ResNet-50 Enhanced
    print("🚀 训练 ResNet-50 Enhanced")
    resnet50_enh = resnet50(pretrained=False)
    resnet50_enh.fc = nn.Sequential(
        nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 100)
    )
    resnet_start = time.time()
    resnet_metrics, resnet_checkpoints = train_model(resnet50_enh, train_loader, test_loader, args.MAX_EPOCHS, RESNET_LR, RESNET_WEIGHT_DECAY, device, 'ResNet-50', scheduler_type='cosine', record_epochs=record_epochs)
    print(f"ResNet-50 Enhanced 完成: Acc={resnet_metrics['accuracy']:.2f}%, 用时 {timedelta(seconds=int(time.time()-resnet_start))}\n")

    # 3. VDLF-Net
    print("🚀 训练 VDLF-Net")
    vdlfnet = VDLFNet(backbone='resnet50', latent_dim=VDLF_LATENT_DIM, num_classes=100, alpha=VDLF_ALPHA)
    vdlf_start = time.time()
    vdlf_metrics, vdlf_checkpoints = train_vdlfnet(vdlfnet, train_loader, test_loader, args.MAX_EPOCHS, VDLF_LR, VDLF_WEIGHT_DECAY, device, alpha=VDLF_ALPHA, kl_anneal_epochs=VDLF_KL_ANNEAL_EPOCHS, record_epochs=record_epochs)
    print(f"VDLF-Net 完成: Acc={vdlf_metrics['accuracy']:.2f}%, 用时 {timedelta(seconds=int(time.time()-vdlf_start))}\n")

    for ep in record_epochs:
        vg = vgg_checkpoints.get(ep, vgg_metrics)
        rn = resnet_checkpoints.get(ep, resnet_metrics)
        vd = vdlf_checkpoints.get(ep, vdlf_metrics)
        append_results_csv(args.OUTPUT_CSV, ep, args.BATCH_SIZE, vg, rn, vd)

    mean_diff = (vdlf_metrics['accuracy'] - resnet_metrics['accuracy'] + vdlf_metrics['precision'] - resnet_metrics['precision'] +
                 vdlf_metrics['recall'] - resnet_metrics['recall'] + vdlf_metrics['f1'] - resnet_metrics['f1']) / 4.0
    print(f"📊 最终 ResNet vs VDLF 均值差: {mean_diff:+.4f} (正值表示 VDLF 更好)")


if __name__ == '__main__':
    main()
