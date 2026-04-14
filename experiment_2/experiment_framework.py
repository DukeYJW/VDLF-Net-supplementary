"""
Few-Shot Learning Experiment Framework for Table 2
实现VDLF-Net和基线方法的few-shot学习实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import random
import time

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# ==================== 配置类 ====================

@dataclass
class FewShotConfig:
    """Few-shot实验配置"""
    # 数据集配置
    dataset: str = "mini-imagenet"
    image_size: int = 84
    num_train_classes: int = 64
    num_val_classes: int = 16
    num_test_classes: int = 20
    
    # Episode配置
    n_way: int = 5
    k_shot: int = 1  # 或5
    q_query: int = 15
    num_test_episodes: int = 600
    
    # 训练配置
    num_train_episodes: int = 40000
    batch_size: int = 16  # episodes per batch
    gradient_accumulation_steps: int = 1  # >1 时模拟大 batch，显存不足时用
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    num_epochs: int = 100
    
    # VDLF-Net特定配置
    latent_dim: int = 128
    num_scales: int = 2  # K
    num_samples: int = 10  # T for support set
    temperature: float = 10.0  # τ
    alpha: float = 0.01  # 变分正则化系数（与 table2_few_shot_experiment.ipynb 及论文 Table 2 一致）
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    def __post_init__(self):
        """设置随机种子"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


# ==================== Episode采样器 ====================

class EpisodeSampler:
    """采样few-shot learning episodes"""
    
    def __init__(self, dataset, n_way: int, k_shot: int, q_query: int, class_to_indices=None):
        """
        Args:
            dataset: 数据集（包含images和labels）
            n_way: N-way分类
            k_shot: K-shot支持样本数
            q_query: 每个类别的查询样本数
            class_to_indices: 可选，预构建的 {label: [indices]} 映射，可避免遍历整个数据集
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        
        if class_to_indices is not None:
            self.class_to_indices = defaultdict(list, class_to_indices)
            self.available_classes = list(self.class_to_indices.keys())
        else:
            self.class_to_indices = defaultdict(list)
            for idx, (_, label) in enumerate(dataset):
                self.class_to_indices[label].append(idx)
            self.available_classes = list(self.class_to_indices.keys())
    
    def sample_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样一个episode
        
        Returns:
            support_images: [N*K, C, H, W]
            support_labels: [N*K]
            query_images: [N*Q, C, H, W]
            query_labels: [N*Q]
        """
        # 随机选择N个类别
        selected_classes = random.sample(self.available_classes, self.n_way)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for class_idx, class_label in enumerate(selected_classes):
            # 获取该类别的所有样本索引
            class_indices = self.class_to_indices[class_label]
            
            # 随机选择K+Q个样本
            selected_indices = random.sample(class_indices, self.k_shot + self.q_query)
            
            # 前K个作为支持集
            for i in range(self.k_shot):
                idx = selected_indices[i]
                img, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(class_idx)  # 重新映射到0-N-1
            
            # 后Q个作为查询集
            for i in range(self.k_shot, self.k_shot + self.q_query):
                idx = selected_indices[i]
                img, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(class_idx)
        
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)
        
        return support_images, support_labels, query_images, query_labels


# ==================== VDLF-Net实现 ====================

class VAEEncoder(nn.Module):
    """VAE编码器"""
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
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
    """VAE解码器"""
    
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        x_recon = self.fc3(z)
        return x_recon


class GatingNetwork(nn.Module):
    """Gating网络，生成融合权重"""
    
    def __init__(self, latent_dim: int, num_scales: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, num_scales)
        self.relu = nn.ReLU()
    
    def forward(self, z):
        z = self.relu(self.fc1(z))
        weights = self.fc2(z)
        return torch.softmax(weights, dim=-1)


class VDLFNet(nn.Module):
    """Variational-Deep Learning Fusion Network"""
    
    def __init__(self, config: FewShotConfig):
        super().__init__()
        self.config = config
        
        # ResNet-50 backbone（截断到layer4）
        import torchvision.models as models
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            except (AttributeError, TypeError):
                resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 移除最后两层
        
        # 多尺度特征提取
        self.adaptive_pool_2x2 = nn.AdaptiveAvgPool2d((2, 2))
        self.adaptive_pool_1x1 = nn.AdaptiveAvgPool2d((1, 1))
        
        # 计算特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, config.image_size, config.image_size)
            features = self.backbone(dummy_input)
            feat_dim_2x2 = features.shape[1] * 2 * 2
            feat_dim_1x1 = features.shape[1] * 1 * 1
        
        # 将 2x2 特征投影到与 1x1 相同维度，便于加权融合
        self.proj_2x2 = nn.Linear(feat_dim_2x2, feat_dim_1x1)
        
        # VAE
        initial_feat_dim = feat_dim_1x1  # 使用1x1作为初始融合特征
        self.vae_encoder = VAEEncoder(initial_feat_dim, config.latent_dim)
        self.vae_decoder = VAEDecoder(config.latent_dim, initial_feat_dim)
        
        # Gating网络
        self.gating_network = GatingNetwork(config.latent_dim, config.num_scales)
        
        # 特征维度映射（融合后统一为 feat_dim_1x1）
        self.feat_dims = [feat_dim_1x1, feat_dim_1x1]
    
    def extract_multiscale_features(self, x):
        """提取多尺度特征，投影到统一维度便于融合"""
        features = self.backbone(x)
        
        # 提取两个尺度的特征
        feat_2x2 = self.adaptive_pool_2x2(features)
        feat_1x1 = self.adaptive_pool_1x1(features)
        
        # Flatten
        feat_2x2 = feat_2x2.view(feat_2x2.size(0), -1)
        feat_1x1 = feat_1x1.view(feat_1x1.size(0), -1)
        
        # 将 2x2 投影到与 1x1 相同维度
        feat_2x2 = self.proj_2x2(feat_2x2)
        
        return [feat_2x2, feat_1x1]
    
    def compute_initial_fused_feature(self, multiscale_features):
        """计算初始融合特征（简单平均）"""
        return multiscale_features[-1]  # 使用1x1特征作为初始
    
    def forward(self, x, num_samples: int = 1, is_support: bool = False):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            num_samples: 采样数量（支持集使用T，查询集使用1）
            is_support: 是否为支持集
        
        Returns:
            fused_features: [B*num_samples, feat_dim] 或 [B, feat_dim]
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        batch_size = x.size(0)
        
        # 提取多尺度特征
        multiscale_features = self.extract_multiscale_features(x)
        
        # 计算初始融合特征
        initial_fused = self.compute_initial_fused_feature(multiscale_features)
        
        # VAE编码
        mu, logvar = self.vae_encoder(initial_fused)
        
        if is_support and num_samples > 1:
            # 支持集：采样多个潜在变量
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(batch_size, num_samples, self.config.latent_dim, 
                            device=x.device)
            z = mu.unsqueeze(1) + std.unsqueeze(1) * eps  # [B, T, latent_dim]
            z = z.view(-1, self.config.latent_dim)  # [B*T, latent_dim]
            
            # 生成融合权重
            weights = self.gating_network(z)  # [B*T, num_scales]
            
            # 融合多尺度特征
            multiscale_tensor = torch.stack(multiscale_features, dim=1)  # [B, num_scales, feat_dim]
            multiscale_tensor = multiscale_tensor.unsqueeze(1).expand(
                batch_size, num_samples, self.config.num_scales, -1
            )  # [B, T, num_scales, feat_dim]
            multiscale_tensor = multiscale_tensor.contiguous().view(
                batch_size * num_samples, self.config.num_scales, -1
            )  # [B*T, num_scales, feat_dim]
            
            weights_expanded = weights.unsqueeze(-1)  # [B*T, num_scales, 1]
            fused_features = (multiscale_tensor * weights_expanded).sum(dim=1)  # [B*T, feat_dim]
            
            # 归一化
            fused_features = self.normalize_features(fused_features)
            
            return fused_features, mu, logvar
        else:
            # 查询集：使用均值（确定性）
            z = mu
            
            # 生成融合权重
            weights = self.gating_network(z)  # [B, num_scales]
            
            # 融合多尺度特征
            multiscale_tensor = torch.stack(multiscale_features, dim=1)  # [B, num_scales, feat_dim]
            weights_expanded = weights.unsqueeze(-1)  # [B, num_scales, 1]
            fused_features = (multiscale_tensor * weights_expanded).sum(dim=1)  # [B, feat_dim]
            
            # 归一化
            fused_features = self.normalize_features(fused_features)
            
            return fused_features, mu, logvar
    
    def normalize_features(self, features):
        """L2归一化"""
        # 中心化
        mean = features.mean(dim=0, keepdim=True)
        features = features - mean
        
        # L2归一化
        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        features = features / (norm + 1e-8)
        
        return features
    
    def decode(self, z):
        """VAE解码"""
        return self.vae_decoder(z)


# ==================== 损失函数 ====================

def compute_vdlf_loss(model, support_images, support_labels, query_images, 
                     query_labels, config: FewShotConfig):
    """
    计算VDLF-Net的总损失
    
    Returns:
        total_loss: 总损失
        ce_loss: 交叉熵损失
        recon_loss: 重构损失
        kl_loss: KL散度损失
    """
    n_way = config.n_way
    k_shot = config.k_shot
    q_query = config.q_query
    num_samples = config.num_samples
    
    # 支持集：采样多个特征
    support_features, support_mu, support_logvar = model(
        support_images, num_samples=num_samples, is_support=True
    )  # [N*K*T, feat_dim]
    
    # 查询集：确定性特征
    query_features, query_mu, query_logvar = model(
        query_images, num_samples=1, is_support=False
    )  # [N*Q, feat_dim]
    
    # 计算类别原型
    support_features_reshaped = support_features.view(
        n_way, k_shot, num_samples, -1
    )  # [N, K, T, feat_dim]
    prototypes = support_features_reshaped.mean(dim=(1, 2))  # [N, feat_dim]
    
    # 计算余弦相似度
    query_features_expanded = query_features.unsqueeze(1)  # [N*Q, 1, feat_dim]
    prototypes_expanded = prototypes.unsqueeze(0)  # [1, N, feat_dim]
    
    cosine_sim = (query_features_expanded * prototypes_expanded).sum(dim=-1)  # [N*Q, N]
    
    # 温度缩放的softmax
    logits = cosine_sim * config.temperature
    ce_loss = nn.functional.cross_entropy(logits, query_labels)
    
    # 变分损失（对所有样本）
    all_images = torch.cat([support_images, query_images], dim=0)
    all_mu = torch.cat([support_mu, query_mu], dim=0)
    all_logvar = torch.cat([support_logvar, query_logvar], dim=0)
    
    # 重构损失
    all_features_initial = model.compute_initial_fused_feature(
        model.extract_multiscale_features(all_images)
    )
    std = torch.exp(0.5 * all_logvar)
    eps = torch.randn_like(std)
    z = all_mu + std * eps
    recon_features = model.decode(z)
    recon_loss = nn.functional.mse_loss(recon_features, all_features_initial)
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(
        1 + all_logvar - all_mu.pow(2) - all_logvar.exp()
    ) / all_images.size(0)
    
    # 总损失
    total_loss = ce_loss + config.alpha * (recon_loss + kl_loss)
    
    return total_loss, ce_loss, recon_loss, kl_loss


# ==================== 评估函数 ====================

def evaluate_episode(model, support_images, support_labels, query_images, 
                    query_labels, config: FewShotConfig):
    """评估单个episode的准确率"""
    model.eval()
    
    with torch.no_grad():
        # 支持集：采样多个特征
        support_features, _, _ = model(
            support_images, num_samples=config.num_samples, is_support=True
        )
        
        # 查询集：确定性特征
        query_features, _, _ = model(
            query_images, num_samples=1, is_support=False
        )
        
        # 计算类别原型
        n_way = config.n_way
        k_shot = config.k_shot
        support_features_reshaped = support_features.view(
            n_way, k_shot, config.num_samples, -1
        )
        prototypes = support_features_reshaped.mean(dim=(1, 2))  # [N, feat_dim]
        
        # 预测
        cosine_sim = torch.mm(query_features, prototypes.t())  # [N*Q, N]
        predictions = cosine_sim.argmax(dim=1)
        
        # 计算准确率
        accuracy = (predictions == query_labels).float().mean().item()
    
    return accuracy


def evaluate_model(model, sampler: EpisodeSampler, config: FewShotConfig, show_progress: bool = True):
    """评估模型在多个episodes上的性能，返回 (mean_acc, ci_95, accuracies, test_time_seconds)"""
    accuracies = []
    iterator = range(config.num_test_episodes)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc="Evaluating", unit="ep")
    t_start = time.time()
    for _ in iterator:
        support_images, support_labels, query_images, query_labels = sampler.sample_episode()
        
        # 移动到设备
        support_images = support_images.to(config.device)
        support_labels = support_labels.to(config.device)
        query_images = query_images.to(config.device)
        query_labels = query_labels.to(config.device)
        
        accuracy = evaluate_episode(
            model, support_images, support_labels, query_images, query_labels, config
        )
        accuracies.append(accuracy)
    
    test_time = time.time() - t_start
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    ci_95 = 1.96 * std_acc / np.sqrt(len(accuracies))
    
    return mean_acc, ci_95, accuracies, test_time


# ==================== 训练函数 ====================

def train_vdlf_net(model, train_sampler: EpisodeSampler, val_sampler: EpisodeSampler,
                  config: FewShotConfig):
    """训练VDLF-Net"""
    if "cuda" in config.device:
        torch.cuda.empty_cache()
    model = model.to(config.device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_train_episodes, eta_min=0.1 * config.learning_rate
    )
    
    best_val_acc = 0.0
    log_interval = max(1, config.num_train_episodes // 50)
    val_interval = max(25, config.num_train_episodes // 20)  # 更频繁验证，便于捕捉早期较优 checkpoint
    saved_any = False
    
    # 时间统计
    t_first_batch = 0.0
    t_train_batches = []
    t_val_total = 0.0
    
    print("进入训练循环，首个 batch 可能较慢（GPU 预热）...", flush=True)
    
    iterator = range(config.num_train_episodes)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Training", unit="ep", 
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        mininterval=1.0)
    
    start_time = time.time()
    
    accum = config.gradient_accumulation_steps
    effective_batch = config.batch_size * accum
    
    for episode_idx in iterator:
        model.train()
        optimizer.zero_grad()
        batch_losses = []
        
        for _ in range(effective_batch):
            support_images, support_labels, query_images, query_labels = train_sampler.sample_episode()
            
            support_images = support_images.to(config.device)
            support_labels = support_labels.to(config.device)
            query_images = query_images.to(config.device)
            query_labels = query_labels.to(config.device)
            
            total_loss, ce_loss, recon_loss, kl_loss = compute_vdlf_loss(
                model, support_images, support_labels, query_images, query_labels, config
            )
            # 梯度累积：每次 backward 前缩放，等效于对 mean 求导
            loss_scaled = total_loss / effective_batch
            loss_scaled.backward()
            batch_losses.append(total_loss.item())
        
        optimizer.step()
        scheduler.step()
        batch_loss = sum(batch_losses) / len(batch_losses)
        
        # 首个 batch 完成后立即提示
        if episode_idx == 0:
            t_first_batch = time.time() - start_time
            print(f"首个 batch 完成 (耗时 {t_first_batch:.1f}s)，进度条已启动", flush=True)
        
        # 更新进度条显示
        if tqdm is not None and isinstance(iterator, tqdm):
            iterator.set_postfix(loss=f"{batch_loss:.3f}", best_val=f"{best_val_acc:.2%}")
        
        # 定期打印损失
        if (episode_idx + 1) % log_interval == 0 and tqdm is None:
            elapsed = time.time() - start_time
            print(f"Ep {episode_idx+1}/{config.num_train_episodes} | loss={batch_loss:.3f} | "
                  f"elapsed={elapsed:.0f}s | best_val={best_val_acc:.2%}")
        
        # 验证
        if (episode_idx + 1) % val_interval == 0:
            t_v0 = time.time()
            val_acc, val_ci, _, _ = evaluate_model(model, val_sampler, config, show_progress=False)
            t_val_total += time.time() - t_v0
            msg = f"Ep {episode_idx+1}/{config.num_train_episodes} | Val: {val_acc:.2%} ± {val_ci:.2%}"
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_vdlf_model.pth")
                saved_any = True
                msg += " (saved)"
            if tqdm is not None and isinstance(iterator, tqdm):
                iterator.write(msg)
            else:
                print(msg)
    
    # 若从未验证（如 num_train_episodes < val_interval），保存最终模型
    if not saved_any:
        torch.save(model.state_dict(), "best_vdlf_model.pth")
        if tqdm is not None and isinstance(iterator, tqdm):
            iterator.write(f"[未验证] 已保存最终模型 (ep {config.num_train_episodes})")

    # 打印时间统计
    total_time = time.time() - start_time
    t_train = total_time - t_val_total
    n_val = (config.num_train_episodes + val_interval - 1) // val_interval
    print(f"\n--- 训练时间统计 ---", flush=True)
    print(f"总耗时:        {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"训练耗时:      {t_train:.1f}s (不含验证)", flush=True)
    print(f"验证耗时:      {t_val_total:.1f}s (共 {n_val} 次)", flush=True)
    print(f"首次 batch:    {t_first_batch:.1f}s (GPU 预热)", flush=True)
    print(f"平均每 ep:     {t_train/config.num_train_episodes:.2f}s", flush=True)
    print(f"-------------------", flush=True)
    
    return model, total_time


# ==================== 主函数 ====================

if __name__ == "__main__":
    # 配置
    config_1shot = FewShotConfig(k_shot=1)
    config_5shot = FewShotConfig(k_shot=5)
    
    # 注意：这里需要实现数据集加载
    # train_dataset = load_mini_imagenet_train()
    # val_dataset = load_mini_imagenet_val()
    # test_dataset = load_mini_imagenet_test()
    
    print("Few-Shot Learning Experiment Framework")
    print("请实现数据集加载和基线方法，然后运行完整实验")
