"""
Few-Shot Learning 基线方法
与 VDLF-Net 使用相同的 episode 采样和评估流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import warnings

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import learn2learn as l2l
    L2L_AVAILABLE = True
except ImportError:
    L2L_AVAILABLE = False


def get_resnet50_backbone():
    """获取 ResNet-50 backbone（与 VDLF-Net 一致）"""
    import torchvision.models as models
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except (AttributeError, TypeError):
            resnet = models.resnet50(pretrained=True)
    return nn.Sequential(*list(resnet.children())[:-2])  # 移除最后两层


# ==================== Prototypical Networks ====================

class PrototypicalNetworks(nn.Module):
    """Prototypical Networks: 类别原型 + 欧氏距离"""

    def __init__(self):
        super().__init__()
        self.backbone = get_resnet50_backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """提取并 L2 归一化特征"""
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return F.normalize(features, p=2, dim=1)


# ==================== MAML (via learn2learn) ====================

def get_maml_model(ways):
    """获取 MAML 模型，需安装 learn2learn: pip install learn2learn"""
    if not L2L_AVAILABLE:
        raise ImportError("请安装 learn2learn: pip install learn2learn")
    model = l2l.vision.models.MiniImagenetCNN(ways)
    return l2l.algorithms.MAML(model, lr=0.5, first_order=False)


def train_maml(maml, train_sampler, val_sampler, config, adaptation_steps=5, meta_batch_size=4, show_progress=True):
    """训练 MAML，使用我们的 EpisodeSampler"""
    if not L2L_AVAILABLE:
        raise ImportError("请安装 learn2learn: pip install learn2learn")
    device = torch.device(config.device)
    maml = maml.to(device)
    opt = optim.Adam(maml.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()

    n_way, k_shot = config.n_way, config.k_shot
    val_interval = max(10, config.num_train_episodes // 10)
    best_val_acc = 0.0

    iterator = range(config.num_train_episodes)
    if show_progress and tqdm:
        iterator = tqdm(iterator, desc="MAML", unit="ep", mininterval=1.0)

    t_start = time.time()
    for ep in iterator:
        opt.zero_grad()
        meta_loss = 0.0
        for _ in range(meta_batch_size):
            s_imgs, s_lbl, q_imgs, q_lbl = train_sampler.sample_episode()
            s_imgs = s_imgs.to(device)
            s_lbl = s_lbl.to(device)
            q_imgs = q_imgs.to(device)
            q_lbl = q_lbl.to(device)

            learner = maml.clone()
            for _ in range(adaptation_steps):
                adapt_loss = loss_fn(learner(s_imgs), s_lbl)
                learner.adapt(adapt_loss)

            eval_loss = loss_fn(learner(q_imgs), q_lbl)
            eval_loss.backward()
            meta_loss += eval_loss.item()

        for p in maml.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

        if (ep + 1) % val_interval == 0:
            maml.eval()
            accs = []
            with torch.no_grad():
                for _ in range(min(20, config.num_test_episodes)):
                    s_imgs, s_lbl, q_imgs, q_lbl = val_sampler.sample_episode()
                    s_imgs = s_imgs.to(device)
                    q_imgs = q_imgs.to(device)
                    q_lbl = q_lbl.to(device)
                    learner = maml.clone()
                    for _ in range(adaptation_steps):
                        adapt_loss = loss_fn(learner(s_imgs), s_lbl.to(device))
                        learner.adapt(adapt_loss)
                    pred = learner(q_imgs).argmax(dim=1)
                    accs.append((pred == q_lbl).float().mean().item())
            val_acc = np.mean(accs)
            maml.train()
            if show_progress and tqdm and isinstance(iterator, tqdm):
                iterator.write(f"Ep {ep+1} | Val: {val_acc:.2%}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(maml.state_dict(), "best_maml.pth")

    train_time = time.time() - t_start
    return maml, train_time


def evaluate_maml(maml, sampler, config, n_way, k_shot, adaptation_steps=5):
    """评估 MAML，返回 (mean_acc, ci, test_time_seconds)"""
    if not L2L_AVAILABLE:
        raise ImportError("请安装 learn2learn: pip install learn2learn")
    device = torch.device(config.device)
    maml.eval()
    loss_fn = nn.CrossEntropyLoss()
    accs = []
    it = range(config.num_test_episodes)
    if tqdm:
        it = tqdm(it, desc="Eval MAML", unit="ep")
    t_start = time.time()
    with torch.no_grad():
        for _ in it:
            s_imgs, s_lbl, q_imgs, q_lbl = sampler.sample_episode()
            s_imgs = s_imgs.to(device)
            q_imgs = q_imgs.to(device)
            q_lbl = q_lbl.to(device)
            s_lbl = s_lbl.to(device)
            learner = maml.clone()
            for _ in range(adaptation_steps):
                adapt_loss = loss_fn(learner(s_imgs), s_lbl)
                learner.adapt(adapt_loss)
            pred = learner(q_imgs).argmax(dim=1)
            accs.append((pred == q_lbl).float().mean().item())
    test_time = time.time() - t_start
    mean_acc = np.mean(accs)
    ci = 1.96 * np.std(accs) / np.sqrt(len(accs))
    return mean_acc, ci, test_time


# ==================== Matching Networks ====================

class MatchingNetworks(nn.Module):
    """Matching Networks: 注意力匹配"""

    def __init__(self):
        super().__init__()
        self.backbone = get_resnet50_backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """提取并 L2 归一化特征"""
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return F.normalize(features, p=2, dim=1)


# ==================== 训练与评估 ====================

def train_prototypical(model, train_sampler, val_sampler, config, show_progress=True, lr=1e-4):
    """训练 Prototypical Networks，返回 (model, train_time_seconds)。lr=1e-4 适配 ResNet 微调"""
    model = model.to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    val_interval = max(10, config.num_train_episodes // 10)
    best_val_acc = 0.0
    t_start = time.time()

    iterator = range(config.num_train_episodes)
    if show_progress and tqdm:
        iterator = tqdm(iterator, desc="ProtoNet", unit="ep", mininterval=1.0)

    for ep in iterator:
        model.train()
        support_images, support_labels, query_images, query_labels = train_sampler.sample_episode()
        support_images = support_images.to(config.device)
        support_labels = support_labels.to(config.device)
        query_images = query_images.to(config.device)
        query_labels = query_labels.to(config.device)

        support_features = model(support_images)
        query_features = model(query_images)

        n_way, k_shot = config.n_way, config.k_shot
        prototypes = support_features.view(n_way, k_shot, -1).mean(dim=1)
        logits = -torch.cdist(query_features, prototypes)
        loss = F.cross_entropy(logits, query_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % val_interval == 0:
            model.eval()
            accs = []
            with torch.no_grad():
                for _ in range(min(20, config.num_test_episodes)):
                    s_imgs, s_lbl, q_imgs, q_lbl = val_sampler.sample_episode()
                    s_imgs = s_imgs.to(config.device)
                    q_imgs = q_imgs.to(config.device)
                    q_lbl = q_lbl.to(config.device)
                    s_feat = model(s_imgs)
                    q_feat = model(q_imgs)
                    proto = s_feat.view(n_way, k_shot, -1).mean(dim=1)
                    pred = (-torch.cdist(q_feat, proto)).argmax(dim=1)
                    accs.append((pred == q_lbl).float().mean().item())
            val_acc = np.mean(accs)
            if show_progress and tqdm and isinstance(iterator, tqdm):
                iterator.write(f"Ep {ep+1} | Val: {val_acc:.2%}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_protonet.pth")

    train_time = time.time() - t_start
    return model, train_time


def train_matching(model, train_sampler, val_sampler, config, show_progress=True, lr=1e-4):
    """训练 Matching Networks（余弦相似度注意力），返回 (model, train_time_seconds)。lr=1e-4 适配 ResNet 微调"""
    model = model.to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    val_interval = max(10, config.num_train_episodes // 10)
    best_val_acc = 0.0
    t_start = time.time()

    iterator = range(config.num_train_episodes)
    if show_progress and tqdm:
        iterator = tqdm(iterator, desc="MatchNet", unit="ep", mininterval=1.0)

    for ep in iterator:
        model.train()
        support_images, support_labels, query_images, query_labels = train_sampler.sample_episode()
        support_images = support_images.to(config.device)
        support_labels = support_labels.to(config.device)
        query_images = query_images.to(config.device)
        query_labels = query_labels.to(config.device)

        support_features = model(support_images)
        query_features = model(query_images)

        # Matching Networks: 注意力加权按类别聚合
        n_way, k_shot, q_query = config.n_way, config.k_shot, config.q_query
        # support_labels: [N*K], 值为 0..N-1
        # query 每个样本对应一个类，query_labels 也是 0..N-1
        # 我们直接用 nearest-neighbor: 找 query 与每个 support 的相似度，然后按 support 的类聚合
        sim = torch.mm(query_features, support_features.t())
        # 扩展 support_labels 用于聚合
        support_onehot = F.one_hot(support_labels, n_way).float()
        logits = torch.mm(sim, support_onehot)
        loss = F.cross_entropy(logits, query_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % val_interval == 0:
            model.eval()
            accs = []
            with torch.no_grad():
                for _ in range(min(20, config.num_test_episodes)):
                    s_imgs, s_lbl, q_imgs, q_lbl = val_sampler.sample_episode()
                    s_imgs = s_imgs.to(config.device)
                    q_imgs = q_imgs.to(config.device)
                    q_lbl = q_lbl.to(config.device)
                    s_feat = model(s_imgs)
                    q_feat = model(q_imgs)
                    sim = torch.mm(q_feat, s_feat.t())
                    s_onehot = F.one_hot(s_lbl, n_way).float().to(config.device)
                    logits = torch.mm(sim, s_onehot)
                    pred = logits.argmax(dim=1)
                    accs.append((pred == q_lbl).float().mean().item())
            val_acc = np.mean(accs)
            if show_progress and tqdm and isinstance(iterator, tqdm):
                iterator.write(f"Ep {ep+1} | Val: {val_acc:.2%}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_matchnet.pth")

    train_time = time.time() - t_start
    return model, train_time


def evaluate_baseline(model, sampler, config, n_way, k_shot, model_type="protonet"):
    """评估基线模型，返回 (mean_acc, ci, test_time_seconds)。model_type: 'protonet' 或 'matchnet'"""
    model.eval()
    accs = []
    it = range(config.num_test_episodes)
    if tqdm:
        it = tqdm(it, desc=f"Eval {model_type}", unit="ep")
    t_start = time.time()
    with torch.no_grad():
        for _ in it:
            s_imgs, s_lbl, q_imgs, q_lbl = sampler.sample_episode()
            s_imgs = s_imgs.to(config.device)
            q_imgs = q_imgs.to(config.device)
            q_lbl = q_lbl.to(config.device)
            s_lbl = s_lbl.to(config.device)
            s_feat = model(s_imgs)
            q_feat = model(q_imgs)
            if model_type == "protonet":
                proto = s_feat.view(n_way, k_shot, -1).mean(dim=1)
                pred = (-torch.cdist(q_feat, proto)).argmax(dim=1)
            else:
                sim = torch.mm(q_feat, s_feat.t())
                s_onehot = F.one_hot(s_lbl, n_way).float().to(config.device)
                logits = torch.mm(sim, s_onehot)
                pred = logits.argmax(dim=1)
            accs.append((pred == q_lbl).float().mean().item())
    test_time = time.time() - t_start
    mean_acc = np.mean(accs)
    ci = 1.96 * np.std(accs) / np.sqrt(len(accs))
    return mean_acc, ci, test_time
