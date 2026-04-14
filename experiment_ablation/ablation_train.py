"""消融训练与测试：复用 experiment_framework 的评估逻辑。"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from ablation_spec import AblationSpec
from vdlf_ablation import VDLFNetAblation, compute_vdlf_loss_ablation


def train_vdlf_ablation(
    model: VDLFNetAblation,
    train_sampler,
    val_sampler,
    config,
    spec: AblationSpec,
    ckpt_dir: Path | None = None,
    val_interval: int | None = None,
):
    """与 experiment_framework.train_vdlf_net 相同训练循环，损失改用 compute_vdlf_loss_ablation。"""
    from experiment_framework import evaluate_model

    if ckpt_dir is None:
        ckpt_dir = Path(".")
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_ablation_{spec.safe_name}.pth"

    if "cuda" in str(config.device):
        torch.cuda.empty_cache()
    device = torch.device(config.device)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_train_episodes, eta_min=0.1 * config.learning_rate
    )

    best_val_acc = 0.0
    log_interval = max(1, config.num_train_episodes // 50)
    if val_interval is None:
        val_interval = max(25, config.num_train_episodes // 20)
    saved_any = False

    accum = max(1, config.gradient_accumulation_steps)
    effective_batch = max(1, config.batch_size) * accum

    iterator = range(config.num_train_episodes)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"Train[{spec.name}]", unit="ep", mininterval=1.0)

    start_time = time.time()
    for episode_idx in iterator:
        model.train()
        optimizer.zero_grad()
        batch_losses = []

        for _ in range(effective_batch):
            s_i, s_l, q_i, q_l = train_sampler.sample_episode()
            s_i = s_i.to(device)
            s_l = s_l.to(device)
            q_i = q_i.to(device)
            q_l = q_l.to(device)

            total_loss, _, _, _ = compute_vdlf_loss_ablation(model, s_i, s_l, q_i, q_l, config, spec)
            (total_loss / effective_batch).backward()
            batch_losses.append(total_loss.item())

        optimizer.step()
        scheduler.step()
        batch_loss = sum(batch_losses) / len(batch_losses)

        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{batch_loss:.3f}", best_val=f"{best_val_acc:.2%}")

        if (episode_idx + 1) % val_interval == 0:
            val_acc, val_ci, _, _ = evaluate_model(model, val_sampler, config, show_progress=False)
            msg = f"Ep {episode_idx+1}/{config.num_train_episodes} | Val: {val_acc:.2%} ± {val_ci:.2%}"
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                saved_any = True
                msg += f" → saved {ckpt_path.name}"
            if tqdm is not None and hasattr(iterator, "write"):
                iterator.write(msg)
            else:
                print(msg)

    if not saved_any:
        torch.save(model.state_dict(), ckpt_path)

    total_time = time.time() - start_time
    return model, total_time, ckpt_path


def eval_ablation_on_test(model, test_sampler, config, ckpt_path: Path | None = None):
    from experiment_framework import evaluate_model

    if ckpt_path is not None and ckpt_path.is_file():
        model.load_state_dict(torch.load(ckpt_path, map_location=config.device))
    model = model.to(config.device)
    mean_acc, ci_95, accs, t_test = evaluate_model(model, test_sampler, config, show_progress=True)
    return mean_acc, ci_95, np.array(accs), t_test
