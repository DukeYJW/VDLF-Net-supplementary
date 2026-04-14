"""消融实验规格：与 experiment_framework.VDLFNet / compute_vdlf_loss 配合使用。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

GatingMode = Literal["learned", "uniform", "coarse_only", "fine_only"]


@dataclass
class AblationSpec:
    """单组消融配置。

    - drop_kl / drop_recon: 在总损失中去掉对应变分项（与原论文 L_total = CE + α(Recon+KL) 对齐）。
    - gating_mode:
        learned: 默认，由门控网络 softmax 产生权重；
        uniform: 两尺度固定各 0.5；
        coarse_only: 仅使用索引 0（2×2 池化分支）；
        fine_only: 仅使用索引 1（1×1 全局分支）。
    """

    name: str
    drop_kl: bool = False
    drop_recon: bool = False
    gating_mode: GatingMode = "learned"

    @property
    def safe_name(self) -> str:
        return "".join(c if c.isalnum() else "_" for c in self.name).strip("_")


def default_ablation_presets() -> List[AblationSpec]:
    """审稿常用消融组合；可按计算预算删行。"""
    return [
        AblationSpec("full", drop_kl=False, drop_recon=False, gating_mode="learned"),
        AblationSpec("no_kl", drop_kl=True, drop_recon=False, gating_mode="learned"),
        AblationSpec("no_recon", drop_kl=False, drop_recon=True, gating_mode="learned"),
        AblationSpec("no_var", drop_kl=True, drop_recon=True, gating_mode="learned"),
        AblationSpec("uniform_gate", drop_kl=False, drop_recon=False, gating_mode="uniform"),
        AblationSpec("fine_scale_only", drop_kl=False, drop_recon=False, gating_mode="fine_only"),
        AblationSpec("coarse_scale_only", drop_kl=False, drop_recon=False, gating_mode="coarse_only"),
    ]
