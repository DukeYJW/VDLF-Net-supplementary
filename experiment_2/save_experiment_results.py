"""
实验设置与结果保存工具
将 Table 2 相关配置和实验结果保存为 JSON（可追溯）+ Markdown（可读）
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path(__file__).parent
EXPERIMENT_JSON = LOG_DIR / "experiment_log.json"
EXPERIMENT_MD = LOG_DIR / "EXPERIMENT_RESULTS.md"


def config_to_dict(config) -> Dict[str, Any]:
    """将 FewShotConfig 转为可序列化字典"""
    if config is None:
        return {}
    d = {}
    for k, v in config.__dict__.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            d[k] = v
        elif hasattr(v, "__str__"):
            d[k] = str(v)
    return d


def save_experiment(
    config,
    results_1shot: Dict[str, tuple],
    results_5shot: Dict[str, tuple],
    note: str = "",
) -> None:
    """
    保存实验设置和结果到 JSON 和 Markdown。

    Args:
        config: FewShotConfig 实例（可为 None）
        results_1shot: {'MAML': (acc,ci,tr,te), 'Prototypical Networks': ..., ...}
        results_5shot: 同上
        note: 可选备注（如 "quick test", "full run"）
    """
    def _tup_to_dict(t):
        if t is None or (isinstance(t, tuple) and all(x is None for x in t)):
            return None
        acc, ci, tr, te = (t[0], t[1], t[2] if len(t) > 2 else None, t[3] if len(t) > 3 else None)
        return {
            "acc": round(acc * 100, 2) if acc is not None else None,
            "ci": round(ci * 100, 2) if ci is not None else None,
            "train_time_s": round(tr, 1) if tr is not None else None,
            "test_time_s": round(te, 1) if te is not None else None,
        }

    entry = {
        "timestamp": datetime.now().isoformat(),
        "note": note,
        "settings": config_to_dict(config) if config else {},
        "1shot": {k: _tup_to_dict(v) for k, v in results_1shot.items()} if results_1shot else {},
        "5shot": {k: _tup_to_dict(v) for k, v in results_5shot.items()} if results_5shot else {},
    }

    # 追加到 JSON 日志
    log = []
    if EXPERIMENT_JSON.exists():
        try:
            with open(EXPERIMENT_JSON, "r", encoding="utf-8") as f:
                log = json.load(f)
        except (json.JSONDecodeError, TypeError):
            log = []
    log.append(entry)
    with open(EXPERIMENT_JSON, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # 写入/追加 Markdown 摘要（仅最新一条）
    _write_md_summary(entry)

    print(f"已保存到 {EXPERIMENT_JSON} 和 {EXPERIMENT_MD}")


def _write_md_summary(entry: dict) -> None:
    """将最新一条记录写入 Markdown"""
    lines = [
        "# Table 2 实验记录",
        "",
        f"> 最后更新: {entry['timestamp']}",
        "",
        "## 实验设置",
        "",
        "| 参数 | 值 |",
        "|------|-----|",
    ]
    s = entry.get("settings", {})
    for k, v in s.items():
        lines.append(f"| {k} | {v} |")
    lines.extend(["", "## 1-shot 结果", ""])

    r1 = entry.get("1shot", {})
    if r1:
        lines.append("| Model | Acc (%) | CI (%) | 训练耗时 | 测试耗时 |")
        lines.append("|-------|---------|--------|----------|----------|")
        for name, d in r1.items():
            if d is None:
                lines.append(f"| {name} | — | — | — | — |")
            else:
                acc = d.get("acc") if d.get("acc") is not None else "—"
                ci = d.get("ci") if d.get("ci") is not None else "—"
                tr = f"{d.get('train_time_s', 0)/60:.1f}min" if d.get("train_time_s") else "—"
                te = f"{d.get('test_time_s', 0):.1f}s" if d.get("test_time_s") else "—"
                lines.append(f"| {name} | {acc} | {ci} | {tr} | {te} |")
    lines.extend(["", "## 5-shot 结果", ""])

    r5 = entry.get("5shot", {})
    if r5:
        lines.append("| Model | Acc (%) | CI (%) | 训练耗时 | 测试耗时 |")
        lines.append("|-------|---------|--------|----------|----------|")
        for name, d in r5.items():
            if d is None:
                lines.append(f"| {name} | — | — | — | — |")
            else:
                acc = d.get("acc") if d.get("acc") is not None else "—"
                ci = d.get("ci") if d.get("ci") is not None else "—"
                tr = f"{d.get('train_time_s', 0)/60:.1f}min" if d.get("train_time_s") else "—"
                te = f"{d.get('test_time_s', 0):.1f}s" if d.get("test_time_s") else "—"
                lines.append(f"| {name} | {acc} | {ci} | {tr} | {te} |")
    else:
        lines.append("（未填写 5-shot 结果）")

    lines.extend(["", "---", f"*备注: {entry.get('note', '')}*"])
    with open(EXPERIMENT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
