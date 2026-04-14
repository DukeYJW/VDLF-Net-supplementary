# experiment_ablation：VDLF-Net 消融与扩展实验

本目录与 `experiment_2` 共用 **Mini-ImageNet + `FewShotConfig` + `EpisodeSampler`**，不修改原有 `experiment_framework.py` 行为。

## 入口

- **`ablation_suite.ipynb`**：按顺序运行；先快速模式验证环境，再改参数跑正式表。
- **`ablation_spec.py`**：消融组合定义；`default_ablation_presets()` 为审稿常用 7 组。
- **`vdlf_ablation.py`**：`VDLFNetAblation`（门控：`learned` / `uniform` / 单尺度）；`compute_vdlf_loss_ablation`（去 KL / 去重建）。
- **`ablation_train.py`**：训练与保存 `best_ablation_<name>.pth`。
- **`mini_imagenet_data.py`**：数据集与增强，与 `table2_few_shot_experiment.ipynb` 一致。

## 数据路径

默认 `DATA_ROOT = Path("../experiment_2/data/mini-imagenet")`（notebook 内可改）。请与 Table 2 实验使用同一数据布局。

## 快速 / 正式

- **快速**：`NUM_TRAIN_EPISODES = 30`，`NUM_TEST_EPISODES = 10`，单组消融几分钟级。
- **正式（对齐论文表）**：与 `table2_few_shot_experiment.ipynb` Section 0 一致：`150` / `100` episodes，`T=15`，`τ=15`，`α=0.01`。

## 后续可扩展

- **新 baseline（如 MAML）**：在 notebook 新 section 调用 `experiment_2/baselines.py`（需 `learn2learn`），与同一 `test_sampler` 评估，结果写入同一汇总表。
