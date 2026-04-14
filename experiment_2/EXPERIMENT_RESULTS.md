# Table 2 实验记录

> 最后更新: 2026-02-12T12:14:32.117429

## 实验设置

| 参数 | 值 |
|------|-----|
| dataset | mini-imagenet |
| image_size | 84 |
| num_train_classes | 64 |
| num_val_classes | 16 |
| num_test_classes | 20 |
| n_way | 5 |
| k_shot | 5 |
| q_query | 15 |
| num_test_episodes | 100 |
| num_train_episodes | 150 |
| batch_size | 1 |
| gradient_accumulation_steps | 4 |
| learning_rate | 0.0001 |
| weight_decay | 5e-05 |
| num_epochs | 100 |
| latent_dim | 128 |
| num_scales | 2 |
| num_samples | 15 |
| temperature | 15.0 |
| alpha | 0.01 |
| device | cuda |
| seed | 42 |

## 1-shot 结果


## 5-shot 结果

| Model | Acc (%) | CI (%) | 训练耗时 | 测试耗时 |
|-------|---------|--------|----------|----------|
| MAML | — | — | — | — |
| Prototypical Networks | 80.09 | 1.14 | 0.8min | 12.0s |
| Matching Networks | 68.87 | 1.87 | 0.8min | 11.6s |
| VDLF-Net | 86.33 | 1.17 | 5.1min | 13.0s |

---
*备注: train_ep=150, test_ep=100*