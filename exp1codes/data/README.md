# Placeholder for CIFAR-100 (torchvision layout)

Place the extracted `cifar-100-python` folder here.

Expected after preparation:
- `cifar-100-python/meta`
- `cifar-100-python/train`
- `cifar-100-python/test`

Scripts in `exp1codes` use `root="data"` relative to the **current working directory** when you run them from `exp1codes/`, or pass `--DATA_ROOT` to `train_vdlfnet.py` / `table1_compare.py`.

The legacy folder `experiment_1/data/` can hold the same layout if you run notebooks from `experiment_1/`.
