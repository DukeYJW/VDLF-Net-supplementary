# Supplementary materials: VDLF-Net

Official supplementary code and experimental materials for the manuscript:

**VDLF-Net: Variational Feature Fusion for Adaptive and Few-Shot Visual Learning**

*(Pattern Analysis and Applications — reproducibility package.)*

This repository preserves the **working layout** used by the authors: runnable code, notebooks, shell scripts, raw result tables (CSV), and implementation notes. Large pretrained weights and full dataset archives are **not** hosted on GitHub due to file-size limits. The **complete supplementary ZIP** (code, datasets, checkpoints, and tables as in our experiments) is archived on **Zenodo** (see below).

---

## Repository layout

| Folder | Contents |
|--------|----------|
| **`exp1codes/`** | **Experiment 1 (final, recommended).** Table 1 scripts (`train_vdlfnet.py`, `table1_compare.py`, …), notebook `table1_experiment.ipynb`, grid-search helpers, CSV results, optimization notes. |
| **`experiment_1/`** | **Experiment 1 (earlier tree).** Same experiment line with legacy notebooks/tables; includes the same CIFAR-100 path convention under `experiment_1/data/` when you run from this folder. |
| **`experiment_2/`** | **Experiment 2 (few-shot, Mini-ImageNet).** `experiment_framework.py`, baselines, notebook `table2_few_shot_experiment.ipynb`, logs, guides. |
| **`experiment_ablation/`** | **Ablation studies** on Mini-ImageNet; uses the **same** prepared data as Experiment 2 (see paths below). |

**Why two Experiment 1 folders?**  
`exp1codes` is the **final runnable** implementation reviewers should start from. `experiment_1` is kept for **completeness** (earlier working copy and records).

---

## Environment

- **Python** 3.10+ recommended (3.12/3.13 were used in parts of the ablation cache filenames).
- **PyTorch** + **torchvision** with CUDA matching your GPU (install wheels from [pytorch.org](https://pytorch.org/) if needed).

```bash
pip install -r requirements.txt
```

---

## Dataset locations (reproduce as in the paper)

### CIFAR-100 (Experiment 1)

- **`exp1codes/data/`**  
  Place the torchvision-style folder **`cifar-100-python/`** here when running scripts/notebooks with working directory `exp1codes/`.  
  Scripts accept `--DATA_ROOT` (default `data`).

- **`experiment_1/data/`**  
  Same layout when the working directory is **`experiment_1/`** (legacy notebooks).

You may use `torchvision.datasets.CIFAR100(..., download=True)` once to populate `data/`, or copy an existing `cifar-100-python` tree from your full supplementary archive.

### Mini-ImageNet (Experiment 2 + ablation)

- **`experiment_2/data/mini-imagenet/`**  
  Required layout is documented in **`experiment_2/README.md`** and can be checked with:

```bash
cd experiment_2
python check_mini_imagenet.py --data_dir data/mini-imagenet
```

- **`experiment_ablation/`**  
  Notebooks/scripts resolve data as **`../experiment_2/data/mini-imagenet`** relative to the ablation folder — keep **one** copy of the dataset under `experiment_2/data/` to avoid duplication.

---

## Full supplementary archive (Zenodo)

Due to GitHub limits (~100 MB per file; large repositories are discouraged), this repository does **not** include full datasets or `.pth` checkpoints. Download the **complete ZIP** used in the paper from Zenodo:

- **DOI:** [https://doi.org/10.5281/zenodo.19580381](https://doi.org/10.5281/zenodo.19580381)  
- **Record page:** [https://zenodo.org/records/19580381](https://zenodo.org/records/19580381)

The archive contains `exp1codes/`, `experiment_1/`, `experiment_2/`, and `experiment_ablation/` with data and checkpoints as in our working environment. After unzip, place or merge folders so that paths match the **`README.md`** files under:

- `exp1codes/data/`
- `experiment_1/data/`
- `experiment_2/data/`
- `experiment_ablation/ablation_checkpoints/`

---

## Quick run hints

- **Experiment 1:** from `exp1codes/`, see `run_table1_compare.sh`, `run_vdlf_grid_search.sh`, and `train_vdlfnet.py --help`.
- **Experiment 2:** open `experiment_2/table2_few_shot_experiment.ipynb` with kernel cwd = `experiment_2`.
- **Ablation:** open `experiment_ablation/ablation_suite.ipynb` with cwd = `experiment_ablation` (see notebook header).

---

## Note to editors / reviewers

The supplementary package includes runnable source code, scripts, raw experimental tables, and implementation notes. The **complete archive** (including large checkpoints and full dataset trees) is available on **Zenodo** via the DOI above; this GitHub repository provides a lightweight, browsable copy of the code and tables.

---

## Contact

**Jiawei Yan** — Shanghai Jiao Tong University  

For questions about reproducing these experiments or accessing the materials: **[yjw1998@sjtu.edu.cn](mailto:yjw1998@sjtu.edu.cn)**

---

**Repository:** [https://github.com/DukeYJW/VDLF-Net-supplementary](https://github.com/DukeYJW/VDLF-Net-supplementary)  

If you clone this repository and need the full datasets and checkpoints, download the **Zenodo** ZIP using the DOI under **Full supplementary archive (Zenodo)** above.
