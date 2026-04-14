# Supplementary materials: VDLF-Net

Official supplementary code and experimental materials for the manuscript:

**VDLF-Net: Variational Feature Fusion for Adaptive and Few-Shot Visual Learning**

*(Pattern Analysis and Applications — reproducibility package.)*

This repository preserves the **working layout** used by the authors: runnable code, notebooks, shell scripts, raw result tables (CSV), and implementation notes. Large pretrained weights and full dataset archives are **not** hosted on GitHub due to file-size limits; they are distributed via **Dropbox** (links below — **replace the placeholders** after you upload).

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

## External large files (Dropbox) — **edit these URLs**

Due to GitHub limits (~100 MB per file; large repositories are discouraged), the following are **not** in this repo:

1. **Pretrained / baseline checkpoints for Experiment 2** (e.g. `best_vdlf_model.pth`, `best_matchnet.pth`, `best_protonet.pth` in the project root of `experiment_2/` in the original workspace).  
   **Download:** `REPLACE_WITH_DROPBOX_LINK_CHECKPOINTS_EXP2`

2. **Ablation checkpoints** (`best_ablation_*.pth` in `experiment_ablation/ablation_checkpoints/` in the original archive).  
   **Download:** `REPLACE_WITH_DROPBOX_LINK_CHECKPOINTS_ABLATION`

3. **Full dataset archives** (CIFAR-100 extract; Mini-ImageNet prepared tree).  
   **Download:** `REPLACE_WITH_DROPBOX_LINK_DATASETS`

After download, place files according to the **`README.md`** files in:

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

The supplementary package includes runnable source code, scripts, raw experimental tables, and implementation notes. Large checkpoints and full dataset archives are shared via **Dropbox links in this README** because of GitHub file-size limits.

---

## Contact

For reproduction questions, contact the corresponding author (add email here).

---

## Push to GitHub (after you create an empty repo)

```bash
cd "c:\Users\15366\Desktop\To Github\VDLF-Net-supplementary-upload"
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

Use a **Personal Access Token** (classic) as the password when GitHub prompts, or install [GitHub CLI](https://cli.github.com/) (`gh auth login`).

Then edit this `README.md` on GitHub or locally: replace the three `REPLACE_WITH_DROPBOX_LINK_*` placeholders with your Dropbox share links.
