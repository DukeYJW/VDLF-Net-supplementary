"""Mini-ImageNet 与变换：与 experiment_2/table2_few_shot_experiment.ipynb 对齐。"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_CACHE_VERSION = 2


def _resolve_image_paths(root: Path, cached_images: list) -> list[str]:
    """支持 v2（相对 root 的路径）与旧版绝对路径；若文件缺失则返回空列表以便触发重建。"""
    root = Path(root).resolve()
    out: list[str] = []
    for p in cached_images:
        path = Path(p)
        if not path.is_absolute():
            path = root / path
        else:
            path = path.resolve()
        out.append(str(path))
    sample = [x for x in out[:32] if x]
    if sample and not Path(sample[0]).is_file():
        return []
    return out


class MiniImageNetDataset(Dataset):
    def __init__(self, root: Path, split: str = "train", transform=None, cache_dir: Path | None = None):
        self.root = Path(root).resolve()
        self.split = split
        self.transform = transform
        self.images: list = []
        self.labels: list = []
        self.class_to_idx: dict = {}

        cache_root = Path(cache_dir) if cache_dir is not None else self.root / ".cache"
        cache_path = cache_root / f"{split}_index.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            ver = cached.get("_cache_version", 1)
            raw_images = cached["images"]
            if ver >= _CACHE_VERSION:
                resolved = [str(self.root / rel) for rel in raw_images]
            else:
                resolved = _resolve_image_paths(self.root, raw_images)
            if resolved and Path(resolved[0]).is_file():
                self.images = resolved
                self.labels = cached["labels"]
                self.class_to_idx = cached["class_to_idx"]
                return
            cache_path.unlink(missing_ok=True)

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"目录不存在: {split_dir}")

        rel_images: list[str] = []
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = split_dir / class_name
            if class_dir.is_dir():
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                class_idx = self.class_to_idx[class_name]
                for f in os.listdir(class_dir):
                    if f.lower().endswith(".jpg"):
                        rel = Path(class_dir / f).relative_to(self.root).as_posix()
                        rel_images.append(rel)
                        self.labels.append(class_idx)

        self.images = [str(self.root / r) for r in rel_images]

        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "_cache_version": _CACHE_VERSION,
                    "images": rel_images,
                    "labels": self.labels,
                    "class_to_idx": self.class_to_idx,
                },
                f,
                protocol=4,
            )

    def get_class_to_indices(self):
        d = defaultdict(list)
        for idx, label in enumerate(self.labels):
            d[label].append(idx)
        return dict(d)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(split: str = "train", image_size: int = 84):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.1)),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
