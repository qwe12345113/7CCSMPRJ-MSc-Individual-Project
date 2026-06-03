from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch


def read_samples_from_txt(txt_file):
    """
    Read samples from a txt file.
    Each line format:
        image_path mask_path
    """
    samples = []
    txt_file = Path(txt_file)

    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        image_path = os.path.normpath(parts[0].strip())
        mask_path = os.path.normpath(parts[1].strip())
        print("----------", image_path)

        if os.path.exists(image_path) and os.path.exists(mask_path):
            samples.append((image_path, mask_path))
        else:
            print(f"[Warning] File not found, skip: {image_path} | {mask_path}")

    if len(samples) == 0:
        raise ValueError(f"No valid samples found in {txt_file}")

    return samples


def resize_and_pad_keep_ratio(image, mask, target_size, image_fill=0, mask_fill=0):
    """
    Resize image and mask with aspect ratio preserved, then pad to target_size.

    Args:
        image: PIL.Image
        mask: PIL.Image
        target_size: (target_h, target_w)
        image_fill: padding fill value for image
        mask_fill: padding fill value for mask

    Returns:
        image, mask, meta
        meta = {
            "original_size": (orig_w, orig_h),
            "resized_size": (new_w, new_h),
            "target_size": (target_w, target_h),
            "padding": [left, top, right, bottom],
            "scale": scale
        }
    """
    target_h, target_w = target_size
    orig_w, orig_h = image.size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    image = TF.resize(image, [new_h, new_w], interpolation=Image.BILINEAR)
    mask = TF.resize(mask, [new_h, new_w], interpolation=Image.NEAREST)

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    if pad_w < 0 or pad_h < 0:
        raise ValueError(
            f"Resized image is larger than target size. "
            f"resized=({new_h}, {new_w}), target=({target_h}, {target_w})"
        )

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padding = [pad_left, pad_top, pad_right, pad_bottom]

    image = TF.pad(image, padding, fill=image_fill)
    mask = TF.pad(mask, padding, fill=mask_fill)

    meta = {
        "original_size": (orig_w, orig_h),
        "resized_size": (new_w, new_h),
        "target_size": (target_w, target_h),
        "padding": padding,
        "scale": scale
    }

    return image, mask, meta


class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples,
        target_size=(672, 928),
        num_classes=1
    ):
        """
        Args:
            samples: list of (image_path, mask_path)
            target_size: (H, W)
            num_classes: 1 for binary segmentation, >1 for multi-class
        """
        self.samples = samples
        self.target_size = target_size
        self.num_classes = num_classes
        self.image_transform = T.ToTensor()

        if len(self.samples) == 0:
            raise ValueError("No valid samples provided.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image, mask, meta = resize_and_pad_keep_ratio(
            image=image,
            mask=mask,
            target_size=self.target_size,
            image_fill=0,
            mask_fill=0
        )

        image = self.image_transform(image).clone()
        mask = torch.tensor(np.array(mask), dtype=torch.uint8)

        if self.num_classes == 1:
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = (mask > 0).float().unsqueeze(0)   # [1, H, W]
        else:
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = mask.long()   # [H, W]

        return {
            "image": image,
            "mask": mask,
            "image_path": image_path,
            "mask_path": mask_path,
            "original_size": torch.tensor(
                [meta["original_size"][0], meta["original_size"][1]],
                dtype=torch.int64
            ),  # (W, H)
            "resized_size": torch.tensor(
                [meta["resized_size"][0], meta["resized_size"][1]],
                dtype=torch.int64
            ),  # (W, H)
            "target_size": torch.tensor(
                [meta["target_size"][0], meta["target_size"][1]],
                dtype=torch.int64
            ),  # (W, H)
            "padding": torch.tensor(meta["padding"], dtype=torch.int64),  # [l, t, r, b]
            "scale": torch.tensor(meta["scale"], dtype=torch.float32),
        }


def get_segmentation_dataloader(
    samples,
    batch_size=4,
    target_size=(672, 928),
    num_classes=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True
):
    dataset = SegmentationDataset(
        samples=samples,
        target_size=target_size,
        num_classes=num_classes
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    return loader


def get_segmentation_dataloader_from_txt(
    txt_file,
    batch_size=4,
    target_size=(672, 928),
    num_classes=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True
):
    samples = read_samples_from_txt(txt_file)
    return get_segmentation_dataloader(
        samples=samples,
        batch_size=batch_size,
        target_size=target_size,
        num_classes=num_classes,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

