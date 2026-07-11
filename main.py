import os
import csv
import json
from datetime import datetime
from typing import Dict, List, Any
import time
import numpy as np

from dataset_sampling import (
    load_trainval_samples,
    save_subset_txt,
    fraction_to_tag,
)
from learning_curve_plot import plot_learning_curve
from run_kfold import run_kfold_training


def build_timestamped_save_root(base_name: str = "learning_curve") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


def save_json(data: Dict[str, Any], save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_kfold_summary_csv(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v is None:
                    parsed[k] = v
                    continue

                vv = v.strip()
                if vv.lower() in ["true", "false"]:
                    parsed[k] = vv.lower() == "true"
                    continue

                try:
                    if "." in vv:
                        parsed[k] = float(vv)
                    else:
                        parsed[k] = int(vv)
                except ValueError:
                    parsed[k] = vv
            rows.append(parsed)
    return rows


def read_test_summary_csv(csv_path: str) -> Dict[str, Dict[str, float]]:
    result = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row["metric"]
            result[metric] = {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }
    return result


def build_fraction_summary(
    fraction: float,
    num_samples: int,
    kfold_summary_csv: str,
    test_summary_csv: str = None
) -> Dict[str, Any]:
    kfold_rows = read_kfold_summary_csv(kfold_summary_csv)

    best_val_dice = [r["best_val_dice"] for r in kfold_rows]
    best_val_iou = [r["best_val_iou"] for r in kfold_rows]
    best_val_loss = [r["best_val_loss"] for r in kfold_rows]
    train_time_sec = [r["train_time_sec"] for r in kfold_rows]
    best_epoch = [r["best_epoch"] for r in kfold_rows]
    gpu_peak_memory_mb = [r.get("gpu_peak_memory_mb", 0.0) for r in kfold_rows]

    summary = {
        "fraction": fraction,
        "num_samples": num_samples,
        "val_dice_mean": float(np.mean(best_val_dice)),
        "val_dice_std": float(np.std(best_val_dice)),
        "val_iou_mean": float(np.mean(best_val_iou)),
        "val_iou_std": float(np.std(best_val_iou)),
        "val_loss_mean": float(np.mean(best_val_loss)),
        "val_loss_std": float(np.std(best_val_loss)),
        "train_time_mean_sec": float(np.mean(train_time_sec)),
        "train_time_std_sec": float(np.std(train_time_sec)),
        "best_epoch_mean": float(np.mean(best_epoch)),
        "best_epoch_std": float(np.std(best_epoch)),
        "gpu_peak_memory_mean_mb": float(np.mean(gpu_peak_memory_mb)),
        "gpu_peak_memory_std_mb": float(np.std(gpu_peak_memory_mb)),
    }

    if test_summary_csv is not None and os.path.exists(test_summary_csv):
        test_summary = read_test_summary_csv(test_summary_csv)
        for metric in [
            "dice",
            "iou",
            "precision",
            "recall",
            "accuracy",
            "test_loss",
            "normalized_surface_distance",
            "normalized_surface_dice",
        ]:
            if metric in test_summary:
                summary[f"test_{metric}_mean"] = test_summary[metric]["mean"]
                summary[f"test_{metric}_std"] = test_summary[metric]["std"]

    return summary


def save_learning_curve_summary(rows: List[Dict[str, Any]], save_path: str) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_by_count(samples, count: int, seed: int = 42, nested: bool = True):
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")

    total = len(samples)
    count = min(count, total)

    shuffled = samples.copy()
    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    shuffled = [samples[i] for i in indices]

    if nested:
        return shuffled[:count]

    rng2 = np.random.default_rng(seed + count)
    selected_idx = rng2.choice(total, size=count, replace=False)
    return [samples[i] for i in selected_idx]


def resolve_subset_samples(samples, subset_value, mode="fraction", seed=42, nested=True):
    total = len(samples)

    if mode == "fraction":
        fraction = float(subset_value)
        if not (0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0,1], got {fraction}")

        count = max(1, int(round(total * fraction)))
        subset = sample_by_count(samples, count=count, seed=seed, nested=nested)
        return subset, fraction, count

    if mode == "count":
        count = int(subset_value)
        if count <= 0:
            raise ValueError(f"count must be > 0, got {count}")

        count = min(count, total)
        fraction = count / total
        subset = sample_by_count(samples, count=count, seed=seed, nested=nested)
        return subset, fraction, count

    raise ValueError(f"Unsupported mode: {mode}")


def make_subset_tag(mode, subset_value, fraction, count):
    if mode == "fraction":
        return f"frac_{fraction_to_tag(float(subset_value))}"
    if mode == "count":
        return f"count_{count}"
    raise ValueError(f"Unsupported mode: {mode}")


def run_fraction_experiment(
    subset_value,
    subset_mode: str,
    all_samples,
    test_txt: str,
    base_config: Dict[str, Any],
    root_save_dir: str
) -> Dict[str, Any]:
    subset_samples, fraction, count = resolve_subset_samples(
        samples=all_samples,
        subset_value=subset_value,
        mode=subset_mode,
        seed=base_config["seed"],
        nested=True
    )

    subset_tag = make_subset_tag(subset_mode, subset_value, fraction, count)
    fraction_dir = os.path.join(root_save_dir, subset_tag)
    os.makedirs(fraction_dir, exist_ok=True)

    subset_txt_path = os.path.join(fraction_dir, f"trainval_subset_{subset_tag}.txt")
    save_subset_txt(subset_samples, subset_txt_path)

    kfold_save_root = os.path.join(fraction_dir, "kfold_run")
    os.makedirs(kfold_save_root, exist_ok=True)

    print(f"\n===== Learning Curve Subset: {subset_tag} =====")
    print(f"Fraction: {fraction:.4f} | Samples: {count}")
    print(f"Subset txt saved to: {subset_txt_path}")
    print(f"K-fold results will be saved to: {kfold_save_root}")

    run_kfold_training(
        all_trainval_txt=subset_txt_path,
        test_txt=test_txt,
        use_kfold=base_config["use_kfold"],
        n_splits=base_config["n_splits"],
        val_ratio=base_config["val_ratio"],
        seed=base_config["seed"],
        batch_size=base_config["batch_size"],
        target_size=tuple(base_config["target_size"]),
        num_classes=base_config["num_classes"],
        learning_rate=base_config["learning_rate"],
        num_workers=base_config["num_workers"],
        max_epochs=base_config["max_epochs"],
        patience=base_config["patience"],
        min_delta=base_config["min_delta"],
        binary_pos_weight=base_config["binary_pos_weight"],
        multiclass_weights=base_config["multiclass_weights"],
        save_root=kfold_save_root,
        use_amp=base_config["use_amp"],
        n_case_samples=base_config["n_case_samples"],
        ranking_metric=base_config["ranking_metric"],
        crop_padding_for_train_loss=base_config["crop_padding_for_train_loss"],
        loss_reduction_mode=base_config["loss_reduction_mode"],
        early_stop_monitor=base_config["early_stop_monitor"],
        scheduler_monitor=base_config["scheduler_monitor"],
        augment_train=base_config["augment_train"],
        normalize_mode=base_config["normalize_mode"],
        aug_prob=base_config["aug_prob"],
        hflip_prob=base_config["hflip_prob"],
        vflip_prob=base_config["vflip_prob"],
        rotation_degree=base_config["rotation_degree"],
        rotation_prob=base_config["rotation_prob"],
        use_color_jitter=base_config["use_color_jitter"],
        color_jitter_prob=base_config["color_jitter_prob"],
        brightness=base_config["brightness"],
        contrast=base_config["contrast"],
        saturation=base_config["saturation"],
        hue=base_config["hue"],
    )

    kfold_summary_csv = os.path.join(kfold_save_root, "kfold_summary.csv")
    test_summary_csv = os.path.join(kfold_save_root, "all_folds_test_summary.csv")

    summary = build_fraction_summary(
        fraction=fraction,
        num_samples=count,
        kfold_summary_csv=kfold_summary_csv,
        test_summary_csv=test_summary_csv if os.path.exists(test_summary_csv) else None
    )

    summary["subset_mode"] = subset_mode
    summary["subset_value"] = subset_value
    summary["subset_tag"] = subset_tag
    summary["fraction_dir"] = fraction_dir
    summary["subset_txt"] = subset_txt_path
    summary["kfold_summary_csv"] = kfold_summary_csv

    return summary


def run_learning_curve_experiment(
    trainval_txt: str,
    test_txt: str,
    subset_values: List[float],
    subset_sizes_mode: str,
    base_config: Dict[str, Any],
    save_root: str = None,
    plot_x_mode: str = "fraction"
) -> str:
    if save_root is None:
        save_root = build_timestamped_save_root("learning_curve")

    os.makedirs(save_root, exist_ok=True)

    all_samples = load_trainval_samples(trainval_txt)

    lc_config = {
        "trainval_txt": trainval_txt,
        "test_txt": test_txt,
        "subset_sizes_mode": subset_sizes_mode,
        "subset_values": subset_values,
        "base_config": base_config,
        "total_trainval_samples": len(all_samples),
        "plot_x_mode": plot_x_mode,
    }
    save_json(lc_config, os.path.join(save_root, "learning_curve_config.json"))

    summary_rows = []

    for subset_value in subset_values:
        summary = run_fraction_experiment(
            subset_value=subset_value,
            subset_mode=subset_sizes_mode,
            all_samples=all_samples,
            test_txt=test_txt,
            base_config=base_config,
            root_save_dir=save_root
        )
        summary_rows.append(summary)

    summary_csv_path = os.path.join(save_root, "learning_curve_summary.csv")
    save_learning_curve_summary(summary_rows, summary_csv_path)

    print(f"\nLearning curve summary saved to: {summary_csv_path}")

    plot_learning_curve(
        summary_csv=summary_csv_path,
        save_dir=save_root,
        x_mode=plot_x_mode
    )

    return save_root



def main():
    trainval_txt = "./train_val.txt"
    test_txt = "./test.txt"

    # "fraction" -> subset_values 填比例
    # "count"    -> subset_values 填實際筆數
    subset_sizes_mode = "count"

    # mode = "fraction" 時，例如：
    # subset_values = [0.25, 0.50, 0.75, 1.00]

    # mode = "count" 時，例如：
    subset_values = [438, 876, 1315, 1753, 3600]

    base_config = {
        "use_kfold": False,
        "n_splits": 2, # more than 2
        "val_ratio": 0.2,
        "seed": int(time.time()),
        "batch_size": 12,
        "target_size": [384, 384],   # JSON-friendly
        "num_classes": 1,
        "learning_rate": 1e-4,
        "num_workers": 0,
        "max_epochs": 200,
        "patience": 8,
        "min_delta": 1e-4,
        "binary_pos_weight": 3.0,
        "multiclass_weights": None,
        "use_amp": True,
        "n_case_samples": 15,
        "ranking_metric": "dice",
        "crop_padding_for_train_loss": True,
        "loss_reduction_mode": "sample_mean",
        "early_stop_monitor": "val_dice", # val_dice, val_loss, train_loss
        "scheduler_monitor": "val_dice", # val_dice, val_loss, train_loss

        # augmentation + normalization
        "normalize_mode": "fixed_05",
        "augment_train": True,
        "aug_prob": 0.6,
        "hflip_prob": 0.5,
        "vflip_prob": 0.0,
        "rotation_degree": 10,
        "rotation_prob": 0.5,
        "use_color_jitter": False,
        "color_jitter_prob": 0,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.02,
    }

    plot_x_mode = "num_samples" if subset_sizes_mode == "count" else "fraction"

    save_root = run_learning_curve_experiment(
        trainval_txt=trainval_txt,
        test_txt=test_txt,
        subset_values=subset_values,
        subset_sizes_mode=subset_sizes_mode,
        base_config=base_config,
        save_root=None,
        plot_x_mode=plot_x_mode
    )

    print(f"\nAll learning curve results saved under: {save_root}")


if __name__ == "__main__":
    main()