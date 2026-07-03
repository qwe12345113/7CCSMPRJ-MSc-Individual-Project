import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch


def remove_padding_from_tensor(tensor, padding):
    """
    Remove padding from a tensor.

    Args:
        tensor:
            [C, H, W] or [H, W]
        padding:
            [left, top, right, bottom]
            can be list / tuple / torch.Tensor

    Returns:
        Cropped tensor
    """
    if isinstance(padding, torch.Tensor):
        padding = padding.tolist()

    left, top, right, bottom = [int(x) for x in padding]

    if tensor.ndim == 3:
        _, h, w = tensor.shape
        x_start = left
        x_end = w - right
        y_start = top
        y_end = h - bottom
        return tensor[:, y_start:y_end, x_start:x_end]

    if tensor.ndim == 2:
        h, w = tensor.shape
        x_start = left
        x_end = w - right
        y_start = top
        y_end = h - bottom
        return tensor[y_start:y_end, x_start:x_end]

    raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def remove_padding_from_array(arr, padding):
    """
    Remove padding from numpy array.

    Args:
        arr:
            [H, W, C] or [H, W]
        padding:
            [left, top, right, bottom]

    Returns:
        Cropped numpy array
    """
    if isinstance(padding, torch.Tensor):
        padding = padding.tolist()

    left, top, right, bottom = [int(x) for x in padding]

    if arr.ndim == 3:
        h, w, _ = arr.shape
        x_start = left
        x_end = w - right
        y_start = top
        y_end = h - bottom
        return arr[y_start:y_end, x_start:x_end, :]

    if arr.ndim == 2:
        h, w = arr.shape
        x_start = left
        x_end = w - right
        y_start = top
        y_end = h - bottom
        return arr[y_start:y_end, x_start:x_end]

    raise ValueError(f"Unsupported array shape: {arr.shape}")


def denormalize_image_tensor(image_tensor, normalize_mode="none"):
    """
    Denormalize image tensor for visualization.

    Args:
        image_tensor: [3, H, W]
        normalize_mode: "none" or "fixed_05"

    Returns:
        [3, H, W] tensor in display-friendly range
    """
    if normalize_mode is None or normalize_mode == "none":
        return image_tensor

    if normalize_mode == "fixed_05":
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
        return image_tensor * std + mean

    raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")


def binary_segmentation_metrics_from_cropped(pred_mask, gt_mask, eps=1e-7):
    """
    Compute binary segmentation metrics on already-cropped tensors.

    Args:
        pred_mask: [H, W] or [1, H, W], binary {0,1}
        gt_mask:   [H, W] or [1, H, W], binary {0,1}

    Returns:
        dict with dice, iou, precision, recall, accuracy
    """
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.squeeze(0)

    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()

    pred_flat = pred_mask.reshape(-1)
    gt_flat = gt_mask.reshape(-1)

    tp = torch.sum((pred_flat == 1) & (gt_flat == 1)).float()
    tn = torch.sum((pred_flat == 0) & (gt_flat == 0)).float()
    fp = torch.sum((pred_flat == 1) & (gt_flat == 0)).float()
    fn = torch.sum((pred_flat == 0) & (gt_flat == 1)).float()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "accuracy": float(accuracy.item()),
    }


def save_overlay_visualization(image_np, gt_np, pred_np, save_path, title_text=""):
    """
    Save visualization with:
    - original image
    - GT mask
    - prediction mask
    - overlay comparison
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    image_np = np.clip(image_np, 0, 1)

    gt_vis = gt_np.astype(np.float32)
    pred_vis = pred_np.astype(np.float32)

    overlay = image_np.copy()

    gt_region = gt_vis > 0
    overlay[gt_region, 1] = np.maximum(overlay[gt_region, 1], 1.0)

    pred_region = pred_vis > 0
    overlay[pred_region, 0] = np.maximum(overlay[pred_region, 0], 1.0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt_vis, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_vis, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (GT=Green, Pred=Red)")
    axes[3].axis("off")

    if title_text:
        fig.suptitle(title_text, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_per_sample_metrics_csv(rows, save_path):
    if len(rows) == 0:
        return

    fieldnames = [
        "sample_id",
        "image_name",
        "mask_name",
        "image_path",
        "mask_path",
        "dice",
        "iou",
        "precision",
        "recall",
        "accuracy",
    ]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


@torch.no_grad()
def test_one_epoch_with_overlay(
    model,
    dataloader,
    criterion,
    device,
    save_dir,
    save_overlay_samples=10,
    use_amp=True,
    normalize_mode="none"
):
    """
    Test model on dataloader.
    - metrics computed after removing padding
    - saved overlay images also remove padding
    - visualization image is automatically denormalized

    Returns:
        fold_test_result: dict
        all_sample_rows: list[dict]
    """
    os.makedirs(save_dir, exist_ok=True)
    overlay_dir = os.path.join(save_dir, "overlay_samples")
    os.makedirs(overlay_dir, exist_ok=True)

    model.eval()

    total_loss = 0.0
    sample_count = 0
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

    all_sample_rows = []
    global_idx = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        paddings = batch["padding"]

        image_paths = batch["image_path"]
        mask_paths = batch["mask_path"]

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            outputs = model(images)

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        batch_size = images.size(0)

        for i in range(batch_size):
            padding = paddings[i]

            logit_crop = remove_padding_from_tensor(outputs[i], padding)
            mask_crop = remove_padding_from_tensor(masks[i], padding)
            pred_crop = remove_padding_from_tensor(preds[i], padding)

            loss_i = criterion(
                logit_crop.unsqueeze(0),
                mask_crop.unsqueeze(0)
            )
            total_loss += loss_i.item()

            metrics = binary_segmentation_metrics_from_cropped(pred_crop, mask_crop)

            dice_scores.append(metrics["dice"])
            iou_scores.append(metrics["iou"])
            precision_scores.append(metrics["precision"])
            recall_scores.append(metrics["recall"])
            accuracy_scores.append(metrics["accuracy"])

            image_path = image_paths[i]
            mask_path = mask_paths[i]

            row = {
                "sample_id": global_idx,
                "image_path": image_path,
                "mask_path": mask_path,
                "image_name": os.path.basename(image_path),
                "mask_name": os.path.basename(mask_path),
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
            }
            all_sample_rows.append(row)

            if global_idx < save_overlay_samples:
                image_vis = images[i].detach().cpu()
                image_vis = denormalize_image_tensor(image_vis, normalize_mode=normalize_mode)
                image_np = image_vis.permute(1, 2, 0).numpy()

                gt_np = masks[i].detach().cpu().squeeze().numpy()
                pred_np = preds[i].detach().cpu().squeeze().numpy()

                image_np = remove_padding_from_array(image_np, padding)
                gt_np = remove_padding_from_array(gt_np, padding)
                pred_np = remove_padding_from_array(pred_np, padding)

                title_text = (
                    f"sample={global_idx} | image={row['image_name']} | "
                    f"dice={row['dice']:.4f} | iou={row['iou']:.4f} | "
                    f"precision={row['precision']:.4f} | recall={row['recall']:.4f}"
                )

                stem = os.path.splitext(row["image_name"])[0]
                filename = (
                    f"{global_idx:04d}_{stem}"
                    f"_dice_{row['dice']:.4f}_iou_{row['iou']:.4f}.png"
                )

                save_overlay_visualization(
                    image_np=image_np,
                    gt_np=gt_np,
                    pred_np=pred_np,
                    save_path=os.path.join(overlay_dir, filename),
                    title_text=title_text
                )

            sample_count += 1
            global_idx += 1

    avg_loss = total_loss / max(sample_count, 1)

    fold_test_result = {
        "test_loss": float(avg_loss),
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "precision": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "accuracy": float(np.mean(accuracy_scores)) if accuracy_scores else 0.0,
    }

    save_per_sample_metrics_csv(
        all_sample_rows,
        os.path.join(save_dir, "per_sample_metrics.csv")
    )

    return fold_test_result, all_sample_rows


def plot_test_metrics(test_result, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    metric_names = ["dice", "iou", "precision", "recall", "accuracy"]
    metric_values = [test_result[m] for m in metric_names]

    plt.figure(figsize=(8, 5))
    plt.bar(metric_names, metric_values)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Test Metrics")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_metrics.png"), dpi=150)
    plt.close()


def plot_test_summary(test_result, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    lines = [
        f"test_loss : {test_result['test_loss']:.6f}",
        f"dice      : {test_result['dice']:.6f}",
        f"iou       : {test_result['iou']:.6f}",
        f"precision : {test_result['precision']:.6f}",
        f"recall    : {test_result['recall']:.6f}",
        f"accuracy  : {test_result['accuracy']:.6f}",
    ]

    plt.figure(figsize=(7, 4))
    plt.axis("off")
    plt.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_summary.png"), dpi=150)
    plt.close()


def select_ranked_cases(all_sample_rows, metric="dice", n_per_group=5):
    """
    Select worst / median / best samples according to a metric.
    """
    if len(all_sample_rows) == 0:
        return {"worst": [], "median": [], "best": []}

    if metric not in all_sample_rows[0]:
        raise ValueError(f"Metric '{metric}' not found in sample rows")

    sorted_rows = sorted(all_sample_rows, key=lambda x: x[metric])

    n_total = len(sorted_rows)
    n = min(n_per_group, n_total)

    worst = sorted_rows[:n]
    best = sorted_rows[-n:][::-1]

    mid = n_total // 2
    half = n // 2
    start = max(0, mid - half)
    end = min(n_total, start + n)
    median = sorted_rows[start:end]

    return {
        "worst": worst,
        "median": median,
        "best": best
    }


def save_selected_cases_csv(case_groups, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "group",
            "rank",
            "sample_id",
            "image_name",
            "mask_name",
            "image_path",
            "mask_path",
            "dice",
            "iou",
            "precision",
            "recall",
            "accuracy"
        ])

        for group_name, rows in case_groups.items():
            for rank, row in enumerate(rows, start=1):
                writer.writerow([
                    group_name,
                    rank,
                    row["sample_id"],
                    row.get("image_name", ""),
                    row.get("mask_name", ""),
                    row.get("image_path", ""),
                    row.get("mask_path", ""),
                    row["dice"],
                    row["iou"],
                    row["precision"],
                    row["recall"],
                    row["accuracy"],
                ])


def save_ranked_case_visualizations(
    model,
    dataset,
    case_groups,
    device,
    save_dir,
    use_amp=True,
    normalize_mode="none"
):
    """
    Save overlay visualizations for selected worst / median / best cases.
    Filename and title include original image name.
    Visualization image is automatically denormalized.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for group_name, rows in case_groups.items():
        group_dir = os.path.join(save_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        for rank, row in enumerate(rows, start=1):
            sample_id = row["sample_id"]

            sample = dataset[sample_id]
            image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
            mask = sample["mask"].unsqueeze(0).to(device, non_blocking=True)
            padding = sample["padding"]

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits = model(image)

            pred = (torch.sigmoid(logits) > 0.5).float()

            image_vis = image[0].detach().cpu()
            image_vis = denormalize_image_tensor(image_vis, normalize_mode=normalize_mode)
            image_np = image_vis.permute(1, 2, 0).numpy()

            gt_np = mask[0].detach().cpu().squeeze().numpy()
            pred_np = pred[0].detach().cpu().squeeze().numpy()

            image_np = remove_padding_from_array(image_np, padding)
            gt_np = remove_padding_from_array(gt_np, padding)
            pred_np = remove_padding_from_array(pred_np, padding)

            title_text = (
                f"{group_name.upper()} | rank={rank} | sample={sample_id} | "
                f"image={row.get('image_name', 'N/A')} | "
                f"dice={row['dice']:.4f} | iou={row['iou']:.4f} | "
                f"precision={row['precision']:.4f} | recall={row['recall']:.4f}"
            )

            orig_name = row.get("image_name", f"sample_{sample_id:04d}")
            orig_stem = os.path.splitext(orig_name)[0]

            filename = (
                f"{rank:02d}_{orig_stem}"
                f"_sample_{sample_id:04d}"
                f"_dice_{row['dice']:.4f}_iou_{row['iou']:.4f}.png"
            )

            save_overlay_visualization(
                image_np=image_np,
                gt_np=gt_np,
                pred_np=pred_np,
                save_path=os.path.join(group_dir, filename),
                title_text=title_text
            )