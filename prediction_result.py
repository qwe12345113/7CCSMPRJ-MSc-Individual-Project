import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch


def _to_padding_list(padding_item):
    """
    Convert padding item to python list [left, top, right, bottom]
    Supports list / tuple / torch.Tensor
    """
    if isinstance(padding_item, torch.Tensor):
        return padding_item.detach().cpu().tolist()
    if isinstance(padding_item, np.ndarray):
        return padding_item.tolist()
    return list(padding_item)


def remove_padding_from_array(arr, padding):
    """
    Remove padding from numpy array.

    Args:
        arr:
            image -> [H, W, C]
            mask/pred -> [H, W]
        padding: [left, top, right, bottom]

    Returns:
        cropped array
    """
    left, top, right, bottom = _to_padding_list(padding)

    h, w = arr.shape[:2]

    x_start = left
    x_end = w - right if right > 0 else w
    y_start = top
    y_end = h - bottom if bottom > 0 else h

    return arr[y_start:y_end, x_start:x_end]


def remove_padding_from_tensor(tensor, padding):
    """
    Remove padding from tensor.

    Args:
        tensor:
            [H, W] or [C, H, W]
        padding: [left, top, right, bottom]

    Returns:
        cropped tensor
    """
    left, top, right, bottom = _to_padding_list(padding)

    if tensor.ndim == 2:
        h, w = tensor.shape
        x_start = left
        x_end = w - right if right > 0 else w
        y_start = top
        y_end = h - bottom if bottom > 0 else h
        return tensor[y_start:y_end, x_start:x_end]

    if tensor.ndim == 3:
        _, h, w = tensor.shape
        x_start = left
        x_end = w - right if right > 0 else w
        y_start = top
        y_end = h - bottom if bottom > 0 else h
        return tensor[:, y_start:y_end, x_start:x_end]

    raise ValueError(f"Unsupported tensor shape for remove_padding_from_tensor: {tensor.shape}")


def overlay_mask_on_image(image_np, mask_np, color="red", alpha=0.35):
    """
    image_np: [H, W, 3], range 0~1
    mask_np:  [H, W], binary mask
    """
    image_np = image_np.copy()
    overlay = image_np.copy()

    if color == "red":
        color_arr = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif color == "green":
        color_arr = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elif color == "blue":
        color_arr = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        color_arr = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    mask_bool = mask_np.astype(bool)
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color_arr
    return np.clip(overlay, 0.0, 1.0)


def save_overlay_visualization(image_np, gt_np, pred_np, save_path, title_text=None):
    """
    Save 4-panel visualization:
    1. image
    2. GT overlay
    3. prediction overlay
    4. TP/FP/FN map

    Note:
        This function assumes image_np / gt_np / pred_np are already de-padded.
    """
    gt_overlay = overlay_mask_on_image(image_np, gt_np, color="green", alpha=0.35)
    pred_overlay = overlay_mask_on_image(image_np, pred_np, color="red", alpha=0.35)

    compare = np.zeros_like(image_np)
    gt_bool = gt_np.astype(bool)
    pred_bool = pred_np.astype(bool)

    tp = gt_bool & pred_bool
    fp = (~gt_bool) & pred_bool
    fn = gt_bool & (~pred_bool)

    compare[tp] = np.array([0.0, 1.0, 0.0])   # TP
    compare[fp] = np.array([1.0, 0.0, 0.0])   # FP
    compare[fn] = np.array([0.0, 0.0, 1.0])   # FN

    plt.figure(figsize=(16, 4))
    if title_text is not None:
        plt.suptitle(title_text, fontsize=12)

    plt.subplot(1, 4, 1)
    plt.imshow(image_np)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gt_overlay)
    plt.title("GT Overlay")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred_overlay)
    plt.title("Pred Overlay")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(compare)
    plt.title("TP / FP / FN")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def binary_segmentation_metrics_from_cropped(preds, targets, eps=1e-6):
    """
    preds:   [H, W] or [1, H, W], binary 0/1 tensor
    targets: [H, W] or [1, H, W], binary 0/1 tensor
    """
    if preds.ndim == 3:
        preds = preds.squeeze(0)
    if targets.ndim == 3:
        targets = targets.squeeze(0)

    preds = preds.float()
    targets = targets.float()

    tp = (preds * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

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


@torch.no_grad()
def test_one_epoch_with_overlay(
    model,
    dataloader,
    criterion,
    device,
    save_dir="test_results",
    save_overlay_samples=0,
    use_amp=True
):
    """
    Test model on a dataloader.

    Important:
    - Metrics are computed only on the de-padded original region.
    - Saved visualizations are also de-padded before plotting.

    Expected batch format:
        batch["image"], batch["mask"], batch["padding"], batch["original_size"]
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    overlay_dir = os.path.join(save_dir, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    all_sample_rows = []
    saved_count = 0
    global_idx = 0

    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        paddings = batch["padding"]

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            outputs = model(images)

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        batch_size = images.size(0)

        for i in range(batch_size):
            padding = paddings[i]

            # crop prediction / target back to original size before metric calculation
            pred_crop = remove_padding_from_tensor(preds[i], padding)
            mask_crop = remove_padding_from_tensor(masks[i], padding)
            logit_crop = remove_padding_from_tensor(outputs[i], padding)

            # compute loss on cropped region only
            loss_i = criterion(
                logit_crop.unsqueeze(0),
                mask_crop.unsqueeze(0)
            )
            total_loss += loss_i.item()

            metrics = binary_segmentation_metrics_from_cropped(pred_crop, mask_crop)

            row = {
                "sample_id": global_idx,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
            }
            all_sample_rows.append(row)

            if saved_count < save_overlay_samples:
                image_np = images[i].detach().cpu().permute(1, 2, 0).numpy()
                gt_np = masks[i].detach().cpu().squeeze().numpy()
                pred_np = preds[i].detach().cpu().squeeze().numpy()

                # remove padding before visualization
                image_np = remove_padding_from_array(image_np, padding)
                gt_np = remove_padding_from_array(gt_np, padding)
                pred_np = remove_padding_from_array(pred_np, padding)

                save_overlay_visualization(
                    image_np=image_np,
                    gt_np=gt_np,
                    pred_np=pred_np,
                    save_path=os.path.join(overlay_dir, f"sample_{global_idx:04d}.png"),
                    title_text=(
                        f"sample={global_idx} | "
                        f"dice={row['dice']:.4f} | iou={row['iou']:.4f}"
                    )
                )
                saved_count += 1

            global_idx += 1

    num_samples = max(len(all_sample_rows), 1)
    avg_loss = total_loss / num_samples

    mean_results = {
        "test_loss": avg_loss,
        "dice": float(np.mean([r["dice"] for r in all_sample_rows])) if all_sample_rows else 0.0,
        "iou": float(np.mean([r["iou"] for r in all_sample_rows])) if all_sample_rows else 0.0,
        "precision": float(np.mean([r["precision"] for r in all_sample_rows])) if all_sample_rows else 0.0,
        "recall": float(np.mean([r["recall"] for r in all_sample_rows])) if all_sample_rows else 0.0,
        "accuracy": float(np.mean([r["accuracy"] for r in all_sample_rows])) if all_sample_rows else 0.0,
    }

    csv_path = os.path.join(save_dir, "per_sample_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "dice", "iou", "precision", "recall", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(all_sample_rows)

    return mean_results, all_sample_rows


def select_ranked_cases(all_sample_rows, metric="dice", n_per_group=5):
    """
    Select worst / median / best cases according to metric.
    """
    if len(all_sample_rows) == 0:
        return {"worst": [], "median": [], "best": []}

    sorted_rows = sorted(all_sample_rows, key=lambda x: x[metric])
    total = len(sorted_rows)
    n = min(n_per_group, total)

    worst_cases = sorted_rows[:n]
    best_cases = sorted_rows[-n:][::-1]

    center = total // 2
    start = max(0, center - n // 2)
    end = start + n
    if end > total:
        end = total
        start = max(0, end - n)

    median_cases = sorted_rows[start:end]

    return {
        "worst": worst_cases,
        "median": median_cases,
        "best": best_cases,
    }


@torch.no_grad()
def save_ranked_case_visualizations(
    model,
    dataset,
    case_groups,
    device,
    save_dir,
    use_amp=True
):
    """
    Re-run selected ranked cases and save de-padded overlay visualizations.

    Expected dataset item format:
        {
            "image": ...,
            "mask": ...,
            "padding": ...,
            ...
        }
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

            image_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
            gt_np = mask[0].detach().cpu().squeeze().numpy()
            pred_np = pred[0].detach().cpu().squeeze().numpy()

            # remove padding before visualization
            image_np = remove_padding_from_array(image_np, padding)
            gt_np = remove_padding_from_array(gt_np, padding)
            pred_np = remove_padding_from_array(pred_np, padding)

            title_text = (
                f"{group_name.upper()} | rank={rank} | sample={sample_id} | "
                f"dice={row['dice']:.4f} | iou={row['iou']:.4f} | "
                f"precision={row['precision']:.4f} | recall={row['recall']:.4f}"
            )

            filename = (
                f"{rank:02d}_sample_{sample_id:04d}"
                f"_dice_{row['dice']:.4f}_iou_{row['iou']:.4f}.png"
            )

            save_overlay_visualization(
                image_np=image_np,
                gt_np=gt_np,
                pred_np=pred_np,
                save_path=os.path.join(group_dir, filename),
                title_text=title_text
            )


def save_selected_cases_csv(case_groups, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "rank", "sample_id", "dice", "iou", "precision", "recall", "accuracy"])

        for group_name, rows in case_groups.items():
            for rank, row in enumerate(rows, start=1):
                writer.writerow([
                    group_name,
                    rank,
                    row["sample_id"],
                    row["dice"],
                    row["iou"],
                    row["precision"],
                    row["recall"],
                    row["accuracy"],
                ])


def plot_test_metrics(results, save_dir="test_results"):
    os.makedirs(save_dir, exist_ok=True)

    metric_names = ["dice", "iou", "precision", "recall", "accuracy"]
    metric_values = [results[m] for m in metric_names]

    plt.figure(figsize=(8, 6))
    plt.bar(metric_names, metric_values)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Test Metrics")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_metrics_bar.png"))
    plt.close()


def plot_test_summary(results, save_dir="test_results"):
    os.makedirs(save_dir, exist_ok=True)

    labels = ["Loss", "Dice", "IoU", "Precision", "Recall", "Accuracy"]
    values = [
        results["test_loss"],
        results["dice"],
        results["iou"],
        results["precision"],
        results["recall"],
        results["accuracy"]
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title("Test Summary")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_summary.png"))
    plt.close()