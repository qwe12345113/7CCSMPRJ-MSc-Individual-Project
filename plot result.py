import os
import csv
import glob
import re
import matplotlib.pyplot as plt


def natural_key(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]


def ordinal_label(n):
    """
    將數字轉成序數字串:
    1 -> 1st
    2 -> 2nd
    3 -> 3rd
    4 -> 4th
    ...
    """
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def load_test_result_csv(csv_path):
    result = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["metric"]] = float(row["value"])
    return result


def plot_multiple_test_results(csv_dir, save_path="all_test_results.png", csv_pattern="*.csv"):
    csv_files = glob.glob(os.path.join(csv_dir, csv_pattern))
    csv_files = sorted(csv_files, key=lambda x: natural_key(os.path.basename(x)))

    if not csv_files:
        raise ValueError("找不到任何 CSV 檔案")

    metrics = ["test_loss", "dice", "iou", "precision", "recall", "accuracy"]
    all_results = [load_test_result_csv(f) for f in csv_files]

    x = list(range(len(csv_files)))
    x_labels = [ordinal_label(i + 1) for i in x]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        y = [r.get(metric, None) for r in all_results]
        axes[i].plot(x, y, marker="o")
        axes[i].set_title(metric)
        axes[i].set_xlabel("Files")
        axes[i].set_ylabel(metric)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(x_labels)
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved to: {save_path}")


if __name__ == "__main__":
    csv_dir = r"./csv_result"
    save_path = os.path.join(csv_dir, "all_test_results.png")

    plot_multiple_test_results(
        csv_dir=csv_dir,
        save_path=save_path,
        csv_pattern="*.csv"
    )