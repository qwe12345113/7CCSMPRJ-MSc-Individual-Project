import os
import random
import csv


def split_dataset(
    image_dir,
    mask_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    image_exts=None,
    mask_exts=None,
    seed=42,
    save_txt=True,
    save_csv=True,
    colab = True
):
    """
    將影像與對應 mask 配對後，隨機切分成 train/val/test，
    並將路徑寫入 txt 或 csv 檔案。

    參數:
        image_dir (str): 影像資料夾
        mask_dir (str): mask 資料夾
        output_dir (str): 輸出檔案資料夾
        train_ratio (float): 訓練集比例
        val_ratio (float): 驗證集比例
        test_ratio (float): 測試集比例
        image_exts (set): 支援的影像副檔名
        mask_exts (set): 支援的 mask 副檔名
        seed (int): 隨機種子
        save_txt (bool): 是否輸出 txt
        save_csv (bool): 是否輸出 csv
    """

    if image_exts is None:
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if mask_exts is None:
        mask_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必須等於 1.0")

    os.makedirs(output_dir, exist_ok=True)

    # 建立 mask 主檔名 -> 完整路徑 對照表
    mask_dict = {}
    for fname in os.listdir(mask_dir):
        fpath = os.path.join(mask_dir, fname)

        if not os.path.isfile(fpath):
            continue

        base, ext = os.path.splitext(fname)
        if ext.lower() in mask_exts:
            mask_dict[base] = fpath

    # 配對 image 和 mask
    pairs = []
    for fname in os.listdir(image_dir):
        # fpath = os.path.join("/kaggle/input/datasets/hsiangyincheng/my dataset images V2", fname).replace("\\", "/")
        fpath = os.path.join(image_dir, fname).replace("\\", "/")
        if not os.path.isfile(fpath):
            continue

        base, ext = os.path.splitext(fname)
        if ext.lower() not in image_exts:
            continue

        if base+"_mask" in mask_dict:
            print(base)
            # mpath = mask_dict[base + "_mask"].replace("\\", "/")
            fpath = fpath.replace(image_dir, "/kaggle/input/datasets/hsiangyincheng/my-dataset-images")
            mpath = mask_dict[base+"_mask"].replace("\\", "/").replace(mask_dir, "/kaggle/input/datasets/hsiangyincheng/my-dataset-mask")
            pairs.append((fpath, mpath))
        else:
            # print(fpath, mask_dict[base+"_mask"])
            print(f"找不到對應 mask，已忽略: {fpath}")

    if len(pairs) == 0:
        raise ValueError("沒有找到任何可配對的 image-mask 資料")

    # 打亂資料
    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    train_val_pairs = train_pairs + val_pairs

    print(f"總配對數量: {total}")
    print(f"Training:   {len(train_pairs)}")
    print(f"Validation: {len(val_pairs)}")
    print(f"Testing:    {len(test_pairs)}")

    # 寫 txt
    def write_txt(file_path, data_pairs):
        with open(file_path, "w", encoding="utf-8") as f:
            for img_path, mask_path in data_pairs:
                f.write(f"{img_path} {mask_path}\n")

    # 寫 csv
    def write_csv(file_path, data_pairs):
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "mask_path"])
            writer.writerows(data_pairs)

    if save_txt:
        # write_txt(os.path.join(output_dir, "train.txt"), train_pairs)
        # write_txt(os.path.join(output_dir, "val.txt"), val_pairs)
        write_txt(os.path.join(output_dir, "train_val.txt"), train_val_pairs)
        write_txt(os.path.join(output_dir, "test.txt"), test_pairs)


    if save_csv:
        write_csv(os.path.join(output_dir, "train.csv"), train_pairs)
        write_csv(os.path.join(output_dir, "val.csv"), val_pairs)
        write_csv(os.path.join(output_dir, "test.csv"), test_pairs)

    print(f"\n已輸出到: {output_dir}")


if __name__ == "__main__":
    image_dir = "../dataset/my_dataset_images"     # 改成你的影像資料夾
    mask_dir = "../dataset/my_dataset_mask"       # 改成你的 mask 資料夾
    output_dir = "./"    # 改成你的輸出資料夾

    split_dataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        train_ratio=0.75,
        val_ratio=0.15,
        test_ratio=0.1,
        seed=42,
        save_txt=True,
        save_csv=False,
        colab=0
    )