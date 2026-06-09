"""
eval/eval.py
============
在测试集（datasets/test）上评估训练好的 FER 模型。

【评测口径对齐·重要】FER2013 原始按 Usage 分三份，文件名带前缀：
    Training_* / PublicTest_* / PrivateTest_*
学界论文只在 **PrivateTest（3589 张）** 上报精度。本仓库 datasets/test/ 把
PublicTest+PrivateTest 混在一起（7178 张），直接报数无法与论文对标。
故本脚本支持 --split 按文件名前缀筛选：
    --split private  仅 PrivateTest（默认，可与论文对标）
    --split public   仅 PublicTest（相当于验证集口径）
    --split all      两者合并（旧口径，仅供参考）

输出：
- 总体 Accuracy 与 macro F1
- sklearn classification_report（各类 precision/recall/f1）
- 混淆矩阵（seaborn 热力图，保存为 png）

可选 --tta：Ten-Crop 测试增强（10 视角预测平均，等价小型集成，通常 +1~3%）。

从仓库根目录运行示例：
    python eval/eval.py --ckpt checkpoints/best.pth --split private
    python eval/eval.py --ckpt checkpoints/best.pth --split private --tta
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.cnn import build_model  # noqa: E402
from preprocess.preprocess import get_eval_transform, get_tta_transform  # noqa: E402

_PREFIX = {"public": "PublicTest", "private": "PrivateTest"}


def parse_args():
    p = argparse.ArgumentParser(description="评估面部表情识别模型")
    p.add_argument("--data-dir", default=os.path.join(_REPO_ROOT, "datasets"),
                   help="数据根目录（包含 test/ 子目录）")
    p.add_argument("--ckpt", default=os.path.join(_REPO_ROOT, "checkpoints", "best.pth"),
                   help="模型权重文件路径")
    p.add_argument("--split", default="private", choices=["all", "public", "private"],
                   help="按文件名前缀筛选评测子集（默认 private，与论文对标）")
    p.add_argument("--tta", action="store_true", help="启用 Ten-Crop 测试增强")
    p.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "checkpoints"),
                   help="混淆矩阵等输出目录")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def filter_by_split(dataset: ImageFolder, split: str) -> Subset:
    """按文件名前缀（PublicTest/PrivateTest）筛出子集。"""
    if split == "all":
        return Subset(dataset, list(range(len(dataset))))
    prefix = _PREFIX[split]
    idxs = [i for i, (path, _) in enumerate(dataset.samples)
            if os.path.basename(path).startswith(prefix)]
    if not idxs:
        raise RuntimeError(f"未找到前缀为 {prefix}_ 的样本，请检查 --split 与数据集。")
    return Subset(dataset, idxs)


@torch.no_grad()
def main():
    args = parse_args()
    from sklearn.metrics import (accuracy_score, f1_score,
                                 classification_report, confusion_matrix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---- 加载 checkpoint（含类别映射与骨干类型）----
    ckpt = torch.load(args.ckpt, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    num_classes = ckpt.get("num_classes", len(class_to_idx))
    arch = ckpt.get("arch", "cnn")
    idx_to_class = [None] * num_classes
    for cls, idx in class_to_idx.items():
        idx_to_class[idx] = cls

    model = build_model(num_classes, arch=arch, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"骨干: {arch} | 评测子集: {args.split} | TTA: {args.tta}")

    # ---- 测试集（按 split 过滤）----
    test_dir = os.path.join(args.data_dir, "test")
    transform = get_tta_transform() if args.tta else get_eval_transform()
    full_ds = ImageFolder(test_dir, transform=transform)
    assert full_ds.class_to_idx == class_to_idx, \
        f"测试集类别映射与 checkpoint 不一致:\n{full_ds.class_to_idx}\n{class_to_idx}"
    test_ds = filter_by_split(full_ds, args.split)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"测试样本: {len(test_ds)}")

    # ---- 推理 ----
    all_preds, all_labels = [], []
    for imgs, labels in test_loader:
        if args.tta:
            # imgs: [B, 10, 1, H, W] -> 合批推理后按 10 个 crop 求 softmax 平均
            b, ncrop = imgs.shape[0], imgs.shape[1]
            imgs = imgs.view(b * ncrop, *imgs.shape[2:]).to(device)
            probs = torch.softmax(model(imgs), dim=1).view(b, ncrop, -1).mean(1)
            preds = probs.argmax(1).cpu().numpy()
        else:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # ---- 指标 ----
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n== 测试集结果（split={args.split}{', TTA' if args.tta else ''}）==")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=idx_to_class, digits=4))

    # ---- 混淆矩阵热力图 ----
    cm = confusion_matrix(y_true, y_pred)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=idx_to_class, yticklabels=idx_to_class)
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title(f"Confusion Matrix [{args.split}] acc={acc:.3f}, macroF1={macro_f1:.3f}")
        plt.tight_layout()
        os.makedirs(args.out_dir, exist_ok=True)
        cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=120)
        print(f"混淆矩阵已保存: {cm_path}")
    except Exception as e:
        print(f"绘制混淆矩阵失败（不影响指标）: {e}")
        print(f"混淆矩阵(原始):\n{cm}")


if __name__ == "__main__":
    main()
