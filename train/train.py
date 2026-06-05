"""
train/train.py
==============
训练 FER CNN 的脚本，支持命令行参数。

要点：
- 数据：datasets/train/<class>/*.jpg（torchvision ImageFolder 格式）。
- 训练集在线变换含数据增强；从训练集切出验证集（验证集用无增强变换）。
- 类别不平衡：用各类样本数计算 class weights，传入 CrossEntropyLoss。
- 优化器 Adam + StepLR；早停（监控验证损失）。
- 自动选择设备（GPU 可用则用 GPU），本地与 Colab 通用。
- 最优权重连同 class_to_idx 一并保存，供 eval/demo 复用。

从仓库根目录运行示例：
    python train/train.py --epochs 50 --lr 1e-3 --batch-size 64
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder

# 把仓库根目录加入 sys.path，使脚本可从根目录直接运行并 import 其它模块
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.cnn import build_model  # noqa: E402
from preprocess.preprocess import get_train_transform, get_eval_transform  # noqa: E402


class ApplyTransform(Dataset):
    """给 random_split 出来的子集套上指定 transform。"""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]   # img 为 PIL 图（base 数据集未设 transform）
        return self.transform(img), label


def parse_args():
    p = argparse.ArgumentParser(description="训练面部表情识别 CNN")
    p.add_argument("--data-dir", default=os.path.join(_REPO_ROOT, "datasets"),
                   help="数据根目录（包含 train/ 子目录）")
    p.add_argument("--epochs", type=int, default=50, help="训练轮数")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--batch-size", type=int, default=64, help="批大小")
    p.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    p.add_argument("--patience", type=int, default=10, help="早停耐心值（轮）")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader 进程数")
    p.add_argument("--out", default=os.path.join(_REPO_ROOT, "checkpoints", "best.pth"),
                   help="最优权重保存路径")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()


def compute_class_weights(targets, num_classes):
    """逆频率类别权重：样本少的类权重大，缓解不平衡。"""
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # 防止除零
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """跑一个 epoch；optimizer 为 None 时表示评估（不更新参数）。"""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, total_correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---- 数据集：base 不设 transform（返回 PIL），切分后再分别套变换 ----
    train_dir = os.path.join(args.data_dir, "train")
    base = ImageFolder(train_dir)            # 无 transform
    class_to_idx = base.class_to_idx
    num_classes = len(class_to_idx)
    print(f"类别 ({num_classes}): {class_to_idx}")

    n_val = int(len(base) * args.val_split)
    n_train = len(base) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(base, [n_train, n_val], generator=g)

    train_ds = ApplyTransform(train_subset, get_train_transform())
    val_ds = ApplyTransform(val_subset, get_eval_transform())
    print(f"训练样本: {len(train_ds)}  验证样本: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ---- 模型 / 损失（带类别权重）/ 优化器 / 调度器 ----
    model = build_model(num_classes).to(device)
    class_weights = compute_class_weights(base.targets, num_classes).to(device)
    print(f"类别权重: {class_weights.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ---- 训练循环 + 早停 ----
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"[{epoch:02d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # 早停：监控验证损失
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            epochs_no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "num_classes": num_classes,
                "args": vars(args),
            }, args.out)
            print(f"   ↳ 验证损失改善，已保存最优权重到 {args.out}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"验证损失连续 {args.patience} 轮未改善，触发早停。")
                break

    # ---- 训练曲线 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history["train_loss"], label="train")
        ax1.plot(history["val_loss"], label="val")
        ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
        ax2.plot(history["train_acc"], label="train")
        ax2.plot(history["val_acc"], label="val")
        ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
        plt.tight_layout()
        curve_path = os.path.join(os.path.dirname(args.out), "train_curves.png")
        plt.savefig(curve_path, dpi=120)
        print(f"训练曲线已保存: {curve_path}")
    except Exception as e:
        print(f"绘制训练曲线失败（不影响训练结果）: {e}")

    print(f"训练完成。最优验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
