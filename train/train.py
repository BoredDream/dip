"""
train/train.py
==============
训练 FER 模型的脚本，支持命令行参数。

要点（每个设计都可解释）：
- 数据：datasets/train/<class>/*.jpg（torchvision ImageFolder 格式）。
- 训练集在线变换含数据增强；从训练集切出验证集（验证集用无增强变换）。
- 类别不平衡：用各类样本数计算 class weights（支持 inv / sqrt 两种力度），传入损失。
- 损失：CrossEntropyLoss + Label Smoothing（缓解 FER2013 标签噪声、抑制过度自信）。
- 优化器 AdamW（带权重衰减的 Adam，正则更干净）。
- 调度器：线性 warmup + 余弦退火（前几轮平滑升温，之后平滑降到 0，收敛更好）。
- 早停 / 存档按 **验证准确率（val_acc）**，并同时另存"按 val_loss 最优"的权重。
- 最优权重连同 class_to_idx、arch 一并保存，供 eval/demo 复用。

从仓库根目录运行示例：
    python train/train.py --arch cnn      --epochs 60 --lr 1e-3
    python train/train.py --arch resnet18 --epochs 40 --lr 5e-4
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
    p = argparse.ArgumentParser(description="训练面部表情识别模型")
    p.add_argument("--data-dir", default=os.path.join(_REPO_ROOT, "datasets"),
                   help="数据根目录（包含 train/ 子目录）")
    p.add_argument("--arch", default="cnn", choices=["cnn", "resnet18"],
                   help="骨干网络：cnn(轻量自定义) / resnet18(预训练微调)")
    p.add_argument("--epochs", type=int, default=60, help="训练轮数")
    p.add_argument("--lr", type=float, default=1e-3, help="峰值学习率")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减(AdamW)")
    p.add_argument("--warmup-epochs", type=int, default=3, help="线性 warmup 轮数")
    p.add_argument("--batch-size", type=int, default=64, help="批大小")
    p.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    p.add_argument("--patience", type=int, default=12, help="早停耐心值（轮）")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="标签平滑系数（0 关闭），缓解标签噪声")
    p.add_argument("--weight-scheme", default="sqrt",
                   choices=["none", "inv", "sqrt"],
                   help="类别权重力度：none/inv(逆频率)/sqrt(逆频率开方,更温和)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader 进程数（Windows 出错可设 0）")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="计算设备；GPU 架构不兼容时可强制 cpu")
    p.add_argument("--out", default=os.path.join(_REPO_ROOT, "checkpoints", "best.pth"),
                   help="最优权重保存路径（按 val_acc）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()


def compute_class_weights(targets, num_classes, scheme="sqrt"):
    """
    类别权重，缓解不平衡。可解释的两档力度：
    - inv ：逆频率 1/f，对极端少数类（disgust ~1.5%）权重很大，易"过度矫正"。
    - sqrt：逆频率开方，力度更温和，避免把 disgust 从"识别不出"推成"过度预测"。
    """
    if scheme == "none":
        return None
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # 防止除零
    inv = counts.sum() / (num_classes * counts)
    weights = np.sqrt(inv) if scheme == "sqrt" else inv
    # 归一化到均值 1，使整体损失尺度与无权重时可比
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_scheduler(optimizer, args, steps_per_epoch):
    """线性 warmup -> 余弦退火（按 step 更新，曲线更平滑）。"""
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:                      # 线性升温
            return step / warmup_steps
        # 余弦退火：从 1 平滑降到 0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
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
                if scheduler is not None:
                    scheduler.step()      # 按 step 更新学习率

            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, total_correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device} | 骨干: {args.arch}")

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

    # ---- 模型 / 损失 / 优化器 / 调度器 ----
    model = build_model(num_classes, arch=args.arch).to(device)
    class_weights = compute_class_weights(base.targets, num_classes, args.weight_scheme)
    if class_weights is not None:
        print(f"类别权重({args.weight_scheme}): {class_weights.numpy().round(3)}")
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch=len(train_loader))

    # ---- 训练循环 + 早停（按 val_acc）----
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_loss_path = os.path.join(os.path.dirname(args.out), "best_loss.pth")

    def save_ckpt(path):
        torch.save({
            "state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "num_classes": num_classes,
            "arch": args.arch,
            "args": vars(args),
        }, path)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer, scheduler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"[{epoch:02d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # 另存"按 val_loss 最优"的权重（仅作对照）
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            save_ckpt(best_loss_path)

        # 主存档与早停：按 val_acc（更贴近分类目标）
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            epochs_no_improve = 0
            save_ckpt(args.out)
            print(f"   ↳ 验证准确率改善 -> {best_val_acc:.4f}，已保存最优权重到 {args.out}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"验证准确率连续 {args.patience} 轮未改善，触发早停。")
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

    print(f"训练完成。最优验证准确率: {best_val_acc:.4f}（best_loss 权重另存于 {best_loss_path}）")


if __name__ == "__main__":
    main()
