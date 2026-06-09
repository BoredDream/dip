"""
eval/visualize_cam.py
=====================
用 Grad-CAM 可视化模型"看脸上哪里"做表情判断，生成一张网格图。

每一类各取若干张测试样本，画出 [原图 | Grad-CAM 热力叠加图] 并标注
预测类别与置信度（预测对错用绿/红区分）。输出 checkpoints/gradcam_demo.png。

这是把"黑箱分类"变成"可解释证据"的展示脚本，适合放进报告/答辩：
- happy 应主要激活嘴角；surprise 多在张大的嘴与睁大的眼；angry 常在眉间。
- 若热力图落在背景/头发等无关区域，说明模型学到了伪相关，是很有价值的调试信号。

从仓库根目录运行示例：
    python eval/visualize_cam.py --ckpt checkpoints/best.pth --per-class 3
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.cnn import build_model  # noqa: E402
from model.gradcam import GradCAM, overlay_cam  # noqa: E402
from preprocess.preprocess import get_eval_transform  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM 可解释性可视化")
    p.add_argument("--data-dir", default=os.path.join(_REPO_ROOT, "datasets"))
    p.add_argument("--ckpt", default=os.path.join(_REPO_ROOT, "checkpoints", "best.pth"))
    p.add_argument("--split", default="private", choices=["all", "public", "private"],
                   help="取样的测试子集（前缀筛选）")
    p.add_argument("--per-class", type=int, default=3, help="每类可视化的样本数")
    p.add_argument("--out", default=os.path.join(_REPO_ROOT, "checkpoints", "gradcam_demo.png"))
    return p.parse_args()


_PREFIX = {"public": "PublicTest", "private": "PrivateTest", "all": ""}


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    num_classes = ckpt.get("num_classes", len(class_to_idx))
    arch = ckpt.get("arch", "cnn")
    idx_to_class = [None] * num_classes
    for cls, idx in class_to_idx.items():
        idx_to_class[idx] = cls

    model = build_model(num_classes, arch=arch, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    cam_engine = GradCAM(model, model.gradcam_target_layer())

    transform = get_eval_transform()
    test_dir = os.path.join(args.data_dir, "test")
    prefix = _PREFIX[args.split]

    # 收集每类的样本路径
    rows = []   # (cls_name, img_path)
    for cls in sorted(class_to_idx, key=lambda c: class_to_idx[c]):
        cls_dir = os.path.join(test_dir, cls)
        files = sorted(f for f in os.listdir(cls_dir)
                       if f.lower().endswith((".jpg", ".png", ".jpeg"))
                       and f.startswith(prefix))
        for f in files[:args.per_class]:
            rows.append((cls, os.path.join(cls_dir, f)))

    ncols = 2 * args.per_class
    nrows = num_classes
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.8))
    axes = np.atleast_2d(axes)

    per = {c: 0 for c in class_to_idx}
    for cls, path in rows:
        col0 = per[cls] * 2
        r = class_to_idx[cls]

        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        from PIL import Image
        x = transform(Image.fromarray(gray)).unsqueeze(0).to(device)
        x.requires_grad_(True)

        cam, pred_idx, conf = cam_engine(x, class_idx=None)
        gray48 = cv2.resize(gray, (48, 48))
        overlay = overlay_cam(gray48, cam)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        correct = (pred_idx == class_to_idx[cls])
        color = "green" if correct else "red"

        ax0 = axes[r, col0]
        ax0.imshow(gray48, cmap="gray"); ax0.axis("off")
        if per[cls] == 0:
            ax0.set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=9)
        ax0.set_title("true", fontsize=7)

        ax1 = axes[r, col0 + 1]
        ax1.imshow(overlay_rgb); ax1.axis("off")
        ax1.set_title(f"{idx_to_class[pred_idx]}\n{conf*100:.0f}%",
                      fontsize=7, color=color)

        per[cls] += 1

    # 关掉没用到的子图
    for c in class_to_idx:
        for j in range(per[c] * 2, ncols):
            axes[class_to_idx[c], j].axis("off")

    fig.suptitle(f"Grad-CAM ({arch}, split={args.split}) — 模型关注区域可视化", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"Grad-CAM 可视化已保存: {args.out}")


if __name__ == "__main__":
    main()
