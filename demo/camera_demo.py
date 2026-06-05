"""
demo/camera_demo.py
===================
摄像头实时表情识别 Demo。

流程（逐帧）：
    VideoCapture 取帧 -> preprocess_face()（灰度/Haar 检测/均衡/缩放/归一化）
    -> 模型推理 -> 在原帧绘制人脸框 + 表情标签 + 置信度 -> 显示

⚠️ 重要：WSL2 默认无法直接访问主机摄像头。请在 **Windows 端的 Python 环境** 中运行本脚本
   （把训练好的 checkpoints/best.pth 拷到 Windows 一侧），按 q 退出。

运行示例（Windows）：
    python demo/camera_demo.py --ckpt checkpoints/best.pth --camera-index 0
"""

import argparse
import os
import sys

import cv2
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.cnn import build_model  # noqa: E402
from preprocess.preprocess import load_haar_cascade, preprocess_face  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="摄像头实时表情识别")
    p.add_argument("--ckpt", default=os.path.join(_REPO_ROOT, "checkpoints", "best.pth"),
                   help="模型权重文件路径")
    p.add_argument("--camera-index", type=int, default=0, help="摄像头编号")
    p.add_argument("--cascade", default=None, help="Haar 级联 xml 路径（默认用 OpenCV 自带）")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 加载模型与类别映射 ----
    ckpt = torch.load(args.ckpt, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    num_classes = ckpt.get("num_classes", len(class_to_idx))
    idx_to_class = [None] * num_classes
    for cls, idx in class_to_idx.items():
        idx_to_class[idx] = cls

    model = build_model(num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cascade = load_haar_cascade(args.cascade)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 {args.camera_index}。"
            "若在 WSL2 中运行，请改到 Windows 端执行本脚本。"
        )

    print("按 q 退出。")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取帧失败，退出。")
            break

        # 完整 DIP 预处理 + 检测出的每张人脸
        faces = preprocess_face(frame, cascade)
        for tensor, (x, y, w, h) in faces:
            logits = model(tensor.to(device))
            prob = torch.softmax(logits, dim=1)
            conf, pred = prob.max(1)
            label = f"{idx_to_class[pred.item()]} {conf.item() * 100:.0f}%"

            # 绘制人脸框 + 标签
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("FER Demo (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
