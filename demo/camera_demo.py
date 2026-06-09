"""
demo/camera_demo.py
===================
摄像头实时表情识别 Demo（含时序平滑，真正"视频流"而非逐帧独立）。

流程（逐帧）：
    VideoCapture 取帧 -> preprocess_face()（灰度/Haar 检测/CLAHE/缩放/归一化）
    -> 模型推理得 softmax 概率 -> 简单 IoU 跟踪把当前人脸关联到历史轨迹
    -> 对每条轨迹最近 N 帧的概率做平均投票（消除标签跳变）
    -> 在原帧绘制人脸框 + 平滑后的表情标签 + 置信度 -> 显示

【为什么要时序平滑·可解释】单帧预测会因眨眼、运动模糊、光照抖动而"跳字"
（这一帧 happy 下一帧 neutral）。对一小段时间窗内的概率取平均，相当于一个
低通滤波，输出更稳定，也更符合"表情是一段持续状态"的常识。

⚠️ 重要：WSL2 默认无法直接访问主机摄像头。请在 **Windows 端的 Python 环境** 中运行本脚本
   （把训练好的 checkpoints/best.pth 拷到 Windows 一侧），按 q 退出。

运行示例（Windows）：
    python demo/camera_demo.py --ckpt checkpoints/best.pth --camera-index 0
"""

import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.cnn import build_model  # noqa: E402
from preprocess.preprocess import load_haar_cascade, preprocess_face  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="摄像头实时表情识别（时序平滑）")
    p.add_argument("--ckpt", default=os.path.join(_REPO_ROOT, "checkpoints", "best.pth"),
                   help="模型权重文件路径")
    p.add_argument("--camera-index", type=int, default=0, help="摄像头编号")
    p.add_argument("--cascade", default=None, help="Haar 级联 xml 路径（默认用 OpenCV 自带）")
    p.add_argument("--window", type=int, default=8, help="时序平滑窗口（帧数）")
    return p.parse_args()


def iou(box_a, box_b):
    """两个 (x, y, w, h) 框的交并比，用于跨帧关联同一张脸。"""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceTrack:
    """一条人脸轨迹：保存最近 N 帧的 softmax 概率，用于平均投票。"""

    def __init__(self, box, window):
        self.box = box
        self.probs = deque(maxlen=window)
        self.missed = 0          # 连续未匹配帧数，用于淘汰旧轨迹

    def update(self, box, prob):
        self.box = box
        self.probs.append(prob)
        self.missed = 0

    def smoothed(self):
        """最近 N 帧概率平均 -> (类别索引, 平滑置信度)。"""
        avg = np.mean(self.probs, axis=0)
        idx = int(avg.argmax())
        return idx, float(avg[idx])


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 加载模型与类别映射（骨干类型从 ckpt 读取）----
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

    cascade = load_haar_cascade(args.cascade)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 {args.camera_index}。"
            "若在 WSL2 中运行，请改到 Windows 端执行本脚本。"
        )

    tracks = []   # 活跃的人脸轨迹列表
    print(f"骨干: {arch} | 时序窗口: {args.window} 帧 | 按 q 退出。")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取帧失败，退出。")
            break

        # 完整 DIP 预处理 + 检测出的每张人脸的当前帧概率
        faces = preprocess_face(frame, cascade)
        for t in tracks:
            t.missed += 1   # 先全部记为"本帧未匹配"，匹配上再清零

        for tensor, box in faces:
            logits = model(tensor.to(device))
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()

            # 与已有轨迹做 IoU 关联；匹配不上则新建轨迹
            best, best_iou = None, 0.3
            for t in tracks:
                v = iou(t.box, box)
                if v > best_iou:
                    best, best_iou = t, v
            if best is None:
                best = FaceTrack(box, args.window)
                tracks.append(best)
            best.update(box, prob)

            # 用平滑后的结果绘制
            idx, conf = best.smoothed()
            label = f"{idx_to_class[idx]} {conf * 100:.0f}%"
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 淘汰长期未匹配的轨迹（人离开画面）
        tracks = [t for t in tracks if t.missed <= args.window]

        cv2.imshow("FER Demo (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
