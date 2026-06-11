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

本脚本跨平台：自动按系统选择摄像头后端（Linux=V4L2，其它平台=默认），并支持
无窗口（headless）模式把标注结果直接存成视频，方便 SSH / 无显示环境下使用。

运行示例：
    # Linux（原生桌面，弹窗实时显示，按 q 退出）
    python demo/camera_demo.py --ckpt checkpoints/best.pth --camera-index 0

    # Linux 无显示 / SSH：不弹窗，把标注后的视频写到 demo_out.mp4，处理 300 帧后停止
    python demo/camera_demo.py --ckpt checkpoints/best.pth --no-display --max-frames 300

    # Windows
    python demo/camera_demo.py --ckpt checkpoints/best.pth --camera-index 0

平台提示：
- Linux：摄像头通常是 /dev/video0；若打不开先确认设备存在，且当前用户在 video 组
  （sudo usermod -aG video $USER 后重新登录），或设备已通过 ACL 放行。
- WSL2：默认无法访问主机摄像头，请改用 --no-display 处理离线视频，或在原生系统上运行。
"""

import argparse
import os
import platform
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
    p.add_argument(
        "--ckpt",
        default=os.path.join(_REPO_ROOT, "checkpoints", "best.pth"),
        help="模型权重文件路径",
    )
    p.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="摄像头编号（Linux 对应 /dev/videoN）",
    )
    p.add_argument(
        "--cascade", default=None, help="Haar 级联 xml 路径（默认用 OpenCV 自带）"
    )
    p.add_argument("--window", type=int, default=8, help="时序平滑窗口（帧数）")
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "v4l2", "any"],
        help="摄像头后端：auto(Linux=V4L2，其它=默认) / v4l2 / any",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="无窗口模式（headless/SSH）：不弹窗，把标注后的视频写到 --save",
    )
    p.add_argument(
        "--save",
        default=None,
        help="把标注结果保存为 mp4 视频；--no-display 时默认 demo_out.mp4",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="处理多少帧后自动停止（0=不限制，配合 --no-display 使用）",
    )
    return p.parse_args()


def resolve_backend(name: str) -> int:
    """把后端名映射成 OpenCV 的 CAP_* 常量。auto: Linux 用 V4L2，其它平台用默认。"""
    if name == "v4l2":
        return cv2.CAP_V4L2
    if name == "any":
        return cv2.CAP_ANY
    return cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY


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
        self.missed = 0  # 连续未匹配帧数，用于淘汰旧轨迹

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

    backend = resolve_backend(args.backend)
    cap = cv2.VideoCapture(args.camera_index, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 {args.camera_index}（后端={args.backend}）。\n"
            "- Linux：确认 /dev/video{0} 存在，且当前用户在 video 组"
            "（sudo usermod -aG video $USER 后重新登录）；可尝试 --camera-index 1。\n"
            "- WSL2：默认无法访问主机摄像头，请在原生系统运行或改用离线视频。".format(
                args.camera_index
            )
        )

    # 是否需要弹窗显示：headless 或显式 --no-display 时关闭
    headless = (
        args.no_display
        or not os.environ.get("DISPLAY")
        and platform.system() == "Linux"
    )
    save_path = args.save or ("demo_out.mp4" if headless else None)
    writer = None  # 惰性创建（拿到第一帧尺寸后再建）

    tracks = []  # 活跃的人脸轨迹列表
    mode = "无窗口(headless)" if headless else "弹窗显示"
    print(
        f"骨干: {arch} | 时序窗口: {args.window} 帧 | 模式: {mode}"
        + (f" | 保存到 {save_path}" if save_path else "")
        + " | 按 q 退出。"
    )
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取帧失败，退出。")
            break

        # 完整 DIP 预处理 + 检测出的每张人脸的当前帧概率
        faces = preprocess_face(frame, cascade)
        for t in tracks:
            t.missed += 1  # 先全部记为"本帧未匹配"，匹配上再清零

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
            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # 淘汰长期未匹配的轨迹（人离开画面）
        tracks = [t for t in tracks if t.missed <= args.window]

        # 保存到视频文件（惰性按首帧尺寸创建 writer）
        if save_path is not None:
            if writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    save_path,
                    fourcc,
                    fps if fps > 1 else 20.0,
                    (frame.shape[1], frame.shape[0]),
                )
            writer.write(frame)

        # 弹窗显示（headless 模式跳过，避免无 DISPLAY 时崩溃）
        if not headless:
            cv2.imshow("FER Demo (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1
        if args.max_frames and frame_count >= args.max_frames:
            print(f"已处理 {frame_count} 帧，达到 --max-frames，停止。")
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"标注视频已保存: {save_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
