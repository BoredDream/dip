"""
preprocess/preprocess.py
========================
图像预处理模块。提供两类能力：

1) 训练/评估用的「在线变换」(torchvision transforms)：
   在 DataLoader 取数据时即时执行，无需把处理后的图片另存到磁盘。

2) 摄像头实时推理用的工具函数 preprocess_face()：
   对一帧 BGR 图像执行完整的经典 DIP 流水线（灰度化 -> Haar 人脸检测 ->
   直方图均衡 -> 缩放 -> 归一化），输出可直接喂给模型的张量。

DIP 原理对照：
- 灰度化（cvtColor）：去除颜色冗余，表情信息主要体现在亮度/纹理。
- 直方图均衡（equalizeHist）：拉伸灰度分布、增强对比度，缓解光照不均。
- Haar 级联人脸检测：基于 Haar-like 特征 + 积分图 + AdaBoost 的经典目标检测。
- 缩放（resize）：统一到网络输入尺寸 48×48（重采样）。
- 归一化（/255）：把像素从 0–255 映射到 0–1，利于网络训练数值稳定。

注意：本项目数据集图片已是 48×48 灰度裁剪人脸，故数据集路径上
      「灰度化/人脸检测/缩放」基本是空操作；Haar 检测只在摄像头帧上才真正需要。
"""

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# 该文件所在仓库根目录（便于独立运行时定位数据集）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 网络输入尺寸
IMG_SIZE = 48


# --------------------------------------------------------------------------- #
# 1) 在线变换（数据集路径）
# --------------------------------------------------------------------------- #
class EqualizeHist:
    """自定义 transform：对单通道 PIL 图做直方图均衡（增强对比度）。"""

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)                 # PIL(L) -> np.uint8 [H, W]
        arr = cv2.equalizeHist(arr)         # 直方图均衡
        return Image.fromarray(arr)


def get_train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """训练集变换：含数据增强（随机水平翻转、±10° 旋转）。"""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保单通道
        EqualizeHist(),                               # 直方图均衡
        transforms.RandomHorizontalFlip(p=0.5),       # 数据增强：左右翻转
        transforms.RandomRotation(degrees=10),        # 数据增强：±10° 旋转
        transforms.Resize((img_size, img_size)),      # 统一尺寸
        transforms.ToTensor(),                        # [0,255] -> [0,1]，并转 CHW 张量
    ])


def get_eval_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """评估/验证集变换：无数据增强，保证结果可复现。"""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        EqualizeHist(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


# --------------------------------------------------------------------------- #
# 2) 摄像头实时推理路径
# --------------------------------------------------------------------------- #
def load_haar_cascade(cascade_path: str = None) -> cv2.CascadeClassifier:
    """加载 Haar 人脸级联分类器。默认使用 OpenCV 自带的正脸模型。"""
    if cascade_path is None:
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError(f"无法加载 Haar 级联文件: {cascade_path}")
    return cascade


def preprocess_face(frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier,
                    size: int = IMG_SIZE):
    """
    对一帧 BGR 图像做完整经典 DIP 预处理。

    返回: [(tensor[1,1,size,size], (x, y, w, h)), ...]
          每个元素对应一张检测到的人脸及其在原图中的包围框。
    """
    # 灰度化：表情主要由亮度/纹理表达
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Haar 人脸检测：在灰度图上扫描，返回若干 (x, y, w, h)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(48, 48))

    results = []
    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]        # 裁剪人脸区域
        roi = cv2.equalizeHist(roi)         # 直方图均衡，增强对比度
        roi = cv2.resize(roi, (size, size)) # 缩放到网络输入尺寸
        roi = roi.astype(np.float32) / 255.0  # 归一化到 [0,1]
        tensor = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0)  # -> [1,1,H,W]
        results.append((tensor, (int(x), int(y), int(w), int(h))))
    return results


# --------------------------------------------------------------------------- #
# 独立运行：演示直方图均衡效果 + 统计各类样本数
# --------------------------------------------------------------------------- #
def _count_dataset(split_dir: str):
    """统计某个 split（train/test）下各类别的样本数。"""
    if not os.path.isdir(split_dir):
        return {}
    counts = {}
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if os.path.isdir(cls_dir):
            counts[cls] = len([f for f in os.listdir(cls_dir)
                               if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    return counts


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境下也能保存图片
    import matplotlib.pyplot as plt

    data_dir = os.path.join(_REPO_ROOT, "datasets")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # 打印数据集各类样本数（用于确认类别不平衡）
    print("== 训练集各类样本数 ==")
    train_counts = _count_dataset(train_dir)
    for cls, n in train_counts.items():
        print(f"  {cls:<10}: {n}")
    print(f"  合计: {sum(train_counts.values())}")

    print("== 测试集各类样本数 ==")
    test_counts = _count_dataset(test_dir)
    for cls, n in test_counts.items():
        print(f"  {cls:<10}: {n}")
    print(f"  合计: {sum(test_counts.values())}")

    # 找一张样本图，演示「直方图均衡前/后」对比
    sample_path = None
    for cls in (train_counts or {}):
        cls_dir = os.path.join(train_dir, cls)
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if files:
            sample_path = os.path.join(cls_dir, files[0])
            break

    if sample_path:
        gray = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        equalized = cv2.equalizeHist(gray)

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes[0, 0].imshow(gray, cmap="gray");      axes[0, 0].set_title("Original")
        axes[0, 1].imshow(equalized, cmap="gray"); axes[0, 1].set_title("Equalized")
        axes[1, 0].hist(gray.ravel(), bins=256, range=(0, 255))
        axes[1, 0].set_title("Original histogram")
        axes[1, 1].hist(equalized.ravel(), bins=256, range=(0, 255))
        axes[1, 1].set_title("Equalized histogram")
        for ax in axes[0]:
            ax.axis("off")
        plt.tight_layout()

        out_path = os.path.join(_REPO_ROOT, "checkpoints", "equalize_demo.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120)
        print(f"\n直方图均衡对比图已保存: {out_path}")
    else:
        print("\n未找到样本图，跳过可视化。")
