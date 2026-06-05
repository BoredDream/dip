"""
model/cnn.py
============
面部表情识别（FER）使用的卷积神经网络（CNN）定义。

DIP / CNN 原理小结：
- 卷积层（Conv2d）：本质是用一组可学习的卷积核对图像做空间滤波，
  相当于自动学习边缘 / 纹理 / 角点等局部特征算子（类似 Sobel、Laplacian 等手工算子的泛化）。
- 批归一化（BatchNorm2d）：对每个通道的特征图做标准化，稳定数值分布、加速收敛、缓解梯度问题。
- 激活函数（ReLU）：引入非线性，使网络能拟合复杂的表情模式。
- 最大池化（MaxPool2d）：在局部窗口取最大响应，做下采样，
  带来一定的平移不变性并降低计算量（DIP 中的降采样思想）。
- 全连接层（Linear）：把高层特征映射到 7 个表情类别的打分。
- Dropout：训练时随机丢弃神经元，抑制过拟合。

输入：1×48×48 的单通道（灰度）人脸图。
本网络采用 valid 卷积（padding=0），尺寸变化：
    48 --conv3--> 46 --pool2--> 23
    23 --conv3--> 21 --pool2--> 10
    10 --conv3-->  8 --pool2-->  4
最终特征图为 128×4×4 = 2048，与全连接层 FC(2048, 512) 对齐。
"""

import torch
import torch.nn as nn

# 7 类表情（与 torchvision.ImageFolder 的字母序一致）
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class FERCNN(nn.Module):
    """48×48 灰度人脸 -> 7 类表情 的 CNN。"""

    def __init__(self, num_classes: int = 7):
        super().__init__()

        # ---- 特征提取部分：3 个 [卷积 + BN + ReLU + 最大池化] 块 ----
        self.features = nn.Sequential(
            # 块1：1 -> 32 通道，48 -> 46 -> 23
            nn.Conv2d(1, 32, kernel_size=3),   # valid 卷积，空间滤波提取低级边缘特征
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 下采样：23×23

            # 块2：32 -> 64 通道，23 -> 21 -> 10
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 下采样：10×10

            # 块3：64 -> 128 通道，10 -> 8 -> 4
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 下采样：4×4
        )

        # ---- 分类部分：展平 -> 全连接 ----
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 128×4×4 -> 2048
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                    # 抑制过拟合
            nn.Linear(512, num_classes),        # 输出 7 类打分（logits）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int = 7) -> FERCNN:
    """工厂函数，便于其它脚本统一构建模型。"""
    return FERCNN(num_classes=num_classes)


if __name__ == "__main__":
    # 独立运行：打印结构、参数量，并对 dummy 输入做前向，验证输出形状为 [N, 7]
    model = build_model()
    print(model)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数总量: {n_params:,}  (可训练: {n_train:,})")

    dummy = torch.randn(1, 1, 48, 48)          # 一张 48×48 灰度图
    out = model(dummy)
    print(f"输入形状: {tuple(dummy.shape)}  ->  输出形状: {tuple(out.shape)}")
    assert out.shape == (1, len(EMOTIONS)), "输出形状应为 [1, 7]"
    print("前向验证通过 ✔")
