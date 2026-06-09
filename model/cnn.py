"""
model/cnn.py
============
面部表情识别（FER）的模型定义。提供两种可切换骨干（arch）：

1) "cnn"       —— 轻量自定义 3 层 CNN（默认，符合 DIP 课"手搓可解释"定位）。
2) "resnet18"  —— ImageNet 预训练的 ResNet18 微调（创新点：迁移学习冲更高精度）。

两种骨干都接受 1×48×48 单通道灰度输入、输出 7 类 logits，可被 train/eval/demo
无差别调用；具体用哪种由 build_model(arch=...) 决定，并随 checkpoint 一起保存。

DIP / CNN 原理小结（可解释性是本项目的硬要求，逐条对应）：
- 卷积层（Conv2d）：用一组可学习卷积核做空间滤波，自动学习边缘/纹理/角点等局部特征算子
  （是 Sobel、Laplacian 等手工算子的泛化）。
- 批归一化（BatchNorm2d）：对每个通道特征图标准化，稳定数值分布、加速收敛。
- 激活（ReLU）：引入非线性，使网络能拟合复杂表情模式。
- 最大池化（MaxPool2d）：局部窗口取最大响应做下采样，带来平移不变性并降算量。
- 全连接（Linear）：把高层特征映射到 7 类打分。
- Dropout：训练时随机丢弃神经元，抑制过拟合。

【相对初版的改动·可解释】
- 卷积改 padding=1（same 卷积）：不再每层丢边缘一圈像素，48×48 的脸边缘信息得以保留。
- 末端加 AdaptiveAvgPool2d((4,4))：把任意空间尺寸自适应压到固定 4×4，使全连接层输入
  维度恒为 128×4×4=2048，结构更稳健（换输入尺寸也不必改 FC）。
"""

import torch
import torch.nn as nn

# 7 类表情（与 torchvision.ImageFolder 的字母序一致）
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# --------------------------------------------------------------------------- #
# 骨干一：轻量自定义 CNN
# --------------------------------------------------------------------------- #
class FERCNN(nn.Module):
    """48×48 灰度人脸 -> 7 类表情 的轻量 CNN（padding=1 + 自适应池化）。"""

    def __init__(self, num_classes: int = 7):
        super().__init__()

        # ---- 特征提取：3 个 [卷积(same) + BN + ReLU + 最大池化] 块 ----
        self.features = nn.Sequential(
            # 块1：1 -> 32 通道，48 --pool--> 24
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # same 卷积，保留边缘
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 块2：32 -> 64 通道，24 --pool--> 12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 块3：64 -> 128 通道，12 --pool--> 6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 自适应平均池化：6×6 -> 固定 4×4，稳定后续全连接输入维度
        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        # ---- 分类：展平 -> 全连接 ----
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 128×4×4 -> 2048
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                    # 抑制过拟合
            nn.Linear(512, num_classes),        # 输出 7 类打分（logits）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def gradcam_target_layer(self) -> nn.Module:
        """Grad-CAM 取最后一个卷积层的输出（语义最丰富、空间仍可定位）。"""
        return self.features[8]  # 第三个 Conv2d


# --------------------------------------------------------------------------- #
# 骨干二：预训练 ResNet18 微调（创新点：迁移学习）
# --------------------------------------------------------------------------- #
class ResNet18FER(nn.Module):
    """
    ImageNet 预训练 ResNet18 适配 1×48×48 灰度 FER。

    可解释的适配手法（只动两处，其余层全部沿用预训练权重）：
    - 第一层 conv1 改成单通道输入：新权重 = 预训练 RGB 三通道权重按通道求平均，
      相当于"把彩色边缘检测器折叠成灰度边缘检测器"，最大程度保留预训练先验。
    - 最后的全连接 fc 改成 512 -> 7，重新学习表情分类头。
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        backbone = None
        if pretrained:
            try:  # 新版 torchvision 用 weights 枚举；失败则退回随机初始化
                backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception as e:
                print(f"[ResNet18FER] 预训练权重加载失败（{e}），改用随机初始化。")
        if backbone is None:
            backbone = models.resnet18(weights=None)

        # 单通道 conv1：用 RGB 权重均值迁移
        old_conv = backbone.conv1                       # Conv2d(3,64,7,2,3)
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv

        # 新分类头
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def gradcam_target_layer(self) -> nn.Module:
        """Grad-CAM 取最后一个残差块组的输出。"""
        return self.backbone.layer4


# --------------------------------------------------------------------------- #
# 工厂函数
# --------------------------------------------------------------------------- #
def build_model(num_classes: int = 7, arch: str = "cnn",
                pretrained: bool = True) -> nn.Module:
    """统一构建模型。arch ∈ {"cnn", "resnet18"}。"""
    arch = arch.lower()
    if arch == "cnn":
        return FERCNN(num_classes=num_classes)
    if arch == "resnet18":
        return ResNet18FER(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"未知 arch: {arch}（可选 cnn / resnet18）")


if __name__ == "__main__":
    # 独立运行：对两种骨干各打印参数量并做前向自检
    for arch in ("cnn", "resnet18"):
        model = build_model(arch=arch, pretrained=False)
        n_params = sum(p.numel() for p in model.parameters())
        dummy = torch.randn(2, 1, 48, 48)              # 两张 48×48 灰度图
        out = model(dummy)
        assert out.shape == (2, len(EMOTIONS)), f"{arch} 输出形状应为 [2, 7]"
        print(f"[{arch:8s}] 参数量: {n_params:>10,}  "
              f"输入 {tuple(dummy.shape)} -> 输出 {tuple(out.shape)}  ✔")
