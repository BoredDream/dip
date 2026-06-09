# 基于深度学习的视频流面部表情识别

本项目是一个面向 FER2013 数据集的面部表情识别项目，包含图像预处理、CNN/ResNet18 模型训练、测试集评估、Grad-CAM 可解释性可视化，以及基于摄像头的视频流实时表情识别 Demo。

## 功能概览

- 使用 FER2013 七分类表情数据集：`angry`、`disgust`、`fear`、`happy`、`neutral`、`sad`、`surprise`
- 支持轻量 CNN 与 ResNet18 两种模型骨干
- 支持 CLAHE 对比度增强、随机裁剪、随机擦除、Ten-Crop TTA 等处理流程
- 支持训练曲线、混淆矩阵、Grad-CAM 热力图输出
- 支持 OpenCV 摄像头视频流实时推理

## 项目结构

```text
.
├── checkpoints/          # 模型权重与可视化输出
├── datasets/             # FER2013 数据集，按 ImageFolder 格式组织
│   ├── train/            # 训练集
│   └── test/             # 测试集，包含 PublicTest/PrivateTest 图片
├── demo/
│   └── camera_demo.py    # 摄像头实时视频流表情识别
├── docs/                 # 项目分析与说明文档
├── eval/
│   ├── eval.py           # 测试集评估
│   └── visualize_cam.py  # Grad-CAM 可视化
├── model/
│   ├── cnn.py            # CNN / ResNet18 模型定义
│   └── gradcam.py        # Grad-CAM 工具
├── preprocess/
│   └── preprocess.py     # 图像预处理与摄像头帧预处理
├── train/
│   └── train.py          # 模型训练脚本
├── requirements.txt      # Python 依赖
└── README.md             # 项目使用说明
```

## 环境配置

建议使用 Conda 创建独立环境。项目推荐 Python 3.9；如果使用 Python 3.10/3.11，通常也可以运行，但需要确保 PyTorch 与本机 CUDA/CPU 环境匹配。

```bash
conda create -n fer python=3.9 -y
conda activate fer
```

安装 PyTorch 和 torchvision。CPU 环境可使用：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

如果你有 NVIDIA GPU，请根据自己的 CUDA 版本安装对应的 PyTorch 版本。例如 CUDA 12.1：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

然后安装项目其余依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

主要依赖包括：

- `torch`、`torchvision`：模型训练与推理
- `opencv-python`：摄像头读取、人脸检测、图像处理
- `numpy`、`pandas`：数据处理
- `scikit-learn`：评估指标与混淆矩阵
- `matplotlib`、`seaborn`：训练曲线和评估图像输出
- `Pillow`：图像读取与 transform 处理

## 数据集准备

项目默认从 `datasets/` 读取数据，目录需要保持 ImageFolder 格式：

```text
datasets/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

FER2013 图片通常为 48x48 灰度人脸图。项目中的摄像头 Demo 会对实时视频帧执行灰度化、人脸检测、CLAHE 增强、缩放和归一化。

## 常用终端命令

以下命令均在项目根目录执行。

查看模型结构并做一次前向自检：

```bash
python model/cnn.py
```

运行预处理演示，统计数据集并输出对比度增强示例图：

```bash
python preprocess/preprocess.py
```

训练轻量 CNN：

```bash
python train/train.py --arch cnn --epochs 60 --lr 1e-3 --batch-size 64
```

训练 ResNet18：

```bash
python train/train.py --arch resnet18 --epochs 40 --lr 5e-4 --batch-size 64
```

如果在 Windows 上 DataLoader 多进程报错，可以把 worker 数量设为 0：

```bash
python train/train.py --arch cnn --epochs 60 --lr 1e-3 --batch-size 64 --num-workers 0
```

训练完成后，默认输出：

- `checkpoints/best.pth`：按验证集准确率保存的最佳权重
- `checkpoints/best_loss.pth`：按验证集 loss 保存的权重
- `checkpoints/train_curves.png`：训练/验证曲线

## 模型评估

在 PrivateTest 子集上评估：

```bash
python eval/eval.py --ckpt checkpoints/best.pth --split private
```

启用 Ten-Crop TTA 测试增强：

```bash
python eval/eval.py --ckpt checkpoints/best.pth --split private --tta
```

可选的 `--split` 参数：

- `private`：只评估 `PrivateTest_*` 图片，默认值
- `public`：只评估 `PublicTest_*` 图片
- `all`：评估全部测试图片

评估脚本会打印 Accuracy、Macro F1、分类报告，并输出：

```text
checkpoints/confusion_matrix.png
```

## Grad-CAM 可解释性可视化

生成每类若干张 Grad-CAM 热力图：

```bash
python eval/visualize_cam.py --ckpt checkpoints/best.pth --per-class 3
```

默认输出：

```text
checkpoints/gradcam_demo.png
```

## Python 视频流应用终端

摄像头实时表情识别入口为：

```bash
python demo/camera_demo.py --ckpt checkpoints/best.pth --camera-index 0 --window 8
```

参数说明：

- `--ckpt`：模型权重路径，默认 `checkpoints/best.pth`
- `--camera-index`：摄像头编号，常见值为 `0`；如果打不开摄像头可尝试 `1`
- `--window`：视频流时序平滑窗口，数值越大，预测标签越稳定但响应越慢
- `--cascade`：自定义 Haar 人脸检测 XML 路径，默认使用 OpenCV 自带模型

运行后会打开摄像头窗口，检测到人脸后显示表情类别和置信度。按 `q` 退出。

注意：WSL2 默认无法直接访问 Windows 主机摄像头。视频流 Demo 建议在 Windows 终端、PowerShell、Anaconda Prompt 或 VS Code 的 Windows Python 环境中运行。

## 常见问题

如果提示找不到 `checkpoints/best.pth`，需要先训练模型，或确认权重文件已经放到 `checkpoints/` 目录。

如果摄像头打不开，先确认系统摄像头权限已开启，再尝试：

```bash
python demo/camera_demo.py --ckpt checkpoints/best.pth --camera-index 1
```

如果安装 OpenCV 后仍无法导入 `cv2`，重新安装：

```bash
pip uninstall opencv-python -y
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果没有 GPU，训练和评估会自动使用 CPU，但训练速度会明显变慢。

## 推荐运行顺序

```bash
conda activate fer
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python preprocess/preprocess.py
python train/train.py --arch cnn --epochs 60 --lr 1e-3 --batch-size 64
python eval/eval.py --ckpt checkpoints/best.pth --split private
python demo/camera_demo.py --ckpt checkpoints/best.pth --camera-index 0 --window 8
```
