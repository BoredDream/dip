# 基于深度学习的视频流面部表情识别

## 一.图像表情识别
1. 图像处理：灰度化(数据集已经完成，但是后续视频流还是需要实现）
2. 
## 二.视频流表情识别
# 面部表情识别项目 (FER) — 项目上下文
## 项目背景
数字图像处理（DIP）课程项目，目标是实现一个基于摄像头的实时面部表情识别系统。
非毕业设计，2–3人团队，预计15天完成。
## 技术栈
- 语言：Python 3.9（conda 虚拟环境，环境名 fer，系统是 Python 3.12）
- 操作系统：WSL2（Ubuntu） + Windows
- 数据集：FER2013（Kaggle，48×48灰度图，7类表情）
- 核心库：OpenCV、PyTorch、NumPy、Pandas、Matplotlib、Seaborn、scikit-learn
- pip 镜像：清华源 https://pypi.tuna.tsinghua.edu.cn/simple
## 项目结构（待初始化）
fer/
├── data/           # FER2013 原始数据
├── preprocess/     # 图像预处理脚本
├── model/          # CNN 模型定义
├── train/          # 训练脚本
├── eval/           # 评估脚本
├── demo/           # 摄像头实时推理
└── notebooks/      # Jupyter 调试用
## 技术路线
### 预处理流程
1. 灰度化（cvtColor）
2. 人脸检测（Haar 级联分类器）
3. 直方图均衡化（equalizeHist）
4. Resize 到 48×48，归一化到 0–1
5. 数据增强（随机翻转、旋转±10°）仅用于训练集
### 模型结构（CNN）
输入 1×48×48
→ Conv2d(1,32,3) + BN + ReLU → MaxPool 2×2
→ Conv2d(32,64,3) + BN + ReLU → MaxPool 2×2
→ Conv2d(64,128,3) + BN + ReLU → MaxPool 2×2
→ Flatten → FC(2048,512) + Dropout(0.5)
→ FC(512,7)
### 训练配置
- 损失函数：CrossEntropyLoss + 类别权重（处理不平衡）
- 优化器：Adam，lr=1e-3
- Scheduler：StepLR，step_size=10，gamma=0.5
- Epochs：50，早停 patience=10
- Batch size：64
- 训练平台：Google Colab（T4 GPU）
### 评估指标
- 测试集 Accuracy、F1 Score
- 混淆矩阵（sklearn）
### 摄像头 Demo
- OpenCV VideoCapture 逐帧推理
- Haar 检测人脸 → 裁剪 → 模型推理 → 绘制矩形框 + 表情标签 + 置信度
- 在 Windows 端运行（绕过 WSL2 摄像头访问限制）
## 表情类别（7类）
angry / disgust / fear / happy / sad / surprise / neutral
## 当前任务
请帮我初始化项目目录结构，并依次生成以下文件的完整可运行代码：
1. preprocess/preprocess.py
2. model/cnn.py
3. train/train.py
4. eval/eval.py
5. demo/camera_demo.py
代码要求：
- 每个文件独立可运行，模块间通过文件路径解耦
- 包含必要注释，说明每个处理步骤对应的 DIP 原理
- 训练脚本支持命令行参数（数据路径、epoch数、学习率）

---

## 运行方式

> 建议在 conda 环境 `fer` 中运行（已含 torch/cuda/opencv 等）。以下命令均从仓库根目录执行。

```bash
# 0. 安装依赖（如需）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 1. 查看模型结构 / 前向自检
python model/cnn.py

# 2. 预处理演示：生成直方图均衡前后对比图 + 统计各类样本数
python preprocess/preprocess.py
#    -> 输出 checkpoints/equalize_demo.png

# 3. 训练（自动检测 GPU/CPU；默认数据目录 datasets/）
python train/train.py --epochs 50 --lr 1e-3 --batch-size 64
#    -> 最优权重 checkpoints/best.pth，训练曲线 checkpoints/train_curves.png

# 4. 在测试集上评估
python eval/eval.py --ckpt checkpoints/best.pth
#    -> 打印 Accuracy / Macro F1 / 分类报告，输出 checkpoints/confusion_matrix.png

# 5. 摄像头实时 Demo（⚠ WSL2 无法访问摄像头，请在 Windows 端运行）
python demo/camera_demo.py --ckpt checkpoints/best.pth --camera-index 0
```

## 目录结构

```
preprocess/   # 在线变换 + 摄像头人脸预处理工具
model/        # CNN 模型定义 (FERCNN)
train/        # 训练脚本（class weights / 早停 / StepLR）
eval/         # 评估脚本（Accuracy / F1 / 混淆矩阵）
demo/         # 摄像头实时推理
notebooks/    # Jupyter 调试
checkpoints/  # 权重与可视化输出
datasets/     # FER2013 数据（train/ 与 test/，ImageFolder 格式）
```

## 说明
- 数据集图片已是 48×48 灰度裁剪人脸（ImageFolder 格式），故"灰度化/人脸检测/缩放"主要作用于摄像头实时帧。
- 7 类（字母序）：angry, disgust, fear, happy, neutral, sad, surprise。
- 训练/评估/Demo 通过 `checkpoints/best.pth`（含 `class_to_idx`）解耦，脚本间不互相 import。
