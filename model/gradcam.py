"""
model/gradcam.py
================
Grad-CAM（Gradient-weighted Class Activation Mapping）实现。

【这是本项目的"可解释性创新点"】
普通 CNN 是黑箱：只给出"happy 78%"，但说不清"凭什么"。Grad-CAM 能画出一张
热力图，指出**模型究竟看着人脸的哪块区域**做出该判断（嘴角？眉头？眼睛？）。
这把分类结果从"信不信由你"变成"看得见的证据"，对答辩和调试都极有价值。

原理（可解释、纯数学，无需额外训练）：
1. 前向传播，拿到最后一个卷积层输出的特征图 A（C 个通道，每个是空间响应图）。
2. 反向传播目标类别的分数，得到该分数对每个通道特征图的梯度。
3. 把每个通道的梯度做全局平均池化 -> 通道权重 αc（= 该通道对此类别的重要程度）。
4. 用 αc 加权求和各通道特征图，再 ReLU（只保留正贡献），即得类激活图。
5. 上采样到原图大小并叠加成热力图。

参考：Selvaraju et al., "Grad-CAM", ICCV 2017。
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """对给定模型的目标卷积层做 Grad-CAM。"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = None     # 前向特征图 A
        self._gradients = None       # A 的梯度
        # 注册前向/反向钩子，截获目标层的激活与梯度
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        """
        input_tensor: [1, 1, H, W]
        返回: (cam[H, W] 取值 0~1 的 numpy 热力图, 预测类别索引, 该类概率)
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)            # 前向（触发 forward hook）
        prob = F.softmax(logits, dim=1)[0]
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        logits[0, class_idx].backward()              # 反向（触发 backward hook）

        # αc：对梯度做空间全局平均，作为各通道权重
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # 加权求和
        cam = F.relu(cam)                            # 只保留正贡献
        cam = F.interpolate(cam, size=input_tensor.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        # 归一化到 0~1，便于上色
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, class_idx, float(prob[class_idx])


def overlay_cam(gray_img: np.ndarray, cam: np.ndarray, alpha: float = 0.5):
    """把热力图（0~1）叠加到灰度人脸上，返回可显示的 BGR 图。需要 OpenCV。"""
    import cv2
    gray_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (gray_img.shape[1], gray_img.shape[0]))
    return cv2.addWeighted(gray_bgr, 1 - alpha, heatmap, alpha, 0)
