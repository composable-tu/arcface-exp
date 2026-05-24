# SPDX-License-Identifier: MulanPSL-2.0

"""
Copyright (c) 2026 composable-tu
This project is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""

import torch
import torch.nn as nn

from losses import ArcFace
from mobilefacenet import get_mbf


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512, s=64.0, margin=0.5):
        super(FaceRecognitionModel, self).__init__()
        # 骨干网络
        self.backbone = get_mbf(fp16=False, num_features=embedding_size)
        # 分类头
        self.fc = nn.Linear(embedding_size, num_classes, bias=False)
        # ArcFace损失
        self.arcface = ArcFace(s=s, margin=margin)

    def forward(self, x, labels=None):
        # 获取特征
        features = self.backbone(x)

        # 归一化特征
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        # 归一化权重
        weight = torch.nn.functional.normalize(self.fc.weight, p=2, dim=1)

        # 计算logits
        logits = torch.nn.functional.linear(features, weight)

        # 如果提供了标签，计算ArcFace损失
        if labels is not None:
            logits = self.arcface(logits, labels)

        return logits, features


def create_model(num_classes, embedding_size=512, s=64.0, margin=0.5):
    """
    创建人脸识别模型
    Args:
        num_classes: 类别数量
        embedding_size: 特征维度
        s: ArcFace的缩放因子
        margin: ArcFace的margin参数
    Returns:
        FaceRecognitionModel: 人脸识别模型
    """
    model = FaceRecognitionModel(num_classes=num_classes, embedding_size=embedding_size, s=s, margin=margin)
    return model
