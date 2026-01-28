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

import os

import torch
from openvino import convert_model
from openvino import save_model

from utils.model import create_model

# 加载模型检查点
checkpoint = torch.load('checkpoints/best.pth', weights_only=False)

# 模型配置参数
num_classes = 5749  # LFW 数据集的类别数，请根据实际情况修改
embedding_size = 512

# 创建模型
model = create_model(num_classes=num_classes, embedding_size=embedding_size)

# 加载模型权重
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 创建示例输入
example_input = torch.randn(1, 3, 112, 112)  # 根据模型调整输入尺寸

# 转换为 TorchScript（只导出骨干网络用于特征提取）
traced_model = torch.jit.trace(model.backbone, example_input)

output_dir_pt = 'model/torchscript'

# 保存为 pt 文件
os.makedirs(output_dir_pt, exist_ok=True)
traced_model.save(os.path.join(output_dir_pt, 'model.pt'))
print(f'模型已成功导出到：{output_dir_pt}/model.pt')

# OpenVINO 导出配置
output_dir = 'model/openvino'
input_shape = [1, 3, 112, 112]

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 转换 TorchScript 模型到 OpenVINO IR
ov_model = convert_model('model/torchscript/model.pt')

# 保存 OpenVINO IR 模型
output_path = os.path.join(output_dir, 'model.xml')
save_model(ov_model, output_path)

print(f'OpenVINO IR 模型已成功导出到：{output_dir}')
