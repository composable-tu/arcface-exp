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


def get_best_device():
    """
    获取最佳可用设备
    Returns:
        str: 最佳可用设备名称
    """
    if torch.cuda.is_available():
        return 'cuda' # NVIDIA GPU
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps' # Apple Silicon
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu' # Intel GPU
    elif hasattr(torch, 'musa') and torch.musa.is_available():
        return 'musa' # 摩尔线程 MUSA
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu' # 华为 Ascend
    elif hasattr(torch, 'xla') and torch.xla.is_available():
        return 'xla' # Google TPU
    else:
        return 'cpu' # CPU


def get_next_checkpoint_dir(base_dir='checkpoints'):
    """获取下一个可用的检查点目录名"""
    counter = 1
    current_dir = base_dir
    while os.path.exists(current_dir):
        current_dir = f"{base_dir}{counter}"
        counter += 1
    return current_dir
