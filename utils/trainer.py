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

import logging
import os
import sys

import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, criterion, device, save_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir

        # 添加最佳性能跟踪
        self.best_f1 = 0.0
        self.best_loss = float('inf')

        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)

        # 配置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(os.path.join(self.save_dir, 'train.log'))])
        # 创建独立的控制台处理器并设置为WARNING级别，这样只有重要信息会显示在控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # 只显示WARNING及以上级别的日志
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        self.logger = logging.getLogger('FaceRecognitionTrainer')
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)  # 整体日志级别保持INFO，这样文件中会有详细信息

    def train_epoch(self, dataloader):
        """
        训练一个epoch
        Args:
            dataloader: 数据加载器
        Returns:
            tuple: (平均损失, 准确率, 精确率, 召回率, F1分数)
        """
        self.model.train()
        total_loss = 0.0
        all_targets = []
        all_predictions = []

        # 使用tqdm创建进度条，单一进度条不嵌套，减少刷屏
        progress_bar = tqdm(dataloader, desc=f'Training on {self.device}', leave=False,  # 每个 epoch 完成后清理这一行
                            dynamic_ncols=True,  # 自动适配终端宽度，避免换行
                            smoothing=0.1, mininterval=1.0,  # 限制刷新频率，避免终端异常刷屏/残留
                            file=sys.stdout  # 统一到 stdout，避免 stdout/stderr 打架
                            )

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # 移动数据到设备
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            logits, _ = self.model(inputs, targets)

            # 计算损失
            loss = self.criterion(logits, targets)

            # 反向传播
            loss.backward()

            # 更新参数
            self.optimizer.step()

            # 统计损失和预测结果
            total_loss += loss.item()
            _, predicted = logits.max(1)

            # 收集所有目标和预测用于后续计算指标
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # 计算当前批次的准确率
            correct = predicted.eq(targets).sum().item()
            total = targets.size(0)
            accuracy = 100. * correct / total

            # 更新进度条显示信息
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix(
                {'Loss': f'{loss.item():.4f}', 'Avg Loss': f'{avg_loss:.4f}', 'Acc': f'{accuracy:.2f}%'})

            # 打印进度 (每100个批次记录一次日志)
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

        # 计算整个epoch的指标
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * sum([p == t for p, t in zip(all_predictions, all_targets)]) / len(all_targets)

        # 防止sklearn计算出错，确保至少有一个样本
        if len(set(all_targets)) == 1 or len(all_predictions) == 0:
            precision = recall = f1 = 0.0
        else:
            precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

        self.logger.info(f'Epoch completed - Average Loss: {avg_loss:.4f}, '
                         f'Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, '
                         f'Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        return avg_loss, accuracy, precision, recall, f1

    def save_model(self, epoch, loss, accuracy, precision, recall, f1):
        """
        保存模型
        Args:
            epoch: 当前epoch
            loss: 当前损失
            accuracy: 当前准确率
            precision: 当前精确率
            recall: 当前召回率
            f1: 当前F1分数
        """
        # 保存最新的模型
        last_save_path = os.path.join(self.save_dir, f'last.pth')
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(), 'loss': loss, 'accuracy': accuracy,
                    'precision': precision, 'recall': recall, 'f1': f1}, last_save_path)
        self.logger.info(f'Latest model saved to {last_save_path}')

        # 如果 ArcFace 损失更低，则更新最佳模型
        if loss < self.best_loss:
            self.best_loss = loss
            best_save_path = os.path.join(self.save_dir, f'best.pth')
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(), 'loss': loss, 'accuracy': accuracy,
                        'precision': precision, 'recall': recall, 'f1': f1}, best_save_path)
            self.logger.info(f'Best model saved to {best_save_path} (Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%)')

    def load_model(self, model_path):
        """
        加载模型
        Args:
            model_path: 模型路径
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))
        accuracy = checkpoint.get('accuracy', 0.0)
        precision = checkpoint.get('precision', 0.0)
        recall = checkpoint.get('recall', 0.0)
        f1 = checkpoint.get('f1', 0.0)
        self.logger.info(f'Model loaded from {model_path} (Epoch: {epoch}, Loss: {loss:.4f}, '
                         f'Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, '
                         f'Recall: {recall:.4f}, F1: {f1:.4f})')
        return epoch, loss, accuracy, precision, recall, f1


def get_optimizer(model, lr=0.01, weight_decay=5e-4):
    """
    获取优化器
    Args:
        model: 模型
        lr: 学习率
        weight_decay: 权重衰减
    Returns:
        torch.optim.Optimizer: 优化器
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    return optimizer


def get_scheduler(optimizer, step_size=10, gamma=0.1):
    """
    获取学习率调度器
    Args:
        optimizer: 优化器
        step_size: 调整步长
        gamma: 衰减因子
    Returns:
        torch.optim.lr_scheduler._LRScheduler: 学习率调度器
    """
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return scheduler
