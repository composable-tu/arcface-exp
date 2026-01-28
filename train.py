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

import argparse
import os

import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.dataset import get_dataloader
from utils.model import create_model
from utils.trainer import Trainer, get_optimizer, get_scheduler
from utils.util import get_best_device, get_next_checkpoint_dir


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Face Recognition Training')
    parser.add_argument('--data_dir', type=str, default='datasets/lfw-deepfunneled', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--num_features', type=int, default=512, help='Embedding feature size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default=get_best_device(), help='Device to use for training')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # 如果未指定保存目录且存在checkpoints目录，则自动生成新目录
    if args.save_dir == 'checkpoints':
        args.save_dir = get_next_checkpoint_dir('checkpoints')

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 获取数据加载器
    tqdm.write(f'Loading dataset from {args.data_dir}...')
    dataloader, num_classes = get_dataloader(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True)
    tqdm.write(f'Dataset loaded successfully. Number of classes: {num_classes}')

    # 创建模型
    tqdm.write('Creating model...')
    model = create_model(num_classes=num_classes, embedding_size=args.num_features)
    model.to(args.device)
    tqdm.write('Model created successfully.')

    # 创建优化器
    optimizer = get_optimizer(model, lr=args.lr)

    # 创建学习率调度器 - 使用监控损失的调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 创建训练器
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=args.device, save_dir=args.save_dir)

    # 如果指定了恢复训练的检查点
    start_epoch = 0
    if args.resume_from:
        tqdm.write(f'Loading checkpoint from {args.resume_from}...')
        start_epoch, loss, accuracy, precision, recall, f1 = trainer.load_model(args.resume_from)
        tqdm.write(f'Resuming training from epoch {start_epoch + 1}. Previous metrics - '
                   f'Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision * 100:.2f}%, '
                   f'Recall: {recall * 100:.2f}%, F1: {f1 * 100:.2f}%')

    # 计算总的训练轮数
    total_epochs = args.epochs
    if args.resume_from:
        # 如果是从检查点恢复，确保训练完整个周期（比如总共训练50个epoch）
        # 但不要超过设定的总轮数
        remaining_epochs = total_epochs - start_epoch
        tqdm.write(f'Total epochs: {total_epochs}, Already completed: {start_epoch}, Remaining: {remaining_epochs}')
    
    # 开始训练
    tqdm.write(f'Starting training on {args.device}...')
    for epoch in range(start_epoch, total_epochs):
        tqdm.write(f'\nEpoch {epoch + 1}/{total_epochs}')
        tqdm.write('-' * 50)

        # 训练一个epoch
        loss, accuracy, precision, recall, f1 = trainer.train_epoch(dataloader)

        # 输出当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        tqdm.write(f'Current learning rate: {current_lr:.6f}')

        # 输出指标
        tqdm.write(
            f'Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1: {f1 * 100:.2f}%')

        # 使用ReduceLROnPlateau调度器，根据损失调整学习率
        scheduler.step(loss)

        # 保存模型
        trainer.save_model(epoch + 1, loss, accuracy, precision, recall, f1)

    tqdm.write('Training completed!')


if __name__ == '__main__':
    main()
