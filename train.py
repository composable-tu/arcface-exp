import os
import argparse
import torch.nn as nn
from utils.dataset import get_dataloader
from utils.model import create_model
from utils.trainer import Trainer, get_optimizer, get_scheduler
from utils.util import get_best_device
from tqdm import tqdm


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Face Recognition Training')
    parser.add_argument('--data_dir', type=str, default='datasets/lfw-deepfunneled',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--num_features', type=int, default=512,
                        help='Embedding feature size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default=get_best_device(),
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 获取数据加载器
    tqdm.write(f'Loading dataset from {args.data_dir}...')
    dataloader, num_classes = get_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    tqdm.write(f'Dataset loaded successfully. Number of classes: {num_classes}')
    
    # 创建模型
    tqdm.write('Creating model...')
    model = create_model(
        num_classes=num_classes,
        embedding_size=args.num_features
    )
    model.to(args.device)
    tqdm.write('Model created successfully.')
    
    # 创建优化器
    optimizer = get_optimizer(model, lr=args.lr)
    
    # 创建学习率调度器
    scheduler = get_scheduler(optimizer, step_size=10, gamma=0.1)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # 开始训练
    tqdm.write(f'Starting training on {args.device}...')
    for epoch in range(args.epochs):
        tqdm.write(f'\nEpoch {epoch+1}/{args.epochs}')
        tqdm.write('-' * 50)
        
        # 训练一个epoch
        loss, accuracy, precision, recall, f1 = trainer.train_epoch(dataloader)
        
        # 输出指标
        tqdm.write(f'Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # 调整学习率
        scheduler.step()
        
        # 保存模型
        trainer.save_model(epoch + 1, loss, accuracy, precision, recall, f1)
    
    tqdm.write('Training completed!')


if __name__ == '__main__':
    main()