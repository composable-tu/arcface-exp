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
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.model import create_model
from utils.dataset import FaceDataset
from utils.util import get_best_device


def load_model(model_path, num_classes=10572, embedding_size=512, device='cpu'):
    """
    加载训练好的模型

    Args:
        model_path: 模型权重文件路径
        num_classes: 类别数量（训练时的类别数）
        embedding_size: 嵌入特征维度
        device: 设备

    Returns:
        加载好的模型
    """
    # 创建模型
    model = create_model(num_classes=num_classes, embedding_size=embedding_size)

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"模型已从 {model_path} 加载")
    return model


def extract_embeddings(model, dataloader, device):
    """
    从数据集中提取所有样本的嵌入向量

    Args:
        model: 人脸识别模型
        dataloader: 数据加载器
        device: 计算设备

    Returns:
        tuple: (embeddings_dict: {label: [embeddings]}, image_paths_dict: {label: [paths]})
    """
    embeddings_dict = defaultdict(list)
    labels_list = []

    model.eval()

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="提取嵌入向量"):
            inputs = inputs.to(device)

            # 获取特征嵌入
            _, features = model(inputs)

            # 归一化特征
            features = torch.nn.functional.normalize(features, p=2, dim=1)

            # 按身份存储
            for i, target in enumerate(targets):
                label = target.item()
                embeddings_dict[label].append(features[i].cpu().numpy())
                labels_list.append(label)

    print(f"已提取 {len(labels_list)} 个嵌入向量，共 {len(embeddings_dict)} 个身份")

    return embeddings_dict


def parse_lfw_pairs(pairs_file='datasets/pairs.txt'):
    """
    解析LFW官方配对文件

    Args:
        pairs_file: LFW配对文件路径

    Returns:
        tuple: (same_pairs, diff_pairs)
            same_pairs: [(name, img1, img2), ...] 同一个人的配对
            diff_pairs: [(name1, img1, name2, img2), ...] 不同人的配对
    """
    same_pairs = []
    diff_pairs = []

    if not os.path.exists(pairs_file):
        print(f"警告: LFW配对文件不存在: {pairs_file}")
        print("将使用简化评估方法...")
        return None, None

    with open(pairs_file, 'r') as f:
        lines = f.readlines()

    # 第一行是配置信息
    config = lines[0].strip().split('\t')
    n_folds = int(config[0])
    n_same = int(config[1])
    n_diff = int(config[2])

    print(f"LFW配置: {n_folds}折, 每折{n_same}个相同配对, {n_diff}个不同配对")

    idx = 1
    for fold in range(n_folds):
        # 读取相同配对
        for _ in range(n_same):
            parts = lines[idx].strip().split('\t')
            name = parts[0]
            img1 = int(parts[1])
            img2 = int(parts[2])
            same_pairs.append((name, img1, name, img2))
            idx += 1

        # 读取不同配对
        for _ in range(n_diff):
            parts = lines[idx].strip().split('\t')
            name1 = parts[0]
            img1 = int(parts[1])
            name2 = parts[2]
            img2 = int(parts[3])
            diff_pairs.append((name1, img1, name2, img2))
            idx += 1

    print(f"解析完成: {len(same_pairs)}个相同配对, {len(diff_pairs)}个不同配对")

    return same_pairs, diff_pairs


def evaluate_with_pairs(embeddings_dict, same_pairs, diff_pairs, threshold=0.4):
    """
    使用配对文件进行评估

    Args:
        embeddings_dict: {label: [embeddings]}
        same_pairs: 相同配对列表
        diff_pairs: 不同配对列表
        threshold: 相似度阈值

    Returns:
        dict: 评估结果
    """
    print("开始评估...")

    positive_similarities = []
    negative_similarities = []

    # 注意：这里需要映射人名到实际的label
    # 由于FaceDataset是按文件夹顺序编号的，我们需要建立映射
    # 这里简化处理，假设你能提供正确的映射关系

    # 如果没有配对文件，使用简化的评估方法
    if same_pairs is None or diff_pairs is None:
        return evaluate_simplified(embeddings_dict, threshold)

    # TODO: 实现基于配对文件的评估
    # 这需要建立人名到dataset label的映射
    print("基于配对文件的评估需要额外的人名映射，使用简化评估...")
    return evaluate_simplified(embeddings_dict, threshold)


def evaluate_simplified(embeddings_dict, threshold=0.4):
    """
    简化评估方法：计算每个身份内的正样本对和随机采样的负样本对

    Args:
        embeddings_dict: {label: [embeddings]}
        threshold: 相似度阈值

    Returns:
        dict: 评估结果
    """
    print("使用简化评估方法...")

    positive_similarities = []
    negative_similarities = []

    identities = list(embeddings_dict.keys())

    # 计算正样本对（同一身份内）
    print("计算正样本对...")
    for identity in tqdm(identities, desc="正样本对"):
        embs = embeddings_dict[identity]
        if len(embs) < 2:
            continue

        # 转换为numpy数组
        embs_array = np.array(embs)

        # 计算该身份内所有配对的相似度
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                sim = cosine_similarity(
                    embs_array[i].reshape(1, -1),
                    embs_array[j].reshape(1, -1)
                )[0][0]
                positive_similarities.append(sim)

    print(f"正样本对数量: {len(positive_similarities)}")

    # 计算负样本对（随机采样）
    print("计算负样本对（随机采样10万对）...")
    max_negative_pairs = 100000  # 采样10万对已经足够准确

    negative_count = 0
    attempts = 0
    max_attempts = max_negative_pairs * 10  # 防止无限循环

    while negative_count < max_negative_pairs and attempts < max_attempts:
        attempts += 1

        # 随机选择两个不同的身份
        if len(identities) < 2:
            break

        id1, id2 = np.random.choice(identities, 2, replace=False)

        # 从每个身份中随机选择一个样本
        emb1 = embeddings_dict[id1][np.random.randint(0, len(embeddings_dict[id1]))]
        emb2 = embeddings_dict[id2][np.random.randint(0, len(embeddings_dict[id2]))]

        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        negative_similarities.append(sim)
        negative_count += 1

    print(f"负样本对数量: {len(negative_similarities)}")

    # 转换为numpy数组
    positive_similarities = np.array(positive_similarities)
    negative_similarities = np.array(negative_similarities)

    if len(positive_similarities) == 0 or len(negative_similarities) == 0:
        print("警告: 没有足够的样本对进行评估")
        return None

    # 计算各项指标
    true_positives = np.sum(positive_similarities > threshold)
    false_negatives = np.sum(positive_similarities <= threshold)

    true_negatives = np.sum(negative_similarities <= threshold)
    false_positives = np.sum(negative_similarities > threshold)

    accuracy = (true_positives + true_negatives) / (len(positive_similarities) + len(negative_similarities))
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    # 计算等错误率(EER)
    thresholds_range = np.linspace(0, 1, 1000)
    best_eer_threshold = 0.5
    min_diff = float('inf')

    for t in thresholds_range:
        far = np.sum(negative_similarities > t) / len(negative_similarities)
        frr = np.sum(positive_similarities <= t) / len(positive_similarities)
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            best_eer_threshold = t

    eer = (np.sum(negative_similarities > best_eer_threshold) / len(negative_similarities) +
           np.sum(positive_similarities <= best_eer_threshold) / len(positive_similarities)) / 2

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'eer': eer,
        'best_threshold': best_eer_threshold,
        'num_positive_pairs': len(positive_similarities),
        'num_negative_pairs': len(negative_similarities),
        'mean_positive_similarity': np.mean(positive_similarities),
        'mean_negative_similarity': np.mean(negative_similarities),
        'std_positive_similarity': np.std(positive_similarities),
        'std_negative_similarity': np.std(negative_similarities),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='LFW数据集模型验证')
    parser.add_argument('--data_dir', type=str, default='datasets/lfw-deepfunneled',
                        help='LFW数据集目录路径 (默认: datasets/lfw-deepfunneled)')
    parser.add_argument('--model_path', type=str, default='checkpoints/best.pth',
                        help='模型权重文件路径 (默认: checkpoints/best.pth)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='测试批次大小 (默认: 128)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作线程数 (默认: 4)')
    parser.add_argument('--num_classes', type=int, default=10572,
                        help='训练时的类别数，CASIA-WebFace约10572 (默认: 10572)')
    parser.add_argument('--embedding_size', type=int, default=512,
                        help='嵌入特征维度 (默认: 512)')
    parser.add_argument('--device', type=str, default=None,
                        help='使用的设备，如 cuda/cpu/mps (默认: 自动检测最佳设备)')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='人脸匹配的相似度阈值 (默认: 0.4)')
    parser.add_argument('--pairs_file', type=str, default=None,
                        help='LFW配对文件路径 (可选)')

    args = parser.parse_args()

    # 自动选择设备
    if args.device is None:
        args.device = get_best_device()

    print(f"使用设备: {args.device}")
    print(f"数据集路径: {args.data_dir}")
    print(f"模型路径: {args.model_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"相似度阈值: {args.threshold}")
    print("-" * 60)

    # 检查数据集是否存在
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据集目录不存在: {args.data_dir}")
        return

    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return

    # 加载模型
    model = load_model(args.model_path, args.num_classes, args.embedding_size, args.device)

    # 准备数据加载器
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = FaceDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    print(f"数据集包含 {len(dataset)} 张图像")
    print("-" * 60)

    # 提取嵌入向量
    embeddings_dict = extract_embeddings(model, dataloader, args.device)
    print("-" * 60)

    # 解析LFW配对文件（如果有）
    same_pairs = None
    diff_pairs = None
    if args.pairs_file:
        same_pairs, diff_pairs = parse_lfw_pairs(args.pairs_file)

    # 评估模型性能
    results = evaluate_with_pairs(embeddings_dict, same_pairs, diff_pairs, args.threshold)

    if results is None:
        print("评估失败")
        return

    # 输出结果
    print("\n" + "="*60)
    print("LFW 数据集验证结果")
    print("="*60)
    print(f"准确率 (Accuracy):     {results['accuracy']*100:.2f}%")
    print(f"精确率 (Precision):    {results['precision']*100:.2f}%")
    print(f"召回率 (Recall):       {results['recall']*100:.2f}%")
    print(f"F1分数 (F1 Score):     {results['f1_score']*100:.2f}%")
    print(f"等错误率 (EER):        {results['eer']*100:.2f}%")
    print(f"最佳阈值:              {results['best_threshold']:.3f}")
    print(f"正样本对数量:          {results['num_positive_pairs']}")
    print(f"负样本对数量:          {results['num_negative_pairs']}")
    print(f"正样本平均相似度:      {results['mean_positive_similarity']:.4f} ± {results['std_positive_similarity']:.4f}")
    print(f"负样本平均相似度:      {results['mean_negative_similarity']:.4f} ± {results['std_negative_similarity']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
