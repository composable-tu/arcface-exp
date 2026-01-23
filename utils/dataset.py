import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._prepare_dataset()
    
    def _prepare_dataset(self):
        # 遍历数据目录，收集所有图片路径和标签
        label = 0
        for person_name in sorted(os.listdir(self.data_dir)):
            person_dir = os.path.join(self.data_dir, person_name)
            if os.path.isdir(person_dir):
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                label += 1
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataloader(data_dir, batch_size=32, num_workers=4, shuffle=True):
    """
    获取数据加载器
    Args:
        data_dir: 数据集目录
        batch_size: 批次大小
        num_workers: 工作线程数
        shuffle: 是否打乱数据
    Returns:
        DataLoader: 数据加载器
    """
    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 创建数据集
    dataset = FaceDataset(data_dir, transform=transform)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader, len(set(dataset.labels))
