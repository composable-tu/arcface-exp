import torch

def get_best_device():
    """
    获取最佳可用设备
    Returns:
        str: 最佳可用设备名称
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    elif torch.xpu.is_available():
        return 'xpu'
    else:
        return 'cpu'