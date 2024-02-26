import os
import torch
from torchvision.models import resnet50


def get_model_size(model):
    torch.save(model.state_dict(), 'temp_model.pth')
    size_bytes = os.path.getsize('temp_model.pth')
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    os.remove('temp_model.pth')
    return size_bytes, size_kb, size_mb


model_size_b, model_size_kb, model_size_mb = get_model_size(model=resnet50(pretrained=True))
print(f'模型大小（B）：{model_size_b} B')
print(f'模型大小（KB）：{model_size_kb} KB')
print(f'模型大小（MB）：{model_size_mb} MB')
