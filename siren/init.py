import math
import torch
from torch.nn.init import _calculate_correct_fan

# 对输入的张量进行Siren初始化，使用均匀分布
def siren_uniform_(tensor: torch.Tensor, mode: str = 'fan_in', c: float = 6):

    fan = _calculate_correct_fan(tensor, mode)
    # 计算标准差Calculate uniform bounds from standard deviation
    std = 1 / math.sqrt(fan)
    # 根据给定的常数c和标准差std计算上下界，即bound = math.sqrt(c) * std。这里使用常数c乘以标准差来计算上下界
    bound = math.sqrt(c) * std
    with torch.no_grad():#不进行梯度计算
        return tensor.uniform_(-bound, bound)