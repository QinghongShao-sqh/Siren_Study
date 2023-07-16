from typing import List
import torch
import torch.nn as nn
from siren.init import siren_uniform_


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine激活函数，支持w0缩放参数。

        参数:
            - w0: 激活步骤中的w0，`act(x; w0) = sin(w0 * x)`。
            默认值为1.0
        类型:
            - w0: float, 可选
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播函数，将输入x进行Sine激活。

        参数:
            - x: 输入张量
        返回值:
            - 经过Sine激活后的张量
        """
        self._check_input(x)
        return torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        """检查输入是否为torch.Tensor类型。"""
        if not isinstance(x, torch.Tensor):
            raise TypeError('input to forward() must be torch.Tensor')




class SIREN(nn.Module):
    def __init__(self, layers: List[int], in_features: int,
                 out_features: int,
                 w0: float = 1.0,
                 w0_initial: float = 30.0,
                 bias: bool = True,
                 initializer: str = 'siren',
                 c: float = 6):

        super(SIREN, self).__init__()
        self._check_params(layers)
        self.layers = [nn.Linear(in_features, layers[0], bias=bias), Sine(
            w0=w0_initial)]

        for index in range(len(layers) - 1):
            self.layers.extend([
                nn.Linear(layers[index], layers[index + 1], bias=bias),
                Sine(w0=w0)
            ])

        self.layers.append(nn.Linear(layers[-1], out_features, bias=bias))
        self.network = nn.Sequential(*self.layers)

        if initializer is not None and initializer == 'siren':
            for m in self.network.modules():
                if isinstance(m, nn.Linear):
                    siren_uniform_(m.weight, mode='fan_in', c=c)

    @staticmethod
    def _check_params(layers):
        assert isinstance(layers, list), 'layers should be a list of ints'
        assert len(layers) >= 1, 'layers should not be empty'

    def forward(self, X):
        return self.network(X)


class SIREN(nn.Module):
    def __init__(self, layers: List[int], in_features: int,
                 out_features: int,
                 w0: float = 1.0,
                 w0_initial: float = 30.0,
                 bias: bool = True,
                 initializer: str = 'siren',
                 c: float = 6):
        """SIREN模型。

        参数:
            - layers: 包含每个线性层神经元数量的列表
            - in_features: 输入特征的数量
            - out_features: 输出特征的数量
            - w0: Sine激活函数的缩放参数，默认为1.0
            - w0_initial: 初始化Sine激活函数的缩放参数，默认为30.0
            - bias: 是否使用偏置，默认为True
            - initializer: 权重初始化方法，默认为'siren'
            - c: SIREN初始化方法中的缩放因子，默认为6
            ----------------------Engilish
            layers: a list containing the number of neurons in each linear layer
            in_features: the number of input features
            out_features: the number of output features
            w0: the scaling parameter for the Sine activation function, defaults to 1.0
            w0_initial: the scaling parameter for initializing the Sine activation function, defaults to 30.0
            bias: whether to use bias or not, defaults to True
            initializer: the weight initialization method, defaults to 'siren'
            c: the scaling factor used to compute the bound in the SIREN initializer, defaults to 6
        """
        super(SIREN, self).__init__()
        self._check_params(layers)
        self.layers = [nn.Linear(in_features, layers[0], bias=bias), Sine(w0=w0_initial)]
        # 通过for循环来构建网络层
        for index in range(len(layers) - 1):
            self.layers.extend([
                nn.Linear(layers[index], layers[index + 1], bias=bias),
                Sine(w0=w0)
            ])

        self.layers.append(nn.Linear(layers[-1], out_features, bias=bias))
        self.network = nn.Sequential(*self.layers)

        if initializer is not None and initializer == 'siren':
            for m in self.network.modules():
                if isinstance(m, nn.Linear):
                    siren_uniform_(m.weight, mode='fan_in', c=c)

    @staticmethod
    def _check_params(layers):
        """检查layers参数是否合法。"""
        assert isinstance(layers, list), 'layers should be a list of ints'
        assert len(layers) >= 1, 'layers should not be empty'

    def forward(self, X):
        """前向传播函数。"""
        return self.network(X)

