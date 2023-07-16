import torch
import torch.nn as nn
import numpy as np

"""
------------------------Siren+L2 NetWork------------------------------
In the example, a weight_decay parameter is added and an L2 regularization term for the weights is calculated 
after each Linear layer. The value of weight_decay can be adjusted as needed

But the result displayed is not ideal
在示例中，添加了一个weight_decay参数，并在每个Linear层后计算权重的L2正则化项。可以根据需要调整weight_decay的值
"""


class SIREN(nn.Module):
    def __init__(self, layers, in_features, out_features, w0, w0_initial, initializer='siren', c=6, weight_decay=1e-5):
        super(SIREN, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(len(layers)):
            self.layers.append(nn.Linear(in_features if i == 0 else layers[i - 1], layers[i]))
            self.layers.append(nn.SiLU())

        self.layers.append(nn.Linear(layers[-1], out_features))
        self.layers.append(nn.Tanh())

        self.w0 = w0

        self.weight_decay = weight_decay

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if self.weight_decay > 0:
                    weight_decay = self.weight_decay * (self.w0 / np.sqrt(self.layers[i].weight.size(1)))
                    x = x + weight_decay * torch.sum(torch.square(self.layers[i].weight))
                x = layer(x)
            else:
                x = layer(x)
        return self.w0 * x

    """
 SIREN(
  (layers): ModuleList(
    (0): Linear(in_features=2, out_features=256, bias=True)
    (1): SiLU()
    (2): Linear(in_features=256, out_features=256, bias=True)
    (3): SiLU()
    (4): Linear(in_features=256, out_features=256, bias=True)
    (5): SiLU()
    (6): Linear(in_features=256, out_features=256, bias=True)
    (7): SiLU()
    (8): Linear(in_features=256, out_features=256, bias=True)
    (9): SiLU()
    (10): Linear(in_features=256, out_features=3, bias=True)
    (11): Tanh()
  )
)
tensor([[0.1971, 0.4029],
        [0.5094, 0.7426],
        [0.5364, 0.8811],
        [0.2723, 0.2498],
        [0.2011, 0.8822],
        [0.8838, 0.4911],
        [0.0828, 0.2051],
        [0.2965, 0.3034],
        [0.1014, 0.0236],
        [0.6359, 0.8607]])
tensor([[ 0.0017, -0.0276,  0.0572],
        [ 0.0010, -0.0273,  0.0572],
        [ 0.0008, -0.0272,  0.0572],
        [ 0.0018, -0.0274,  0.0570],
        [ 0.0011, -0.0276,  0.0576],
        [ 0.0011, -0.0268,  0.0566],
        [ 0.0021, -0.0276,  0.0572],
        [ 0.0018, -0.0274,  0.0571],
        [ 0.0022, -0.0275,  0.0571],
        [ 0.0008, -0.0271,  0.0571]], grad_fn=<MulBackward0>)
 
    """



# # defining the model
# layers = [256, 256, 256, 256, 256]
# in_features = 2
# out_features = 3
# initializer = 'siren'
# w0 = 1.0
# w0_initial = 30.0
# c = 6
# weight_decay = 1e-5
# model = SIREN(
#     layers, in_features, out_features, w0, w0_initial,
#     initializer=initializer, c=c, weight_decay=weight_decay)
#
# # defining the input
# x = torch.rand(10, 2)
#
# # forward pass
# y = model(x)
#
# print(model)
# print(x)
# print(y)
