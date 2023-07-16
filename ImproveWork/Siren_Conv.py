import torch
import torch.nn as nn
import numpy as np

"""
------------------------Siren+Conv NetWork------------------------------
In this example, we add two Conv2d layers as the front end of the SIREN model, 
and then flatten its output to 1D. Next, we added a linear layer for subsequent processing 
and used the Tanh activation function in the final output layer. Note that the shape of the input data is (batch_size, channels, height, width) 
and that the input/output sizes of the Conv2d layer and the linear layer need to match correctly

But the result displayed is not ideal
在这个示例中，我们添加了两个Conv2d层作为SIREN模型的前端，然后将其输出展平到一维。接下来，我们添加了线性层进行后续处理，并在最后输出层使用了Tanh激活函数。
请注意，输入数据的形状是(batch_size, channels, height, width)，并且Conv2d层和线性层的输入/输出大小需要正确匹配
"""


class SIREN(nn.Module):
    def __init__(self, conv_channels, conv_kernel_size, conv_stride, conv_padding, linear_layers, in_channels, out_features, w0, w0_initial, initializer='siren', c=6, weight_decay=1e-5):
        super(SIREN, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()

        conv_in_channels = in_channels
        for channels, kernel_size, stride, padding in zip(conv_channels, conv_kernel_size, conv_stride, conv_padding):
            self.conv_layers.append(nn.Conv2d(conv_in_channels, channels, kernel_size, stride=stride, padding=padding))
            conv_in_channels = channels

        self.linear_layers.append(nn.Linear(conv_channels[-1] * 64 * 64, linear_layers[0]))
        self.linear_layers.append(nn.SiLU())
        for i in range(1, len(linear_layers)):
            self.linear_layers.append(nn.Linear(linear_layers[i-1], linear_layers[i]))
            self.linear_layers.append(nn.SiLU())

        self.linear_layers.append(nn.Linear(linear_layers[-1], out_features))
        self.linear_layers.append(nn.Tanh())

        self.w0 = w0
        self.weight_decay = weight_decay

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)  # flatten the tensor

        for layer in self.linear_layers:
            if isinstance(layer, nn.Linear):
                if self.weight_decay > 0:
                    weight_decay = self.weight_decay * (self.w0 / np.sqrt(layer.weight.size(1)))
                    x = x + weight_decay * torch.sum(torch.square(layer.weight))
            x = layer(x)

        return self.w0 * x
    """
    SIREN(
  (conv_layers): ModuleList(
    (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (linear_layers): ModuleList(
    (0): Linear(in_features=131072, out_features=256, bias=True)
    (1): SiLU()
    (2): Linear(in_features=256, out_features=256, bias=True)
    (3): SiLU()
    (4): Linear(in_features=256, out_features=256, bias=True)
    (5): SiLU()
    (6): Linear(in_features=256, out_features=2, bias=True)
    (7): Tanh()
  )
)
    torch.Size([10, 3, 64, 64])
    torch.Size([10, 2])
    """

# defining the model
conv_channels = [16, 32]
conv_kernel_size = [3, 3]
conv_stride = [1, 1]
conv_padding = [1, 1]
linear_layers = [256, 256, 256]
in_channels = 3
out_features = 2
initializer = 'siren'
w0 = 1.0
w0_initial = 30.0
c = 6
weight_decay = 1e-5
model = SIREN(
    conv_channels, conv_kernel_size, conv_stride, conv_padding,
    linear_layers, in_channels, out_features, w0, w0_initial,
    initializer=initializer, c=c, weight_decay=weight_decay)

# defining the input
x = torch.rand(10, 3, 64, 64)  # assuming input size of 64x64 with 3 channels

# forward pass
y = model(x)

print(model)
print(x.shape)
print(y.shape)
