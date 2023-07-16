import torch
import torch.nn as nn
from torch.nn.functional import mse_loss



"""
------------------------Nerf+Siren NetWork------------------------------
In this code, I try to improve the MLP neural network structure in the Nerf neural radiation field 
by adding the Siren periodic activation function module.
The specific network structure can be seen below.

But the result displayed is not ideal

在这段代码中，我尝试改进Nerf神经辐射领域的MLP神经网络结构
通过添加Siren周期性激活功能模块。具体网络结构见下图。
"""


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, is_last=False):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.is_first = is_first
        self.is_last = is_last

    def forward(self, x):
        out = torch.sin(self.linear(x))
        if self.is_first:
            out = out * 30.0
        if self.is_last:
            out = out / 30.0
        return out

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4]):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        self.xyz_encoding = nn.Sequential(
            SineLayer(in_channels_xyz, W, is_first=True),
            *[SineLayer(W, W) for _ in range(D-1)]
        )
        self.xyz_encoding_final = SineLayer(W, W)

        self.dir_encoding = nn.Sequential(
            nn.Linear(W + in_channels_dir, W // 2),
            nn.ReLU(True)
        )

        self.density = nn.Sequential(
            SineLayer(W // 2, 1)
        )

        self.rgb = nn.Sequential(
            SineLayer(W // 2, 3)
        )

    def forward(self, x):
        input_xyz, input_dir = torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)

        xyz_encoding = self.xyz_encoding(input_xyz)
        xyz_encoding_final = self.xyz_encoding_final(xyz_encoding)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)

        density = self.density(dir_encoding)
        rgb = self.rgb(dir_encoding)
        return density, rgb
    """
    NeRF(
  (xyz_encoding): Sequential(
    (0): SineLayer(
      (linear): Linear(in_features=63, out_features=256, bias=True)
    )
    (1): SineLayer(
      (linear): Linear(in_features=256, out_features=256, bias=True)
    )
    (2): SineLayer(
      (linear): Linear(in_features=256, out_features=256, bias=True)
    )
    (3): SineLayer(
      (linear): Linear(in_features=256, out_features=256, bias=True)
    )
    (4): SineLayer(
      (linear): Linear(in_features=256, out_features=256, bias=True)
    )
    (5): SineLayer(
      (linear): Linear(in_features=256, out_features=256, bias=True)
    )
    (6): SineLayer(
      (linear): Linear(in_features=256, out_features=256, bias=True)
    )
    (7): SineLayer(
      (linear): Linear(in_features=256, out_features=256, bias=True)
    )
  )
  (xyz_encoding_final): SineLayer(
    (linear): Linear(in_features=256, out_features=256, bias=True)
  )
  (dir_encoding): Sequential(
    (0): Linear(in_features=283, out_features=128, bias=True)
    (1): ReLU(inplace=True)
  )
  (rgb): Sequential(
    (0): SineLayer(
      (linear): Linear(in_features=128, out_features=3, bias=True)
    )
  )
)
torch.Size([1, 3])

Process finished with exit code 0

    """

# Test code
input_xyz = torch.randn(1, 63)
input_dir = torch.randn(1, 27)
model = NeRF()
print(model)
output_rgb = model(torch.cat([input_xyz, input_dir], -1))
print(output_rgb.shape)
