import torch
import torch.nn as nn
import numpy as np

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

# defining the model
layers = [256, 256, 256, 256, 256]
in_features = 2
out_features = 3
initializer = 'siren'
w0 = 1.0
w0_initial = 30.0
c = 6
weight_decay = 1e-5
model = SIREN(
    layers, in_features, out_features, w0, w0_initial,
    initializer=initializer, c=c, weight_decay=weight_decay)

# defining the input
x = torch.rand(10, 2)

# forward pass
y = model(x)

print(model)
print(x)
print(y)
