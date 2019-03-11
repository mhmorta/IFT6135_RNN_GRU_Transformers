from rnn_georgiy.models import RNNUnit
from torch import nn
import torch

x = torch.rand(3)

rnn = RNNUnit(5, 3, 2, 0.5)

h = torch.Tensor([-0.0074,  0.1360, -0.0136,  0.0266,  0.0090])
o, h = rnn.forward(x, h)

print(o, h)