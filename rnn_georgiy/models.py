import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

class RNNUnit(nn.Module):

    def __init__(self, hidden_size, input_size, output_size, dropout_prob):
        super(RNNUnit, self).__init__()
        self.hidden_size = hidden_size
        self.Wx = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.Wh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.Wy = nn.Parameter(torch.zeros(output_size, hidden_size))
        self.by = nn.Parameter(torch.zeros(output_size, 1))
        self.bh = nn.Parameter(torch.zeros(hidden_size, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.tanh = nn.Tanh()

        self.init_weights_uniform()

    def forward(self, x, hidden=None):
        """
        :param x: 1D Tensor of shape: (emb_size)
        :param hidden: previous hidden state (t-1) 1D tensor of shape (hidden_size)
        :return: output: 1D Tensor: (output_size), 1D Tensor size: (hidden size)
        """
        if hidden is None:
            hidden = self.init_hidden()

        # [x, h_t-1, 1]
        combined_input = torch.cat((torch.cat((x, hidden), 0), torch.ones(1)), 0)

        # [Wx, Wh, bh]
        combined_input_hidden_params = torch.cat((torch.cat((self.Wx, self.Wh), 1), self.bh), 1)

        # [Wy, by]
        combined_output_params = torch.cat((self.Wy, self.by), 1)

        # fully connected linear layer with dropout
        # [Wy, by] * [h_t-1, 1]
        hidden_augmented = torch.cat((self.dropout(hidden), torch.ones(1)), 0)
        output = torch.matmul(combined_output_params, hidden_augmented)

        # update hidden
        # h_t = tanh(Wx * x + Wh * h_t-1 + bh) = tanh([Wx, Wh, bh] * [x, h_t-1, 1])
        hidden = self.tanh(torch.matmul(combined_input_hidden_params, combined_input))

        # apply dropout to the output and return
        return self.dropout(output), hidden

    def init_hidden(self):
        return torch.zeros(self.hidden_size)

    def init_weights_uniform(self):
        nn.init.uniform_(self.Wx, -0.1, 0.1)
        nn.init.uniform_(self.Wh, -0.1, 0.1)
        nn.init.uniform_(self.Wy, -0.1, 0.1)
