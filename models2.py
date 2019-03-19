import torch
import torch.nn as nn

import math, copy
from torch.autograd import Variable
from collections import defaultdict


class GRUUnit(nn.Module):

    def __init__(self, hidden_size, input_size):
        super(GRUUnit, self).__init__()
        self.hidden_size = hidden_size

        self.i2r = nn.Linear(input_size, hidden_size)
        self.i2z = nn.Linear(input_size, hidden_size)

        self.h2r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h2z = nn.Linear(hidden_size, hidden_size, bias=False)

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.hiddens = []

    def forward(self, inputs, hidden):
        """
        :param inputs: shape: (batch_size, input_size)
        :param hidden: previous hidden state (t-1) 2D tensor of shape (batch_size, hidden_size)
        :return: output: shape: (batch_size, input_size), hidden: new hidden state : shape (batch_size, hidden_size)
        """
        r = self.sigmoid(self.i2r(inputs) + self.h2r(hidden))
        z = self.sigmoid(self.i2z(inputs) + self.h2z(hidden))
        h1 = self.tanh(self.i2h(inputs) + r * self.h2h(hidden))
        hidden = (1 - z) * h1 + z * hidden
        self.hiddens.append(hidden)
        return hidden

    def init_weights_uniform(self):
        # TODO ========================
        # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        # and output biases to 0 (in place). The embeddings should not use a bias vector.
        # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly
        # in the range [-k, k] where k is the square root of 1/hidden_size

        k = math.sqrt(1/self.hidden_size)

        nn.init.uniform_(self.i2h.weight, -k, k)
        nn.init.uniform_(self.i2r.weight, -k, k)
        nn.init.uniform_(self.i2z.weight, -k, k)

        nn.init.uniform_(self.i2h.bias, -k, k)
        nn.init.uniform_(self.i2r.bias, -k, k)
        nn.init.uniform_(self.i2z.bias, -k, k)

        nn.init.uniform_(self.h2h.weight, -k, k)
        nn.init.uniform_(self.h2r.weight, -k, k)
        nn.init.uniform_(self.h2z.weight, -k, k)


class GRU(nn.Module):  # Implement a stacked GRU RNN
    """
    Follow the same instructions as for RNN (above), but use the equations for
    GRU, not Vanilla RNN.
    """

    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.hiddens = []

        # actual dropout rate
        dropout_rate = 1 - dp_keep_prob
        self.dropout = nn.Dropout(dropout_rate)

        self.embeddings = nn.Embedding(vocab_size, emb_size)

        # create stack of hidden layers as modules list
        gru_unit = GRUUnit(hidden_size, emb_size)
        hidden_modules = [gru_unit]

        for i in range(max(0, num_layers - 1)):
            gru_unit = GRUUnit(hidden_size, hidden_size)
            hidden_modules.append(gru_unit)
        self.hidden_stack = nn.ModuleList(hidden_modules)

        self.output = nn.Linear(hidden_size, self.vocab_size)

        self.init_weights_uniform()

    def init_weights_uniform(self):

        # initialize output layer
        nn.init.uniform_(self.output.weight, -0.1, 0.1)

        # override the default init, reset to zeros
        nn.init.zeros_(self.output.bias)

        # initialize embeddings weight
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)

        for gru_unit in self.hidden_stack:
            gru_unit.init_weights_uniform()

    def init_hidden(self):

        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):

        embedded = self.embeddings(inputs)  # shape (seq_len, batch_size, emb_size)

        # will collect the output
        logits = []

        for timestep_idx, token_batch in enumerate(embedded):  # shape (batch_size, emb_size)

            # first hidden layer is connected to the embeddings layer
            layer_output = self.dropout(token_batch)

            # collect the output of hidden layers
            hidden_list = []

            # all hidden layers: 1, 2, 3 ...
            for idx, layer in enumerate(self.hidden_stack):
                # s_{t-1}
                layer_hidden_prev = hidden[idx]

                # apply the hidden layer
                layer_hidden = layer(layer_output, layer_hidden_prev)

                # apply dropout to the vertical outputs
                layer_output = self.dropout(layer_hidden)  # shape (batch_size, hidden_size)

                # save output
                hidden_list.append(layer_hidden)

                self.hiddens.append(layer_hidden)
            # update hidden state after processing all layers for a single batch
            # (num_layers, seq_len, hidden_size)

            hidden = torch.stack(hidden_list)
            # collect outputs of the last layer
            logits.append(self.output(layer_output))

        # transform list of outputs to a tensor (seq_len, batch_size, vocab_size)
        logits = torch.stack(logits)
        return logits, hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        samples = []
        return samples