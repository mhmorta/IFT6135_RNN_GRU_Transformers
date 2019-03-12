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


# Problem 1
class RNN(nn.Module):  # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.hidden_linear_layer_size = hidden_size * 2

        dropout_rate = 1 - dp_keep_prob

        self.embeddings = nn.Embedding(vocab_size, emb_size)

        hidden_modules = [RNNUnit(hidden_size, emb_size, self.hidden_linear_layer_size, dropout_rate)]
        if num_layers > 1:
            for i in range(num_layers - 1):
                hidden_modules.append(RNNUnit(hidden_size, hidden_size, self.hidden_linear_layer_size, dropout_rate))
        self.hidden_stack = nn.ModuleList(hidden_modules)

        self.output = nn.Linear(self.hidden_linear_layer_size, self.vocab_size)

        # self.

        # TODO ========================
        # Initialization of the parameters of the recurrent and fc layers.
        # Your implementation should support any number of stacked hidden layers
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding
        # modules, but not recurrent modules.
        #
        # To create a variable number of parameter tensors and/or nn.Modules
        # (for the stacked hidden layer), you may need to use nn.ModuleList or the
        # provided clones function (as opposed to a regular python list), in order
        # for Pytorch to recognize these parameters as belonging to this nn.Module
        # and compute their gradients automatically. You're not obligated to use the
        # provided clones function.

    def init_weights_uniform(self):
        # TODO ========================
        # Initialize all the weights uniformly in the range [-0.1, 0.1]
        # and all the biases to 0 (in place)

        # init hidden layer
        for module in self.hidden_stack:
            module.init_weights_uniform()

        # init output layer
        nn.init.uniform_(self.output.weight, -0.1, 0.1)
        # override the default init, reset to zeros
        nn.init.zeros_(self.output.bias)


    def init_hidden(self):
        # TODO ========================
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """
        # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        # TODO ========================
        # Compute the forward pass, using a nested python for loops.
        # The outer for loop should iterate over timesteps, and the
        # inner for loop should iterate over hidden layers of the stack.
        #
        # Within these for loops, use the parameter tensors and/or nn.modules you
        # created in __init__ to compute the recurrent updates according to the
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).

        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the
                  mini-batches in an epoch, except for the first, where the return
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details,
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """

        # NOTE: for the first implementation consider inputs as 1D tensor of shape (seq_len)
        # once we get it working we will implement batch mode
        embedded = self.embd(inputs)  # shape (seq_len, emb_size)

        for token_index in range(self.seq_len):
            x = embedded[token_index]  # shape (emb_size)
            # first hidden layer
            output, layer_hidden = self.hidden_stack[0].forward(x, hidden[0])

            for idx, layer in enumerate(self.hidden_stack[1:]):
                layer_hidden = hidden[idx+1]
                output, layer_hidden = layer.forward(output, layer_hidden)


        return self.output(output), hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        #
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution,
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """

        return
        # todo uncomment return samples
