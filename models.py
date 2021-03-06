import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).
    
    inputs: 
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class RNNUnit(nn.Module):
    """
    Class RNNUnit represents a single RNN recurrent layer cell
    """

    def __init__(self, hidden_size, input_size):
        super(RNNUnit, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.non_linearity = nn.Tanh()
        self.hiddens = None

    def forward(self, inputs, hidden):
        """
        :param inputs: shape: (batch_size, input_size)
        :param hidden: previous hidden state (t-1) 2D tensor of shape (batch_size, hidden_size)
        :return: output: shape: (batch_size, input_size), hidden: new hidden state : shape (batch_size, hidden_size)
        """
        hidden = self.h2h(hidden) + self.i2h(inputs)
        hidden = self.non_linearity(hidden)
        # we only set hiddens to [] during the 5.2 experiments via model.init_hidden_state_list()
        if self.hiddens is not None:
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

        nn.init.uniform_(self.h2h.weight, -k, k)
        nn.init.uniform_(self.h2h.bias, -k, k)


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

        # actual dropout rate
        dropout_rate = 1 - dp_keep_prob
        self.dropout = nn.Dropout(dropout_rate)

        self.embeddings = nn.Embedding(vocab_size, emb_size)

        # create stack of hidden layers as modules list
        hidden_modules = [RNNUnit(hidden_size, emb_size)]
        for _ in range(max(0, num_layers - 1)):
            hidden_modules.append(RNNUnit(hidden_size, hidden_size))
        self.hidden_stack = nn.ModuleList(hidden_modules)

        self.output = nn.Linear(hidden_size, self.vocab_size)

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
        self.init_weights_uniform()

    def init_weights_uniform(self):
        # TODO ========================
        # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        # and output biases to 0 (in place). The embeddings should not use a bias vector.
        # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly
        # in the range [-k, k] where k is the square root of 1/hidden_size

        # initialize output layer
        nn.init.uniform_(self.output.weight, -0.1, 0.1)

        # override the default init, reset to zeros
        nn.init.zeros_(self.output.bias)

        # initialize embeddings weight
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)

        for rnn_unit in self.hidden_stack:
            rnn_unit.init_weights_uniform()

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

        embedded = self.embeddings(inputs)  # shape (seq_len, batch_size, emb_size)

        # will collect the output
        logits = []

        for token_batch in embedded:  # shape (batch_size, emb_size)

            # first hidden layer is connected to the embeddings layer
            layer_output = self.dropout(token_batch)

            # collect the output of hidden layers
            hidden_list = []

            # all other hidden layers: 2, 3 ...
            for idx, layer in enumerate(self.hidden_stack):

                # s_{t-1}
                layer_hidden_prev = hidden[idx]

                # apply the hidden layer
                layer_hidden = layer(layer_output, layer_hidden_prev)
                # apply dropout to the vertical outputs
                layer_output = self.dropout(layer_hidden)  # shape (batch_size, hidden_size)

                # save output
                hidden_list.append(layer_hidden)

            # update hidden state after processing all layers for a single batch (num_layers, seq_len, hidden_size)
            hidden = torch.stack(hidden_list)

            # collect outputs of the last layer
            logits.append(self.output(layer_output))

        # transform list of outputs to a tensor (seq_len, batch_size, vocab_size)
        logits = torch.stack(logits)
        return logits, hidden

    def init_hidden_state_list(self):
        # we use it for 5.2
        for unit in self.hidden_stack:
            unit.hiddens = []

    def generate(self, input, hidden, generated_seq_len, temperature=1):
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
        samples = []
        # samples.append(input)
        for i in range(generated_seq_len):
            output, hidden = self.generate_batch(input, hidden, temperature)
            samples.append(output)
            input = output
        return torch.stack(samples) # shape (generated_seq_len, batch_size)

    def generate_batch(self, input, hidden, temperature):
        """
        Forward pass for a minibatch
        without applying dropout

        :param input: A mini-batch of input tokens
                            shape: (batch_size)
        :param hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
        :param temperature: Control randomness of the generated samples, from 0 to 1
                            https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally
        :return: - output: A mini-batch of input tokens
                            shape: (batch_size)
                 - hidden: updated states of hidden layers
        """

        embedded = self.dropout(self.embeddings(input))  # shape (batch_size, emb_size)

        # first hidden layer is connected to the embeddings layer
        layer_output = embedded

        hidden_list = []

        for idx, layer in enumerate(self.hidden_stack):

            layer_hidden_prev = hidden[idx]

            # apply the hidden layer
            layer_hidden = layer(layer_output, layer_hidden_prev)

            # vertical outputs
            layer_output = self.dropout(layer_hidden)  # shape (batch_size, hidden_size)

            # save output
            hidden_list.append(layer_hidden)

        # outputs of the last layer
        distribution = torch.softmax(self.output(layer_output) / temperature, dim=1)

        # sample from distribution
        output = torch.multinomial(distribution, 1).squeeze() # shape (batch_size)

        # update hidden state after processing all layers for a single batch (num_layers, hidden_size)
        hidden = torch.stack(hidden_list)
        return output, hidden


# Problem 2
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
        self.hiddens = None

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
        # we only set hiddens to [] during the 5.2 experiments via model.init_hidden_state_list()
        if self.hiddens is not None:
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

        # actual dropout rate
        dropout_rate = 1 - dp_keep_prob
        self.dropout = nn.Dropout(dropout_rate)

        self.embeddings = nn.Embedding(vocab_size, emb_size)

        # create stack of hidden layers as modules list
        hidden_modules = [GRUUnit(hidden_size, emb_size)]
        for _ in range(max(0, num_layers - 1)):
            hidden_modules.append(GRUUnit(hidden_size, hidden_size))
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

        for token_batch in embedded:  # shape (batch_size, emb_size)

            # first hidden layer is connected to the embeddings layer
            layer_output = self.dropout(token_batch)

            # collect the output of hidden layers
            hidden_list = []

            # all other hidden layers: 1, 2, 3 ...
            for idx, layer in enumerate(self.hidden_stack):
                # s_{t-1}
                layer_hidden_prev = hidden[idx]

                # apply the hidden layer
                layer_hidden = layer(layer_output, layer_hidden_prev)
                # apply dropout to the vertical outputs
                layer_output = self.dropout(layer_hidden)  # shape (batch_size, hidden_size)

                # save output
                hidden_list.append(layer_hidden)

            # update hidden state after processing all layers for a single batch (num_layers, seq_len, hidden_size)
            hidden = torch.stack(hidden_list)

            # collect outputs of the last layer
            logits.append(self.output(layer_output))

        # transform list of outputs to a tensor (seq_len, batch_size, vocab_size)
        logits = torch.stack(logits)
        return logits, hidden

    def generate(self, inputs, hidden, generated_seq_len, temperature=1):
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
        samples = []
        # samples.append(input)
        for i in range(generated_seq_len):
            output, hidden = self.generate_batch(inputs, hidden, temperature)
            samples.append(output)
            inputs = output
        return torch.stack(samples)  # shape (generated_seq_len, batch_size)

    def init_hidden_state_list(self):
        # we use it for 5.2
        for unit in self.hidden_stack:
            unit.hiddens = []

    def generate_batch(self, inputs, hidden, temperature):
        """
        Forward pass for a minibatch
        without applying dropout

        :param inputs: A mini-batch of input tokens
                            shape: (batch_size)
        :param hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
        :param temperature: Control randomness of the generated samples, from 0 to 1
                            https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally
        :return: - output: A mini-batch of input tokens
                            shape: (batch_size)
                 - hidden: updated states of hidden layers
        """

        embedded = self.dropout(self.embeddings(inputs))  # shape (batch_size, emb_size)

        # first hidden layer is connected to the embeddings layer
        layer_output = embedded

        hidden_list = []

        for idx, layer in enumerate(self.hidden_stack):

            layer_hidden_prev = hidden[idx]

            # apply the hidden layer
            layer_hidden = layer(layer_output, layer_hidden_prev)

            # vertical outputs
            layer_output = self.dropout(layer_hidden)  # shape (batch_size, hidden_size)

            # save output
            hidden_list.append(layer_hidden)

        # outputs of the last layer
        distribution = torch.softmax(self.output(layer_output) / temperature, dim=1)

        output = torch.multinomial(distribution, 1).squeeze() # shape (batch_size, vocab_size)

        # update hidden state after processing all layers for a single batch (num_layers, hidden_size)
        hidden = torch.stack(hidden_list)
        return output, hidden


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#----------------------------------------------------------------------------------



#----------------------------------------------------------------------------------

def ScaledDotProductAttention(query, key, value, mask = None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units 
        self.h = n_heads
        self.attention = None

        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout

        k = math.sqrt(1 / n_units)


        self.linears = clones(nn.Linear(n_units, n_units), 4)
        for my_linear in self.linears:
            nn.init.uniform_(my_linear.weight, -k, k)
            nn.init.uniform_(my_linear.bias, -k, k)

        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.
        if mask is not None:
        	mask = mask.unsqueeze(1)

        bs = query.size(0)

        q, k, v = [l(x).view(bs, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attention = ScaledDotProductAttention(q, k, v, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(bs, -1, self.n_units)
        return self.linears[-1](x) # size: (batch_size, seq_len, self.n_units)




#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)>
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

