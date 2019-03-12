import torch
import torch.nn as nn


class RNNUnit(nn.Module):
    """
    Class RNNUnit represents a single RNN recurrent layer cell
    """

    def __init__(self, hidden_size, input_size, dropout_rate):
        super(RNNUnit, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.non_linearity = nn.Tanh()

    def forward(self, inputs, hidden):
        """
        :param inputs: shape: (batch_size, input_size)
        :param hidden: previous hidden state (t-1) 2D tensor of shape (batch_size, hidden_size)
        :return: output: shape: (batch_size, input_size), hidden: new hidden state : shape (batch_size, hidden_size)
        """
        hidden = self.h2h(hidden) + self.i2h(inputs)
        hidden = self.non_linearity(hidden)

        # apply dropout to the vertical outputs
        outputs = self.dropout(hidden)  # shape (batch_size, hidden_size)

        return outputs, hidden


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
        hidden_modules = [RNNUnit(hidden_size, emb_size, dropout_rate)]
        if num_layers > 1:
            for i in range(num_layers - 1):
                hidden_modules.append(RNNUnit(hidden_size, hidden_size, dropout_rate))
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

        for token_batch in embedded: # shape (batch_size, emb_size)

            # first hidden layer is connected to the embeddings layer

            # output of the previous token x_{t-1}
            layer_hidden_prev = hidden[0]

            # apply the hidden layer, use embeddings as input
            layer_output, layer_hidden = self.hidden_stack[0](self.dropout(token_batch), layer_hidden_prev)

            # collect the output of hidden layers
            hidden_list = [layer_hidden]

            # all other hidden layers: 2, 3 ...
            for idx, layer in enumerate(self.hidden_stack[1:]):

                # output of the previous token x_{t-1}
                layer_hidden_prev = hidden[idx + 1]

                # apply the hidden layer
                layer_output, layer_hidden = layer(layer_output, layer_hidden_prev)

                # save output
                hidden_list.append(layer_hidden)

            # update hidden state after processing all layers for a single batch
            hidden = torch.stack(hidden_list)

            # collect outputs of the last layer
            logits.append(self.output(layer_output))

        # transform list of outputs to a tensor
        logits = torch.stack(logits)
        return logits, hidden

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
