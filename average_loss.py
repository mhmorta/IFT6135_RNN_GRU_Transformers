#!/bin/python
# coding: utf-8

# Code outline/scaffold for
# ASSIGNMENT 2: RNNs, Attention, and Optimization
# By Tegan Maharaj, David Krueger, and Chin-Wei Huang
# IFT6135 at University of Montreal
# Winter 2019
#
# based on code from:
#    https://github.com/deeplearningathome/pytorch-language-model/blob/master/reader.py
#    https://github.com/ceshine/examples/blob/master/word_language_model/main.py
#    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_word_ptb.py
#    https://github.com/harvardnlp/annotated-transformer
#
# GENERAL INSTRUCTIONS:
#    - ! IMPORTANT!
#      Unless we're otherwise notified we will run exactly this code, importing
#      your models from models.py to test them. If you find it necessary to
#      modify or replace this script (e.g. if you are using TensorFlow), you
#      must justify this decision in your report, and contact the TAs as soon as
#      possible to let them know. You are free to modify/add to this script for
#      your own purposes (e.g. monitoring, plotting, further hyperparameter
#      tuning than what is required), but remember that unless we're otherwise
#      notified we will run this code as it is given to you, NOT with your
#      modifications.
#    - We encourage you to read and understand this code; there are some notes
#      and comments to help you.
#    - Typically, all of your code to submit should be written in models.py;
#      see further instructions at the top of that file / in TODOs.
#          - RNN recurrent unit
#          - GRU recurrent unit
#          - Multi-head attention for the Transformer
#    - Other than this file and models.py, you will probably also write two
#      scripts. Include these and any other code you write in your git repo for
#      submission:
#          - Plotting (learning curves, loss w.r.t. time, gradients w.r.t. hiddens)
#          - Loading and running a saved model (computing gradients w.r.t. hiddens,
#            and for sampling from the model)

# PROBLEM-SPECIFIC INSTRUCTIONS:
#    - For Problems 1-3, paste the code for the RNN, GRU, and Multi-Head attention
#      respectively in your report, in a monospace font.
#    - For Problem 4.1 (model  comparison), the hyperparameter settings you should run are as follows:
#          --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
#          --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
#          --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
#    - In those experiments, you should expect to see approximately the following
#      perplexities:
#                  RNN: train:  120  val: 157
#                  GRU: train:   65  val: 104
#          TRANSFORMER:  train:  67  val: 146
#    - For Problem 4.2 (exploration of optimizers), you will make use of the
#      experiments from 4.1, and should additionally run the following experiments:
#          --model=RNN --optimizer=SGD --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
#          --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
#          --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9
#          --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35
#          --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
#          --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9
#    - For Problem 4.3 (exloration of hyperparameters), do your best to get
#      better validation perplexities than the settings given for 4.1. You may
#      try any combination of the hyperparameters included as arguments in this
#      script's ArgumentParser, but do not implement any additional
#      regularizers/features. You may (and will probably want to) run a lot of
#      different things for just 1-5 epochs when you are trying things out, but
#      you must report at least 3 experiments on each architecture that have run
#      for at least 40 epochs.
#    - For Problem 5, perform all computations / plots based on saved models
#      from Problem 4.1. NOTE this means you don't have to save the models for
#      your exploration, which can make things go faster. (Of course
#      you can still save them if you like; just add the flag --save_best).
#    - For Problem 5.1, you can modify the loss computation in this script
#      (search for "LOSS COMPUTATION" to find the appropriate line. Remember to
#      submit your code.
#    - For Problem 5.3, you must implement the generate method of the RNN and
#      GRU.  Implementing this method is not considered part of problems 1/2
#      respectively, and will be graded as part of Problem 5.3


import os
import torch
import torch.nn
import torch.nn as nn
import numpy
import utils

np = numpy

# NOTE ==============================================
# This is where your models are imported
from models import RNN, GRU
from models import make_model as TRANSFORMER

##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

results_dir = "results/4_1"
for dir_name in [x[0] for x in os.walk(results_dir)]:
    if dir_name == results_dir:
        continue

    #if dir_name not in [os.path.join(results_dir, 'RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_save_dir=output_timestep_loss=true_0')]:
    #    continue
    args = utils.load_model_config(dir_name)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)


    ###############################################################################
    #
    # LOADING & PROCESSING
    #
    ###############################################################################

    device = utils.get_device()

    # LOAD DATA
    print('Loading data from ' + args.data)
    raw_data = utils.ptb_raw_data(data_path=args.data)
    _, valid_data, _, word_to_id, id_2_word = raw_data
    vocab_size = len(word_to_id)
    print('  vocabulary size: {}'.format(vocab_size))

    ###############################################################################
    #
    # MODEL SETUP
    #
    ###############################################################################

    # NOTE ==============================================
    # This is where your model code will be called. You may modify this code
    # if required for your implementation, but it should not typically be necessary,
    # and you must let the TAs know if you do so.
    if args.model == 'RNN':
        model = RNN(emb_size=args.emb_size, hidden_size=args.hidden_size,
                    seq_len=args.seq_len, batch_size=args.batch_size,
                    vocab_size=vocab_size, num_layers=args.num_layers,
                    dp_keep_prob=args.dp_keep_prob)
    elif args.model == 'GRU':
        model = GRU(emb_size=args.emb_size, hidden_size=args.hidden_size,
                    seq_len=args.seq_len, batch_size=args.batch_size,
                    vocab_size=vocab_size, num_layers=args.num_layers,
                    dp_keep_prob=args.dp_keep_prob)
    elif args.model == 'TRANSFORMER':
        if args.debug:  # use a very small model
            model = TRANSFORMER(vocab_size=vocab_size, n_units=16, n_blocks=2)
        else:
            # Note that we're using num_layers and hidden_size to mean slightly
            # different things here than in the RNNs.
            # Also, the Transformer also has other hyperparameters
            # (such as the number of attention heads) which can change it's behavior.
            model = TRANSFORMER(vocab_size=vocab_size, n_units=args.hidden_size,
                                n_blocks=args.num_layers, dropout=1. - args.dp_keep_prob)
            # these 3 attributes don't affect the Transformer's computations;
        # they are only used in run_epoch
        model.batch_size = args.batch_size
        model.seq_len = args.seq_len
        model.vocab_size = vocab_size
    else:
        print("Model type not recognized.")

    model = model.to(device)

    print("###Loading the model from best_params.pt###")
    model.load_state_dict(torch.load('{}/best_params.pt'.format(args['experiment_path']), map_location=device))

    # LOSS FUNCTION
    loss_fn = nn.CrossEntropyLoss()

    def run_epoch(model, data):
        """
        One epoch of training/validation (depending on flag is_train).
        """
        model.eval()
        if args.model != 'TRANSFORMER':
            hidden = model.init_hidden()
            hidden = hidden.to(device)
        costs = 0.0
        iters = 0
        losses = []
        seq_losses = []

        # LOOP THROUGH MINIBATCHES
        for step, (x, y) in enumerate(utils.ptb_iterator(data, model.batch_size, model.seq_len)):
            if args.model == 'TRANSFORMER':
                batch = utils.Batch(torch.from_numpy(x).long().to(device))
                model.zero_grad()
                outputs = model.forward(batch.data, batch.mask).transpose(1, 0)
                # print ("outputs.shape", outputs.shape)
            else:
                inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)  # .cuda()
                model.zero_grad()
                hidden = utils.repackage_hidden(hidden)
                outputs, hidden = model(inputs, hidden)

            targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)  # .cuda()
            tt = torch.squeeze(targets.view(-1, model.batch_size * model.seq_len))

            # LOSS COMPUTATION
            # This line currently averages across all the sequences in a mini-batch
            # and all time-steps of the sequences.
            # For problem 5.3, you will (instead) need to compute the average loss
            # at each time-step separately.
            loss = loss_fn(outputs.contiguous().view(-1, model.vocab_size), tt)
            costs += loss.data.item() * model.seq_len
            losses.append(costs)
            iters += model.seq_len
            if args.debug:
                print(step, loss)
            for output, target in zip(outputs, targets):
                l = loss_fn(output, target)
                seq_losses.append(l.data.item())

        return np.exp(costs / iters), losses, seq_losses


    print("\n########## Running Main Loop ##########################")

    # RUN MODEL ON VALIDATION DATA
    val_ppl, val_loss, seq_loss = run_epoch(model, valid_data)

    print('args:', args)
    # LOC RESULTS
    log_str = 'val ppl: ' + str(val_ppl)
    if len(seq_loss) > 0:
        log_str += '\nAdditional data (modified)'
        log_str += '\nval_loss={}, \nseq_losses (len={}, sum={}): {}'.format(val_loss, len(seq_loss), sum(seq_loss), seq_loss)
    print(log_str)

    sl_path = os.path.join(args['experiment_path'], 'seq_loss.npy')
    print('\nDONE\n\nSaving seq_loss to ' + sl_path)
    np.save(sl_path, seq_loss)

