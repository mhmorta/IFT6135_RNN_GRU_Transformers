import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import torchviz
from torch.autograd import Variable
import torch
import argparse
import os

def plots(train_losses, val_losses, train_ppls, val_ppls, experiment ):
    plt.figure(1)
    plt.plot(range(len(train_losses)), train_losses, label="Train Losses")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Losses")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.suptitle(experiment, fontsize=10)
    plt.legend()

    plt.figure(2)
    plt.plot(range(len(train_ppls)), train_ppls, label="Train PPL" )
    plt.plot(range(len(val_ppls)), val_ppls, label="Validation PPL" )
    plt.xlabel("epochs")
    plt.ylabel("PPL")
    plt.suptitle(experiment, fontsize=10)
    plt.legend()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Visualization Argument Parser')

    parser.add_argument('--save_dir', type=str, default='',
                        help='path to save the experimental config, logs, model \
                        This is automatically generated based on the command line \
                        arguments you pass and only needs to be set if you want a \
                        custom dir name')

    args = parser.parse_args()
    argsdict = args.__dict__

    lc_path = os.path.join(args.save_dir, 'learning_curves.npy')
    experiment = str.split(args.save_dir, "/")[1]
    print
    return lc_path, experiment;

lc_path, experiment = parse_args()
x = np.load(lc_path)[()]

train_ppls, val_ppls, train_losses, val_losses = [x['train_ppls'], x['val_ppls'], x['train_losses'], x['val_losses'] ]
plots(train_losses, val_losses, train_ppls, val_ppls, experiment)

