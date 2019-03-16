import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import torchviz
from torch.autograd import Variable
import torch
import argparse
import os
import re

def plots(train_losses, val_losses, train_ppls, val_ppls, epoch_times, experiment ):
    plt.figure(1)
    train_losses = np.array(train_losses)
    train_losses_index = [(i % 1327 == 0) for i in range(len(train_losses))]
    val_losses = np.array(val_losses)
    val_losses_index = [(i % 105 == 0) for i in range(len(val_losses))]

    epochs = range(40)
    plt.plot(epochs, train_losses[train_losses_index], label="Train Losses")
    plt.scatter(epochs, train_losses[train_losses_index], label="Train Losses")
    plt.plot(epochs, val_losses[val_losses_index], label="Validation Losses")
    plt.scatter(epochs, val_losses[val_losses_index], label="Validation Losses")

    plt.xlabel("Epoch number")
    plt.ylabel("Loss value")
    plt.xticks(epochs,rotation='vertical')

    plt.autoscale(enable=True, axis='both', tight=None)
    plt.suptitle(experiment, fontsize=10)
    plt.legend()

    plt.figure(2)
    plt.plot(epoch_times, train_ppls, label="Train PPL" )
    plt.scatter(epoch_times, train_ppls, label="Train PPL" )
    plt.plot(epoch_times, val_ppls, label="Validation PPL" )
    plt.scatter(epoch_times, val_ppls, label="Validation PPL" )
    plt.xlabel("Wall-clock-time")
    plt.xticks(epoch_times,rotation='vertical')
    plt.ylabel("PPL value")
    plt.suptitle(experiment, fontsize=10)
    plt.autoscale(enable=True, axis='both', tight=None)
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

    lc_path  = os.path.join(args.save_dir, 'learning_curves.npy')
    log_path = os.path.join(args.save_dir, 'log.txt')
    experiment = str.split(args.save_dir, "/")[1]
    
    return lc_path, log_path, experiment;

def extract_epoch_time(log_path):
    epoch_times = []
    epoch_string = re.compile("epoch")
    time_string = "time (s) spent in epoch: "
    with open(log_path) as f:
        lines = f.readlines()
        for line in lines:
            if epoch_string.search(line):
                time = float(line.strip().split(time_string)[1].split("A")[0]) #frist stripping the line, finding the part that has the time info, splitting by A which comes with "Additional..."
                epoch_times.append(time)
    for i in range(1, len(epoch_times)):
        epoch_times[i] = epoch_times[i-1]+ epoch_times[i]
    return epoch_times


lc_path, log_path, experiment = parse_args()
x = np.load(lc_path)[()]

epoch_times = extract_epoch_time(log_path)

train_ppls, val_ppls, train_losses, val_losses = [x['train_ppls'], x['val_ppls'], x['train_losses'], x['val_losses'] ]
plots(train_losses, val_losses, train_ppls, val_ppls, epoch_times, experiment)

