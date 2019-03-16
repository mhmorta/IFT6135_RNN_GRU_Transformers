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
import glob

def plots(train_losses, val_losses, train_ppls, val_ppls, epoch_times, experiment, directory ):
    plt.figure(1, figsize=(16, 9))
    epochs = range(40)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    temp = len(train_losses)/40
    temp2 = len(val_losses)/40
    train_losses_index = [(i % temp) == (temp-1 ) for i in range(len(train_losses))]
    val_losses_index = [(i % temp2 )== (temp2-1) for i in range(len(val_losses))]
    plt.plot(epochs, train_losses[train_losses_index], label="Train Losses (end)")
    plt.scatter(epochs, train_losses[train_losses_index], label="Train Losses")
    plt.plot(epochs, val_losses[val_losses_index], label="Validation Losses")
    plt.scatter(epochs, val_losses[val_losses_index], label="Validation Losses")

    # train_losses_index = [(i % 1327 == 0) for i in range(len(train_losses))]
    # val_losses_index = [(i % 105 == 0) for i in range(len(val_losses))]
    # plt.plot(epochs, train_losses[train_losses_index], label="Train Losses(beggining)")
    # plt.scatter(epochs, train_losses[train_losses_index], label="Train Losses")
    # plt.plot(epochs, val_losses[val_losses_index], label="Validation Losses")
    # plt.scatter(epochs, val_losses[val_losses_index], label="Validation Losses")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss value")
    plt.xticks(epochs,rotation='vertical')

    plt.suptitle(experiment+ '\n' + "Training and Validation losses at the end of each epoch", fontsize=10)
    plt.legend()
    plt.savefig(directory+'/1.png')

    plt.figure(2, figsize=(16, 9))
    plt.plot(epoch_times, train_ppls, label="Train PPL" )
    plt.scatter(epoch_times, train_ppls, label="Train PPL" )
    plt.plot(epoch_times, val_ppls, label="Validation PPL" )
    plt.scatter(epoch_times, val_ppls, label="Validation PPL" )
    plt.xlabel("Wall-clock-time")
    plt.xticks(epoch_times,rotation='vertical')
    plt.ylabel("PPL value")
    plt.suptitle(experiment + '\n' + 'PPL over wall-clock-time', fontsize=10)
    plt.legend()
    plt.savefig(directory+'/2.png')


    plt.figure(3, figsize=(16, 9))
    plt.plot(epochs, train_ppls, label="Train PPL" )
    plt.scatter(epochs, train_ppls, label="Train PPL" )
    plt.plot(epochs, val_ppls, label="Validation PPL" )
    plt.scatter(epochs, val_ppls, label="Validation PPL" )
    plt.xlabel("Wall-clock-time")
    plt.xticks(epochs,rotation='vertical')
    plt.ylabel("PPL value")
    plt.suptitle(experiment + '\n' + 'PPL over epcoh#', fontsize=10)
    plt.legend()
    plt.savefig(directory+'/3.png')
    # plt.show()

def parse_args(directory):
    # parser = argparse.ArgumentParser(description='Visualization Argument Parser')

    # parser.add_argument('--save_dir', type=str, default='',
    #                     help='path to save the experimental config, logs, model \
    #                     This is automatically generated based on the command line \
    #                     arguments you pass and only needs to be set if you want a \
    #                     custom dir name')

    # args = parser.parse_args()
    # argsdict = args.__dict__

    lc_path  = os.path.join(directory, 'learning_curves.npy')
    log_path = os.path.join(directory, 'log.txt')
    experiment = str.split(directory, "/")[1]
    
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

def main():
    directories = glob.glob("results/*")
    for directory in directories:
        lc_path, log_path, experiment = parse_args(directory)
        x = np.load(lc_path)[()]
        epoch_times = extract_epoch_time(log_path)
        train_ppls, val_ppls, train_losses, val_losses = [x['train_ppls'], x['val_ppls'], x['train_losses'], x['val_losses'] ]
        plots(train_losses, val_losses, train_ppls, val_ppls, epoch_times, experiment, directory)

main()
print('Done...')