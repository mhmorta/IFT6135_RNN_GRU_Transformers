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
import math
import utils

def saveFig(address):
    if os.path.isfile(address):
        os.remove(address)
    plt.savefig(address) 

def plots(train_losses, val_losses, train_ppls, val_ppls, epoch_times, experiment, directory ):
    epochs = range(1, 41)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    temp = len(train_losses)/40
    temp2 = len(val_losses)/40
    ytickss = np.append(np.array([np.min(np.clip(val_ppls, 50, 350)), np.min(np.clip(train_ppls, 50, 350))],dtype=np.int8), np.arange(0, 400, 50))
    label_font_size = 20
    ticks_font_size = 13

    plt.figure(1, figsize=(20, 12))
    train_losses_index = [(i % temp) == (temp-1 ) for i in range(len(train_losses))]
    val_losses_index = [(i % temp2 )== (temp2-1) for i in range(len(val_losses))]
    plt.plot(epochs, train_losses[train_losses_index], label="Train Losses (end)", marker='o' )
    plt.plot(epochs, val_losses[val_losses_index], label="Validation Losses", marker='o' )
    plt.xlabel("Epoch number")
    plt.ylabel("Loss value")
    plt.xticks(epochs,rotation='vertical')
    plt.suptitle(experiment+ '\n' + "Training and Validation losses at the end of each epoch", fontsize=13)
    plt.legend(fontsize = 'xx-large')
    saveFig(directory+'/losees.png')
    plt.close()

    plt.figure(2, figsize=(20, 12))
    plt.plot(epoch_times, np.clip(train_ppls, 50, 350), label="Train PPL", marker='o' )
    plt.plot(epoch_times, np.clip(val_ppls, 50, 350), label="Validation PPL" , marker='o' )
    plt.xlabel("Wall-clock-time", fontsize=label_font_size)
    plt.ylabel("PPL value", fontsize=label_font_size)
    plt.xticks(epoch_times,rotation='vertical', fontsize=ticks_font_size)
    plt.yticks(ytickss, fontsize=ticks_font_size)
    plt.suptitle('(Clipped) PPL between w.r.t the wall-clock-time\n' + experiment, fontsize=20)
    plt.legend(fontsize = 'xx-large')
    saveFig(directory+'/PPL_wrt_wct.png')
    plt.close()

    plt.figure(3, figsize=(20, 12))
    plt.plot(epochs, np.clip(train_ppls, 50, 350 ), label="Train PPL", marker='o' )
    plt.plot(epochs, np.clip(val_ppls, 50, 350), label="Validation PPL", marker='o' )
    plt.xlabel("Epoch", fontsize=label_font_size)
    plt.ylabel("PPL value", fontsize=label_font_size)
    plt.suptitle('(Clipped) PPL w.r.t the epoch number \n' + experiment , fontsize=label_font_size)
    plt.xticks(epochs,rotation='vertical', fontsize=ticks_font_size)
    plt.yticks(ytickss, fontsize=ticks_font_size)
    plt.legend(fontsize = 'xx-large')
    saveFig(directory+'/PPL_wrt_epoch.png')
    plt.close()

def parse_args(directory):
    lc_path  = os.path.join(directory, 'learning_curves.npy')
    log_path = os.path.join(directory, 'log.txt')
    experiment = str.split(directory, "/")[1]
    
    return lc_path, log_path, experiment

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

def files_exits(directory):
    lc_path  = os.path.join(directory, 'learning_curves.npy')
    log_path = os.path.join(directory, 'log.txt')
    if os.path.exists(lc_path) and os.path.exists(log_path):
        return True
    return False 

def generate_experiment_string(args):
    return str(args["model"]) + ", " + str(args["optimizer"]) + ", lr=" + str(args["initial_lr"] )+ ",  num_layers=" +  str(args["num_layers"]) + ",  hidden_size=" + str(args["hidden_size"])

def main():
    directories = glob.glob("results/Hyper_param/*")
    for directory in directories:
        if(files_exits(directory)):
            args = utils.load_model_config(directory)
            lc_path, log_path, experiment = parse_args(directory)
            x = np.load(lc_path)[()]
            experiment = generate_experiment_string(args)
            epoch_times = extract_epoch_time(log_path)
            train_ppls, val_ppls, train_losses, val_losses = [x['train_ppls'], x['val_ppls'], x['train_losses'], x['val_losses'] ]
            plots(train_losses, val_losses, train_ppls, val_ppls, epoch_times, experiment, directory)

main()
print('Done...')