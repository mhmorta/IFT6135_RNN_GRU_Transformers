import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import utils
from textwrap import wrap
from collections import defaultdict

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

parser.add_argument('--saved_models_dir', type=str,
                    help='Directory with saved models \
                         (best_params.pt and exp_config.txt must be present there). \
                         All its\' individual subdirectories will be iterated')

saved_model_dir = parser.parse_args().saved_models_dir
for dir_name in [x[0] for x in os.walk(saved_model_dir) if x[0] != saved_model_dir]:
    args = utils.load_model_config(dir_name)
    x = np.load(os.path.join(dir_name, 'seq_loss.npy'))
    plt.figure#(figsize=(12, 12))
    avgs = defaultdict(list)
    plt.plot(x, marker='o')
    plt.title('{}: Average loss per time-step'.format(args['model']))
    plt.xlabel("Time-step")
    plt.ylabel("Loss value")
    plt.savefig('{}/{}_average_loss.png'.format(dir_name, args['model']))
    plt.show()
