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
#            GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_dir=output_timestep_loss=true_save_best_0
#results/4_1/GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_dir=output_timestep_loss=true_save_best_0/TRANSFORMER_SGD_LR_SCHEDULE_model=TRANSFORMER_optimizer=SGD_LR_SCHEDULE_initial_lr=20_batch_size=128_seq_len=35_hidden_size=512_num_layers=6_dp_keep_prob=0.9_save_best_36/exp_config.txt
# results/4_1/GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_dir=output_timestep_loss=true_save_best_0/TRANSFORMER_SGD_LR_SCHEDULE_model=TRANSFORMER_optimizer=SGD_LR_SCHEDULE_initial_lr=20_batch_size=128_seq_len=35_hidden_size=512_num_layers=6_dp_keep_prob=0.9_save_best_36
saved_model_dir = parser.parse_args().saved_models_dir
dirs = [x[0] for x in os.walk(saved_model_dir) if x[0] != saved_model_dir]
for dir_name in dirs:
    args = utils.load_model_config(dir_name)
    x = np.load(os.path.join(dir_name, 'seq_loss.npy'))
    plt.figure#(figsize=(12, 12))
    plt.plot(x, marker='o', label=args['model'])
    plt.title('Average loss per time-step')
    plt.xlabel("Time-step")
    plt.ylabel("Loss value")
    plt.legend()

for dir in dirs:
    plt.savefig('{}/average_loss.png'.format(dir))
    plt.show()
