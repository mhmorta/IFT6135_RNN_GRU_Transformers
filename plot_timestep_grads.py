import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import utils
from sklearn.preprocessing import minmax_scale

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

parser.add_argument('--saved_models_dir', type=str,
                    help='Directory with saved models \
                         (best_params.pt and exp_config.txt must be present there). \
                         All its\' individual subdirectories will be iterated')

saved_model_dir = parser.parse_args().saved_models_dir
plt.figure()#(figsize=(12, 12))
for dir_name in [x[0] for x in os.walk(saved_model_dir)]:
    if dir_name == saved_model_dir:
        continue
    args = utils.load_model_config(dir_name)
    x_raw = np.load(os.path.join(dir_name, 'timestep_grads.npy'))
    x = minmax_scale(x_raw)
    plt.plot(x, marker='o', label=args['model'])
    plt.title('{}'.format('Final time-step loss gradient wrt hidden states'))

plt.xlabel("Hidden state (concatenated)")
plt.ylabel("Rescaled gradient norm")
plt.legend()
plt.savefig('{}/average_loss.png'.format(dir_name))
plt.show()
