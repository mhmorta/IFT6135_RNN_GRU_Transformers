import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

grads = OrderedDict()
#grads['RNN'] = np.load('results/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_save_dir=output_timestep_loss=true_0/gradient_norms.npy')
#grads['GRU'] = np.load('results/GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_dir=output_timestep_loss=true_save_best_0/gradient_norms.npy')
#grads['GRU'] = np.load('output/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=1000_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_dir=output_timestep_loss=True_compute_gradient=True_0/gradient_norms.npy')
grads['RNN'] = np.load('output/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_save_dir=output_compute_gradient=True_0/gradient_norms.npy')
#grads['GRU'] = np.load('output/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=1000_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_dir=output_timestep_loss=True_compute_gradient=True_0/gradient_norms.npy')


def normalized(x):
    return (x - min(x)) / (max(x) - min(x))


# https://matplotlib.org/gallery/subplots_axes_and_figures/subplots_demo.html
num_plots = grads['RNN'].shape[1]
f, axarr = plt.subplots(num_plots)
lines = []
line_labels = []
for i in range(num_plots):
    axarr[i].set_title('Hidden layer {}'.format(i), fontsize=10)
    for k, v in grads.items():
        lines.append(axarr[i].plot(normalized(v[:, i]), label=k))
    axarr[i].legend(loc="upper left", bbox_to_anchor=[0, 1],
                 ncol=2, fancybox=False)

for ax in axarr.flat:
    ax.set(xlabel='Time-step', ylabel='Normalized gradient norm')
plt.show()
