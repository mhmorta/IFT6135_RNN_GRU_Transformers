import matplotlib.pyplot as plt
import numpy as np

grad_norms = np.load('output/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=1000_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_dir=output_timestep_loss=True_compute_gradient=True_0/gradient_norms.npy')

print(grad_norms)

plt.title('Gradient norms')


def normalized(x):
    return (x - min(x)) / (max(x) - min(x))


plt.plot(normalized(grad_norms[:, 0]), label='RNN hidden layer 1')
#plt.plot(normalized(grad_norms[:, 1]), label='RNN hidden layer 2')
plt.xlabel('Time-step')
plt.ylabel('Norm')
plt.legend()
plt.show()

