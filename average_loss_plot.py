import matplotlib.pyplot as plt
import numpy as np
import os
import utils
from textwrap import wrap
from collections import defaultdict


results_dir = "results/4_1"
for dir_name in [x[0] for x in os.walk(results_dir)]:
    if dir_name == results_dir:
        continue
    args = utils.load_model_config(dir_name)
    x = np.load(os.path.join(dir_name, 'seq_loss.npy'))
    plt.figure(figsize=(12, 12))
    avgs = defaultdict(list)
    #for idx, val in enumerate(x):
    #    avgs[idx % args['seq_len']].append(val)
    #l = []
    #for idx in range(args['seq_len']):
    #    l.append(np.mean(avgs[idx]))
    #print('l:', l)
    #print('sum:', sum(l))
    l = x
    plt.plot(l, marker='o')
    plt.title('{}\n\n({})'.format('Average loss per time-step', "\n".join(wrap(args['name']))))
    plt.xlabel("Time-step")
    plt.ylabel("Loss value")
    plt.savefig('{}/average_loss.png'.format(dir_name))
    plt.show()
