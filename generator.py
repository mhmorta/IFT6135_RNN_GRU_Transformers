import argparse
import torch.nn
from models import GRU, RNN
from utils import ptb_raw_data
import os
import sys
from collections import OrderedDict

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='RNN',
                    help='type of recurrent net (RNN, GRU)')
parser.add_argument('--saved_params', type=str,
                    help='location of the saved params')
parser.add_argument('--batch_size', type=int, default=20,
                    help='size of one minibatch')
parser.add_argument('--hidden_size', type=int, default=1500,
                    help='size of hidden layers.')
parser.add_argument('--generated_seq_len', type=int, default=35,
                    help='length of generated sequences')
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of hidden layers in RNN/GRU')
parser.add_argument('--temperature', type=float, default=1,
                    help='randomness of predictions: [0, 1]')
parser.add_argument('--output', type=str, default="samples",
                    help='output directory to write generated samples')

args = parser.parse_args()
argsdict = OrderedDict()
argsdict.update(args.__dict__)
argsdict.pop("saved_params")
argsdict.pop("output")
argsdict.pop("data")
output_path = os.path.join(args.output, '_'.join(map(lambda item: "%s=%s" % item, argsdict.items())))

if not os.path.exists(output_path):
    os.makedirs(output_path)

# LOAD DATA
print('Loading data from ' + args.data)
raw_data = ptb_raw_data(data_path=args.data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))

if args.model == "RNN":
    model = RNN(args.emb_size, args.hidden_size, 0, args.batch_size, vocab_size, args.num_layers, 1)
    model.load_state_dict(torch.load("results/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_save_dir=output_timestep_loss=true_0/best_params.pt", map_location='cpu'))
elif args.model == "GRU":
    model = GRU(args.emb_size, args.hidden_size, 0, args.batch_size, vocab_size, args.num_layers, 1)
    model.load_state_dict(torch.load("results/GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_dir=output_timestep_loss=true_save_best_0/best_params.pt", map_location='cpu'))
else:
    raise Exception("Unknown model")

model.to(device)

# start with zeros
seed = torch.zeros(args.batch_size, dtype=torch.long)

samples = model.generate(seed, model.init_hidden(), args.generated_seq_len)
samples = samples.transpose(0, 1) # shape (batch_size, generated_seq_len)

with open(os.path.join(output_path, "samples.txt"), "w") as of:
    for sample in samples:
        print(" ".join([id_2_word[idx] for idx in sample.numpy()]), file=of)
