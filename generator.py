import argparse
import torch.nn
from models import GRU, RNN
from utils import ptb_raw_data, load_model_config
import os


# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')


parser.add_argument('--saved_models_dir', type=str,
                    help='Directory with saved models \
                         (best_params.pt and exp_config.txt must be present there). \
                         All its\' individual subdirectories will be iterated')

parser.add_argument('--generated_seq_len', type=int, default=35,
                    help='length of generated sequences')


args = parser.parse_args()
output_dir = parser.parse_args().saved_models_dir
seq_len = args.generated_seq_len

# load model configuration
args = load_model_config(output_dir)


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# LOAD DATA
print('Loading data from ' + args.data)
raw_data = ptb_raw_data(data_path=args.data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))

if args.model == 'RNN':
    model = RNN(emb_size=args.emb_size, hidden_size=args.hidden_size,
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=vocab_size, num_layers=args.num_layers,
                dp_keep_prob=args.dp_keep_prob)
elif args.model == 'GRU':
    model = GRU(emb_size=args.emb_size, hidden_size=args.hidden_size,
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=vocab_size, num_layers=args.num_layers,
                dp_keep_prob=args.dp_keep_prob)
else:
    raise Exception("Unknown model")

model.load_state_dict(torch.load('{}/best_params.pt'.format(output_dir), map_location=device))

model.to(device)

# start with zeros
seed = torch.zeros(args.batch_size, dtype=torch.long)

samples = model.generate(seed, model.init_hidden(), seq_len)
samples = samples.transpose(0, 1) # shape (batch_size, generated_seq_len)

with open(os.path.join(output_dir, "samples.{}.txt".format(seq_len)), "w") as of:
    for sample in samples:
        print(" ".join([id_2_word[idx] for idx in sample.numpy()]), file=of)
