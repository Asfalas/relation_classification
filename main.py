from code.att_bilstm import train_att_bilstm, test_att_bilstm
import argparse
import torch
import torch.nn as nn
import os
from code.bert import train_bert, test_bert
from transformers import BertConfig, BertTokenizer
from code.data_utils import class2label

parser = argparse.ArgumentParser()
# task setting
parser.add_argument('--model', type=str, default='att_bilstm', choices=['bert', 'att_bilstm'])

# train setting
parser.add_argument('--max_len', type=int, default=90)

parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--num_train_epochs', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=5e-3)

# for BiLSTM
parser.add_argument('--embedding_dim', type=int, default=200, choices=[50, 100, 200, 300])
parser.add_argument('--hidden_dim', type=int, default=400)
parser.add_argument('--keep_prob', type=int, default=0.5)

# file path
parser.add_argument('--save_model', type=str, default='./model/')
parser.add_argument('--output_dir', type=str, default='./out/')

# others
parser.add_argument('--max_grad_norm', type=float, default=1.0)
args = parser.parse_args()

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('use: ' + str(args.device))
for file_dir in [args.save_model, args.output_dir]:
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

save_name = args.model
args.save_model_path = os.path.join(args.save_model, save_name + ".pkl")
args.output_path = os.path.join(args.output_dir, save_name + ".txt")
args.num_labels = len(class2label)


def main():
    if args.model == 'bert':
        train_bert(args)
    else:
        train_att_bilstm(args)
        test_att_bilstm(args)


if __name__ == "__main__":
    print(args)
    main()
