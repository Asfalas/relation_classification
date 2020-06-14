import os
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
from .data_utils import load_data_and_labels, class2label


def convert_tokens_to_ids(tokenizer, tokens):
    input_ids = []
    for t in tokens:
        tokenized_text = tokenizer.tokenize(t)

        input_ids.append(tokenizer.convert_tokens_to_ids(tokenized_text))
    return input_ids


def sequence_padding(tokenizer, X, y, max_len=90, add_CLS=True):
    X = convert_tokens_to_ids(tokenizer, X)

    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    cls_id = tokenizer.convert_tokens_to_ids("[CLS]")

    tensorX = []
    tensory = []
    for i in range(len(X)):
        curX, cury = [], y[i]
        if len(X[i]) <= max_len:
            curX = X[i] + [pad_id] * (max_len - len(X[i]))
        else:
            curX = X[i][:max_len]
        if add_CLS:
            curX = [cls_id] + curX[:max_len - 1]
        tensorX.append(curX)
        tensory.append(cury)

    tensorX = torch.LongTensor(tensorX)
    tensory = torch.LongTensor(tensory)
    tensorMask = tensorX.ne(pad_id).byte()
    return tensorX, tensorMask, tensory


class RCDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len=90, add_CLS=True):
        self.X, self.y = X, y
        # a = set()
        # for i in self.y:
        #     a = a | set(i)
        # print(a)
        self.X, self.mask, self.y = sequence_padding(tokenizer, self.X, self.y, max_len=max_len,
                                                     add_CLS=add_CLS)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]
