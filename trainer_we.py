import math
import numpy as np
import logging
import argparse
import os
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
from data_helper import build_vocab
from model import AttentionLSTMClassifier
from torch.utils.data import Dataset, DataLoader


class DataSet(Dataset):
    def __init__(self, __fold_path, __pad_len, __word2id, __num_labels, max_size=None):

        self.pad_len = __pad_len
        self.word2id = __word2id
        self.pad_int = __word2id['<pad>']
        if max_size is not None:
            self.source = self.source[:max_size]
            self.target = self.target[:max_size]
            self.tag = self.tag[:max_size]
        self.data = []
        self.label = []
        self.num_label = __num_labels
        self.read_data(__fold_path)

    def read_data(self, __fold_path):
        with open(__fold_path, 'r') as f:
            for line in f.readlines():
                tokens = line.split('\t')
                tmp = [self.word2id[x] if x in self.word2id else self.word2id['<unk>'] for x in tokens[1].split()]
                self.data.append(tmp + [self.pad_int] * (self.pad_len - len(tmp)))
                tmp2 = tokens[2:]
                a_label = [0] * self.num_label
                for item in tmp2:
                    a_label[int(item)] = 1
                self.label.append(a_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.LongTensor(self.label[idx])


if __name__ == '__main__':
    fold_path = 'data/Folds/fold_0'
    vocab_size = 20000
    pad_len = 30
    num_labels = 16
    batch_size = 300
    embedding_dim = 200
    hidden_dim = 600

    word2id, id2word = build_vocab(fold_path, vocab_size, use_unk=True)
    train_data = DataSet(os.path.join(fold_path, 'train.csv'), pad_len, word2id, num_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size)

    test_data = DataSet(os.path.join(fold_path, 'test.csv'), pad_len, word2id, num_labels)
    test_loader = DataLoader(train_data, batch_size=batch_size)

    model = AttentionLSTMClassifier(embedding_dim, hidden_dim, vocab_size, word2id,
                                    num_labels, batch_size).cuda()

    for i, (data, label) in enumerate(train_loader):
        loss = model(Variable(data).cuda(), label)

