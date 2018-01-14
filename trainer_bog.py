import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
from model import AttentionLSTMClassifier
from torch.utils.data import Dataset, DataLoader
from early_stop import EarlyStop
from measurement import CalculateFM
import numpy as np


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
                if len(tmp) > self.pad_len:
                    tmp = tmp[: self.pad_len]
                self.data.append(tmp + [self.pad_int] * (self.pad_len - len(tmp)))
                tmp2 = tokens[2:]
                a_label = [0] * self.num_label
                for item in tmp2:
                    a_label[int(item)] = 1
                self.label.append(a_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.FloatTensor(self.label[idx])


def build_vocab(fold_path, use_unk=True):
    word_count = {}
    word2id = {}
    id2word = {}
    with open(os.path.join(fold_path, 'vocubulary.txt')) as f:
        # add <pad> first
        word2id['<pad>'] = 0
        id2word[0] = '<pad>'
        if use_unk:
            word2id['<unk>'] = 1
            id2word[1] = '<unk>'
        n = len(word2id)
        for word in f.readlines():
            w = word.strip()
            word2id[w] = n
            id2word[n] = w
            n += 1
    return word2id, id2word

if __name__ == '__main__':
    fold_path = 'data/Folds_9_Emotions/fold_0'
    pad_len = 30
    num_labels = 9
    batch_size = 400
    hidden_dim = 600
    word2id, id2word = build_vocab(fold_path, use_unk=True)
    vocab_size = len(word2id)
    embedding_dim = len(word2id)

    es = EarlyStop(3)
    train_data = DataSet(os.path.join(fold_path, 'train.csv'), pad_len, word2id, num_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = DataSet(os.path.join(fold_path, 'test.csv'), pad_len, word2id, num_labels)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = AttentionLSTMClassifier(embedding_dim, hidden_dim, vocab_size, word2id,
                                    num_labels, batch_size)
    model.load_bog_embedding(word2id)
    model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    loss_criterion = nn.BCELoss()
    for epoch in range(20):
        print('Epoch:', epoch, '===================================')
        train_loss = 0
        for i, (data, label) in enumerate(train_loader):
            y_pred = model(Variable(data).cuda())
            optimizer.zero_grad()
            loss = loss_criterion(y_pred, Variable(label).cuda())
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
        pred_list = []
        gold_list = []
        test_loss = 0
        for i, (data, label) in enumerate(test_loader):
            y_pred = model(Variable(data, volatile=True).cuda())
            loss = loss_criterion(y_pred, Variable(label, volatile=True).cuda())
            test_loss += loss.data[0]
            pred_list.append(y_pred.data.cpu().numpy())
            gold_list.append(label.numpy())

        threshold = 0.2
        print(threshold, ":",
              CalculateFM(np.concatenate(pred_list, axis=0), np.concatenate(gold_list, axis=0), threshold=threshold))
        threshold = 0.25
        print(threshold, ":",
              CalculateFM(np.concatenate(pred_list, axis=0), np.concatenate(gold_list, axis=0), threshold=threshold))
        threshold = 0.18
        print(threshold, ":",
              CalculateFM(np.concatenate(pred_list, axis=0), np.concatenate(gold_list, axis=0), threshold=threshold))
        threshold = 0.3
        print(threshold, ":",
              CalculateFM(np.concatenate(pred_list, axis=0), np.concatenate(gold_list, axis=0), threshold=threshold))

        print("Train Loss: ", train_loss, " Evaluation: ", test_loss)

        es.new_loss(test_loss)
        if es.if_stop():
            print('Start over fitting')
            break
