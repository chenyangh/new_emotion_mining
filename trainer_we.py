import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from early_stop import EarlyStop
import numpy as np
from measurement import CalculateFM

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
                a_data = [self.word2id[x] if x in self.word2id else self.word2id['<unk>'] for x in tokens[1].split()]
                # assert len(a_data) < self.pad_len
                if len(a_data) >= self.pad_len:
                    a_data = a_data[: self.pad_len]
                self.data.append(a_data + [self.pad_int] * (self.pad_len - len(a_data)))
                label_list = [int(x.strip()) for x in tokens[2:]]

                a_label = []
                for idx in range(self.num_label):
                    if idx in label_list:
                        a_label.extend([0, 1])
                    else:
                        a_label.extend([1, 0])
                self.label.append(a_label)
        assert len(self.label) == len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.FloatTensor(self.label[idx])


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


if __name__ == '__main__':
    fold_path = 'data/Folds/fold_0'
    vocab_size = 20000
    pad_len = 30
    num_labels = 16
    batch_size = 400
    embedding_dim = 200
    hidden_dim = 600

    es = EarlyStop(3)
    word2id, id2word = build_vocab(fold_path, vocab_size, use_unk=True)
    train_data = DataSet(os.path.join(fold_path, 'train.csv'), pad_len, word2id, num_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = DataSet(os.path.join(fold_path, 'test.csv'), pad_len, word2id, num_labels)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = AttentionLSTMClassifier(embedding_dim, hidden_dim, vocab_size, word2id,
                                    num_labels, batch_size).cuda()

    print('Training set size:', len(train_data), 'Test set size:', len(test_data))
    optimizer = optim.Adam(model.parameters())
    loss_criterion = nn.BCELoss()
    for epoch in range(40):
        print('Epoch:', epoch, '===================================')
        train_loss = 0
        for i, (data, label) in enumerate(train_loader):
            y_pred = model(Variable(data).cuda())
            optimizer.zero_grad()
            loss = weighted_binary_cross_entropy(y_pred, Variable(label).cuda(), weights=[0.5, 0.5])
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
        print("Train Loss: ", train_loss)

        test_loss = 0
        pred_list = []
        gold_list = []
        for i, (data, label) in enumerate(test_loader):
            y_pred = model(Variable(data, volatile=True).cuda())
            loss = weighted_binary_cross_entropy(y_pred, Variable(label, volatile=True).cuda(), weights=[0.9, 0.1])
            test_loss += loss.data[0]
            pred_list.append(np.argmax(y_pred.data.cpu().numpy().reshape((-1, num_labels, 2)), axis=2))
            gold_list.append(np.argmax(label.numpy().reshape((-1, num_labels, 2)), axis=2))

        print(CalculateFM(np.concatenate(pred_list, axis=0), np.concatenate(gold_list, axis=0)))
        es.new_loss(test_loss)
        print("Evaluation: ", test_loss)

        if es.if_stop():
            print('Start over fitting')
            break
