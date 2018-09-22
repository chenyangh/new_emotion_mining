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
import matplotlib.pyplot as plt
from sklearn.metrics import *
import itertools


class DataSet(Dataset):
    def __init__(self, __fold_path, __pad_len, __word2id, __num_labels, max_size=None, use_unk=False):

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
        self.seq_len = []
        self.use_unk = use_unk
        self.read_data(__fold_path)
        assert len(self.seq_len) == len(self.data) == len(self.label)

    def read_data(self, __fold_path):
        with open(__fold_path, 'r') as f:
            empty_line_count = 0
            all_lines = 0
            for line in f.readlines():

                tokens = line.lower().split('\t')
                if tokens[0][0] != 's':
                    continue
                all_lines += 1
                if self.use_unk:
                    tmp = [self.word2id[x] if x in self.word2id else self.word2id['<unk>'] for x in tokens[0].split()]
                else:
                    tmp = [self.word2id[x] for x in tokens[1].split() if x in self.word2id]
                if len(tmp) == 0:
                    empty_line_count += 1
                    continue
                self.seq_len.append(len(tmp) if len(tmp) < self.pad_len else self.pad_len)
                if len(tmp) > self.pad_len:
                    tmp = tmp[: self.pad_len]
                self.data.append(tmp + [self.pad_int] * (self.pad_len - len(tmp)))
                tmp2 = tokens[-1]
                a_label = [0] * self.num_label

                a_label[int(tmp2)] = 1
                self.label.append(a_label)
            print('found', empty_line_count, 'lines over', all_lines)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.LongTensor([self.seq_len[idx]]), torch.FloatTensor(self.label[idx])


def build_vocab(fold_path, fold_id, use_unk=False):
    word2id = {}
    id2word = {}
    word_list = []
    with open(fold_path + '/inf_vocab.txt', 'r') as f:
        for line in f.readlines():
            word = line.strip()

            word_list.append(word)

        # add <pad> first
        word2id['<pad>'] = 0
        id2word[0] = '<pad>'
        if use_unk:
            word2id['<unk>'] = 1
            id2word[1] = '<unk>'
        n = len(word2id)
        for word in word_list:
            word2id[word] = n
            id2word[n] = word
            n += 1
    return word2id, id2word


def sort_batch(batch, ys, lengths):
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    targ_tensor = ys[perm_idx]
    return seq_tensor, targ_tensor, seq_lengths


def one_fold(fold_int, is_nine_folds):
    fold_id = str(fold_int)
    if is_nine_folds:
        fold_path = 'data/Folds_9_Emotions/fold_' + fold_id
        num_labels = 9
    else:
        fold_path = 'data/Folds/fold_' + fold_id
        num_labels = 16

    pad_len = 30
    batch_size = 64
    hidden_dim = 400

    es = EarlyStop(2)
    word2id, id2word = build_vocab(fold_path, fold_id, use_unk=True)
    embedding_dim = len(word2id)
    vocab_size = len(word2id)
    train_data = DataSet(os.path.join(fold_path, 'train.csv'), pad_len, word2id, num_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = DataSet(os.path.join(fold_path, 'test.csv'), pad_len, word2id, num_labels)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = AttentionLSTMClassifier(embedding_dim, hidden_dim, vocab_size, word2id,
                                    num_labels, batch_size, use_att=True)
    model.load_bog_embedding(word2id)
    model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    loss_criterion = nn.BCELoss()
    for epoch in range(4):
        print('Epoch:', epoch, '===================================')
        train_loss = 0
        for i, (data, seq_len, label) in enumerate(train_loader):
            data, label, seq_len = sort_batch(data, label, seq_len.view(-1))
            y_pred = model(Variable(data).cuda(), seq_len)
            optimizer.zero_grad()
            loss = loss_criterion(y_pred, Variable(label).cuda())
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
        pred_list = []
        gold_list = []
        test_loss = 0
        for i, (data, seq_len, label) in enumerate(test_loader):
            data, label, seq_len = sort_batch(data, label, seq_len.view(-1))
            y_pred = model(Variable(data, volatile=True).cuda(), seq_len)
            loss = loss_criterion(y_pred, Variable(label, volatile=True).cuda())
            test_loss += loss.data[0]
            pred_list.append(y_pred.data.cpu().numpy())
            gold_list.append(label.numpy())

        print("Train Loss: ", train_loss, " Evaluation: ", test_loss)
        es.new_loss(test_loss)
        if es.if_stop():
            print('Start over fitting')
            break
    return np.concatenate(pred_list, axis=0), np.concatenate(gold_list, axis=0)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        cm = cm.astype('float')
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def confusion_matrix(pred_list, gold_list):
    assert gold_list.shape == pred_list.shape
    # m, n = pred_list.shape

    m = len(pred_list)
    cm = np.zeros([len(emotions), len(emotions)])

    for i in range(m):
        j = gold_list[i]
        k = pred_list[i]
        cm[j][k] += 1
    return cm


if __name__ == '__main__':
    p_avg = 0
    r_avg = 0
    f_avg = 0
    emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'thankfulness', 'disgust', 'guilt']

    cnf_matrix_list = []
    cm = np.zeros([len(emotions), len(emotions)])
    measure_9_emo = np.zeros([3])
    for i in range(5):
        pred_list, gold_list = one_fold(i, is_nine_folds=True)

        pred_list = np.argmax(pred_list, axis=1)
        gold_list = np.argmax(gold_list, axis=1)

        measure_9_emo[0] += precision_score(gold_list, pred_list, average='macro')
        measure_9_emo[1] += recall_score(gold_list, pred_list, average='macro')
        measure_9_emo[2] += f1_score(gold_list, pred_list, average='macro')


        cnf_matrix = confusion_matrix(pred_list, gold_list)
        cnf_matrix_list.append(cnf_matrix)

    for cnf_tmp in cnf_matrix_list:
        cm += cnf_tmp

    measure_9_emo /= 5
    print(measure_9_emo)

    cm /= 5
    plt.figure()
    plot_confusion_matrix(cm, classes=emotions, normalize=False)
    plt.show()



