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
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
        self.seq_len = []
        self.read_data(__fold_path)
        assert len(self.seq_len) == len(self.data) == len(self.label)

    def read_data(self, __fold_path):
        with open(__fold_path, 'r') as f:
            for line in f.readlines():
                tokens = line.split('\t')
                tmp = [self.word2id[x] for x in tokens[1].split() if x in self.word2id]
                if len(tmp) == 0:
                    tmp = [self.word2id['<unk>']]
                self.seq_len.append(len(tmp) if len(tmp) < self.pad_len else self.pad_len)
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
        return torch.LongTensor(self.data[idx]), torch.LongTensor([self.seq_len[idx]]), torch.FloatTensor(self.label[idx])


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
    hidden_dim = 600
    embedding_dim = 200

    word2id, id2word = build_vocab(fold_path, use_unk=True)
    vocab_size = len(word2id)

    es = EarlyStop(2)
    train_data = DataSet(os.path.join(fold_path, 'train.csv'), pad_len, word2id, num_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = DataSet(os.path.join(fold_path, 'test.csv'), pad_len, word2id, num_labels)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = AttentionLSTMClassifier(embedding_dim, hidden_dim, vocab_size, word2id,
                                    num_labels, batch_size)
    model.load_glove_embedding(id2word)
    model.cuda()

    optimizer = optim.Adam(model.parameters())
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

    return pred_list, gold_list


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
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
    m, n = pred_list.shape
    #
    # for i in range(m):
    #     for j in range(n):
    #         if pred_list()


    cm = np.zeros([len(emotions), len(emotions)])


if __name__ == '__main__':
    np.set_printoptions(precision=2)

    emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'thankfulness', 'disgust', 'guilt',
                     'betrayed', 'frustrated', 'hopeless', 'lonely', 'rejected', 'schadenfreude', 'self_loath']

    pred_list, gold_list = one_fold(0, is_nine_folds=False)

    import pickle
    with open('cnf_data.pkl', 'bw') as f:
        pickle.dump((pred_list, gold_list), f)
    # pred_list = np.concatenate(pred_list, axis=0)
    # gold_list = np.concatenate(gold_list, axis=0)
    # pred_list = [1 & (v > 0.3) for v in pred_list]
    # pred_list = np.array(pred_list)
    #
    # cnf_matrix = confusion_matrix(pred_list, gold_list)
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=emotions, normalize=True,
    #                   title='Model 1.1 confusion matrix')
    # plt.show()
