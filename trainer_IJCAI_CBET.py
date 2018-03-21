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
import copy

NUM_CLASS = 9


def cbet_data(file_path):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words('english'))
    label = []
    train_text = []
    for t in open(file_path, 'r').readlines():
        tokens = t.strip().split('\t')
        text = tokens[1].lower()
        a_label = [1 if str(x) in tokens[2:] else 0 for x in range(NUM_CLASS)]
        a_label = ''.join(str(e) for e in reversed(a_label))
        label.append(int(a_label, 2))
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        train_text.append(' '.join(filtered_sentence))

    return train_text, label


def stratified_shuffle_split(X, y):
    split_ratio = 0.1
    X_s, y_s, X_d, y_d = [], [], [], []
    single_label_decimal = [2**x for x in range(NUM_CLASS)]
    for i_x, i_y in zip(X, y):
        if i_y in single_label_decimal:
            X_s.append(i_x)
            y_s.append(i_y)
        else:
            X_d.append(i_x)
            y_d.append(i_y)

    from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=0)
    train_index, dev_index = next(sss.split(X_s, y_s))

    sss = ShuffleSplit(n_splits=1, test_size=split_ratio, random_state=0)
    train_index_d, dev_index_d = next(sss.split(X_d, y_d))

    train_index = np.concatenate((train_index, train_index_d), axis=0)
    dev_index = np.concatenate((dev_index, dev_index_d), axis=0)

    assert len(train_index) + len(dev_index) == len(y)
    return train_index, dev_index


class DataSet(Dataset):
    def __init__(self, __X, __y, __pad_len, __word2id, __num_labels, max_size=None, use_unk=True):

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
        self.only_single = True
        self.use_unk = use_unk

        self.read_data(__X, __y) # process data
        assert len(self.seq_len) == len(self.data) == len(self.label)

    def read_data(self, __X, __y):
        assert len(__X) == len(__y)
        num_empty_lines = 0
        for X, y in zip(__X, __y):
            tokens = X.split()
            if self.use_unk:
                tmp = [self.word2id[x] if x in self.word2id else self.word2id['<unk>'] for x in tokens]
            else:
                tmp = [self.word2id[x] for x in tokens if x in self.word2id]
            if len(tmp) == 0:
                tmp = [self.word2id['<empty>']]
                num_empty_lines += 1
                # continue
            self.seq_len.append(len(tmp) if len(tmp) < self.pad_len else self.pad_len)
            if len(tmp) > self.pad_len:
                tmp = tmp[: self.pad_len]
            self.data.append(tmp + [self.pad_int] * (self.pad_len - len(tmp)))
            # a_label = [0] * self.num_label

            tmp = bin(y)[2:]
            a_label = [int(x) for x in tmp[::-1]] + (self.num_label - len(tmp)) * [0]

            self.label.append(a_label)
        print(num_empty_lines, 'empty lines found')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.LongTensor([self.seq_len[idx]]), torch.FloatTensor(self.label[idx])


def build_vocab(X_train, vocab_size):
    word_count = {}
    word2id = {}
    id2word = {}
    for line in X_train:
        tokens = line.split()
        for word in tokens:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    word_list = [x for x, _ in sorted(word_count.items(), key=lambda v: v[1], reverse=True)]
    if len(word_count) < vocab_size:
        raise Exception('Vocab less than requested!!!')

    # add <pad> first
    word2id['<pad>'] = 0
    id2word[0] = '<pad>'

    word2id['<unk>'] = 1
    id2word[1] = '<unk>'
    word2id['<empty>'] = 2
    id2word[2] = '<empty>'

    n = len(word2id)
    word_list = word_list[:vocab_size - n]

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


def one_fold(fold_path):

    vocab_size = 20000
    pad_len = 30
    batch_size = 64
    embedding_dim = 200
    hidden_dim = 800
    num_labels = NUM_CLASS

    X, y = cbet_data(os.path.join(fold_path, 'train.csv'))

    train_index, dev_index = stratified_shuffle_split(X, y)
    y = np.asarray(y)
    X_train, X_dev = [X[i] for i in train_index], [X[i] for i in dev_index]
    y_train, y_dev = y[train_index], y[dev_index]

    word2id, id2word = build_vocab(X_train, vocab_size)
    # __X, __y, __pad_len, __word2id, __num_labels
    train_data = DataSet(X_train, y_train, pad_len, word2id, num_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    dev_data = DataSet(X_dev, y_dev, pad_len, word2id, num_labels)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=True)

    X_test, y_test = cbet_data(os.path.join(fold_path, 'test.csv'))
    test_data = DataSet(X_test, y_test, pad_len, word2id, num_labels)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = AttentionLSTMClassifier(embedding_dim, hidden_dim, vocab_size, word2id,
                                    num_labels, batch_size, use_att=True, soft_last=True)
    model.load_glove_embedding(id2word)
    model.cuda()

    optimizer = optim.Adam(model.parameters())
    loss_criterion = nn.BCELoss()
    es = EarlyStop(2)
    old_model = None
    for epoch in range(10):
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
        # evaluation
        for i, (data, seq_len, label) in enumerate(dev_loader):
            data, label, seq_len = sort_batch(data, label, seq_len.view(-1))
            y_pred = model(Variable(data, volatile=True).cuda(), seq_len)
            loss = loss_criterion(y_pred, Variable(label, volatile=True).cuda())
            test_loss += loss.data[0]
            pred_list.append(y_pred.data.cpu().numpy())
            gold_list.append(label.numpy())

        if old_model is not None:
            del old_model
            old_model = copy.deepcopy(model)
        else:
            old_model = copy.deepcopy(model)
        print("Train Loss: ", train_loss, " Evaluation: ", test_loss)
        es.new_loss(test_loss)
        if es.if_stop():
            print('Start over fitting')
            del model
            model = old_model
            break

    # testing
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


def one_vs_all_measure(gold, pred):
    one_hot_gold = np.zeros([len(gold), NUM_CLASS])
    one_hot_pred = np.zeros([len(pred), NUM_CLASS])
    assert len(gold) == len(pred)
    for i in range(len(gold)):
        one_hot_gold[i, gold[i]] = 1
        one_hot_pred[i, pred[i]] = 1
    retval = np.zeros([NUM_CLASS, 3])
    for i in range(NUM_CLASS):
        per_gold = one_hot_gold[:, i]
        per_pred = one_hot_pred[:, i]
        p = precision_score(per_gold, per_pred, average='binary')
        r = recall_score(per_gold, per_pred, average='binary')
        f = f1_score(per_gold, per_pred, average='binary')
        retval[i, :] = np.asarray([p, r, f])
    return retval


if __name__ == '__main__':
    p_avg = 0
    r_avg = 0
    f_avg = 0
    emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'thankfulness', 'disgust', 'guilt']
    # emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'thankfulness', 'disgust', 'guilt', 'betrayed', 'frustrated', 'hopeless', 'lonely', 'rejected', 'schadenfreude', 'self_loath']

    cnf_matrix_list = []
    cm = np.zeros([len(emotions), len(emotions)])
    measure_9_emo = np.zeros([3])
    one_vs_all = np.zeros([NUM_CLASS, 3])

    for i in range(5):
        fold_path = 'data/Folds_9_Emotions/fold_' + str(i)
        # fold_path = 'data/Folds/fold_' + str(i)
        pred_list, gold_list = one_fold(fold_path)

        f_ma = []
        f_mi = []
        for threshold in range(0, 100, 5):
            threshold /= 100
            tmp = CalculateFM(pred_list, gold_list, threshold=threshold)
            f_ma.append(tmp['MacroFM'])
            f_mi.append(tmp['MicroFM'])

        print(f_ma)
        print(f_mi)

        pred_list = np.argmax(pred_list, axis=1)
        gold_list = np.argmax(gold_list, axis=1)
        one_vs_all += one_vs_all_measure(gold_list, pred_list)

        measure_9_emo[0] += precision_score(gold_list, pred_list, average='macro')
        measure_9_emo[1] += recall_score(gold_list, pred_list, average='macro')
        measure_9_emo[2] += f1_score(gold_list, pred_list, average='macro')


        cnf_matrix = confusion_matrix(pred_list, gold_list)
        cnf_matrix_list.append(cnf_matrix)

    for cnf_tmp in cnf_matrix_list:
        cm += cnf_tmp


    one_vs_all /= 5
    print(one_vs_all)

    measure_9_emo /= 5
    print(measure_9_emo)

    cm /= 5
    plt.figure()
    plot_confusion_matrix(cm, classes=emotions, normalize=False)
    plt.show()
