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
from sklearn.metrics import *
import itertools
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
NUM_CLASS = 9


def cbet_data(label_cols):
    data = pd.read_csv('data/CBET.csv')
    # test_data = pd.read_csv('data/test.csv')
    # label = pd.read_csv('data/test.csv')
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    label = data[label_cols]
    # for col in label_cols:
    #     label.append(data[col])
    #
    # label = np.asarray(label).transpose()
    # example_sent = "This is a sample sentence, showing off the stop words filtration."

    stop_words = set(stopwords.words('english'))

    train_text = []
    for t in data['text'].fillna("fillna").values:
        t = t.lower()
        word_tokens = word_tokenize(t)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        train_text.append(' '.join(filtered_sentence))

    # test_text = []
    # for t in test_data['comment_text'].fillna("fillna").values:
    #     t = t.lower()
    #     word_tokens = word_tokenize(t)
    #     filtered_sentence = [w for w in word_tokens if not w in stop_words]
    #     test_text.append(' '.join(filtered_sentence))
    #
    # sub_id = test_data['id']
    return train_text, label


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
            # if int(y) == 1:
            #     a_label = [0, 1]
            # else:
            #     a_label = [1, 0]

            self.label.append(y)
        print(num_empty_lines, 'empty lines found')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.LongTensor([self.seq_len[idx]]), torch.FloatTensor(self.label[idx])


class TestDataSet(Dataset):
    def __init__(self, __X, __pad_len, __word2id, __num_labels, max_size=None, use_unk=True):

        self.pad_len = __pad_len
        self.word2id = __word2id
        self.pad_int = __word2id['<pad>']
        if max_size is not None:
            self.source = self.source[:max_size]
            self.target = self.target[:max_size]
            self.tag = self.tag[:max_size]
        self.data = []
        self.num_label = __num_labels
        self.seq_len = []
        self.only_single = True
        self.use_unk = use_unk

        self.read_data(__X)  # process data
        assert len(self.seq_len) == len(self.data)

    def read_data(self, __X):
        num_empty_lines = 0
        for X in __X:
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
        print(num_empty_lines, 'empty lines found')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.LongTensor([self.seq_len[idx]])


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


def sort_batch_test(batch, lengths):
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    rever_sort = np.zeros(len(seq_lengths))
    for i, l in enumerate(perm_idx):
        rever_sort[l] = i
    return seq_tensor, seq_lengths, rever_sort.astype(int)


def one_fold(X_train, y_train, X_dev, y_dev):

    num_labels = NUM_CLASS
    vocab_size = 20000
    pad_len = 40
    batch_size = 64
    embedding_dim = 200
    hidden_dim = 500
    __use_unk = False

    word2id, id2word = build_vocab(X_train, vocab_size)

    train_data = DataSet(X_train, y_train, pad_len, word2id, num_labels, use_unk=__use_unk)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    dev_data = DataSet(X_dev, y_dev, pad_len, word2id, num_labels, use_unk=__use_unk)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

    # test_data = TestDataSet(X_test, pad_len, word2id, num_labels, use_unk=__use_unk)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = AttentionLSTMClassifier(embedding_dim, hidden_dim, vocab_size, word2id,
                                    num_labels, batch_size, use_att=False, soft_last=False)
    model.load_glove_embedding(id2word)
    model.cuda()
    es = EarlyStop(2)
    optimizer = optim.Adam(model.parameters())
    loss_criterion = nn.MSELoss()  #
    threshold = 0.65
    for epoch in range(30):
        print('Epoch:', epoch, '===================================')
        train_loss = 0
        for i, (data, seq_len, label) in enumerate(train_loader):

            data, label, seq_len = sort_batch(data, label, seq_len.view(-1))
            y_pred = model(Variable(data).cuda(), seq_len)

            #roc_reward = roc_auc_score(label.numpy().argmax(axis=1), y_pred.data.cpu().numpy()[:, 1])
            optimizer.zero_grad()
            loss = loss_criterion(y_pred, Variable(label).cuda()) #* Variable(torch.FloatTensor([roc_reward])).cuda()
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]

        pred_list = []
        gold_list = []
        test_loss = 0
        for _, (_data, _seq_len, _label) in enumerate(dev_loader):
            data, label, seq_len = sort_batch(_data, _label, _seq_len.view(-1))
            y_pred = model(Variable(data, volatile=True).cuda(), seq_len)
            loss = loss_criterion(y_pred, Variable(label).cuda()) #* Variable(torch.FloatTensor([roc_reward])).cuda()
            test_loss += loss.data[0]
            y_pred = y_pred.data.cpu().numpy()
            y_pred = np.asarray([1 & (v > threshold) for v in y_pred])
            pred_list.append(y_pred)  # x[np.where( x > 3.0 )]
            gold_list.append(label.numpy())

        # pred_list_2 = np.concatenate(pred_list, axis=0)[:, 1]
        pred_list = np.concatenate(pred_list, axis=0).argmax(axis=1)
        gold_list = np.concatenate(gold_list, axis=0).argmax(axis=1)
        # roc = roc_auc_score(gold_list, pred_list_2)
        # print('roc:', roc)
        # a = accuracy_score(gold_list, pred_list)
        # p = precision_score(gold_list, pred_list, average='binary')
        # r = recall_score(gold_list, pred_list, average='binary')
        # f1 = f1_score(gold_list, pred_list, average='binary')
        # print('accuracy:', a, 'precision_score:', p, 'recall:', r, 'f1:', f1)
        print("Train Loss: ", train_loss, " Evaluation: ", test_loss)
        es.new_loss(test_loss)
        if es.if_stop():
            print('Start over fitting')
            break

    return gold_list, pred_list
    # pred_list = []
    # for i, (data, seq_len) in tqdm(enumerate(test_loader), total=len(test_data.data)/batch_size):
    #     data, seq_len, rever_sort = sort_batch_test(data, seq_len.view(-1))
    #     y_pred = model(Variable(data, volatile=True).cuda(), seq_len)
    #     pred_list.append(y_pred.data.cpu().numpy()[rever_sort])
    #
    # print('Inference done')
    # re_val = np.concatenate(pred_list, axis=0)
    # return re_val


if __name__ == '__main__':
    # label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    label_cols = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'thankfulness', 'disgust', 'guilt']
    X, y = cbet_data(label_cols)

    # if True:
    #     X, y, X_test, sub_id = cbet_data(label_cols)
    #     with open('tmp', 'bw') as f:
    #         pickle.dump([X, y, X_test, sub_id], f)
    # else:
    #     with open('tmp', 'br') as f:
    #         X, y, X_test, sub_id = pickle.load(f)


    # import pickle
    # with open('tmp', 'bw') as f:
    #     pickle.dump([X_train, X_test, y_train, y_test], f)

    from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
    sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    # golds = np.zeros((int(len(X)*0.1) + 1, y.shape[1]))
    # preds = np.zeros((int(len(X)*0.1) + 1, y.shape[1]))
    y = np.asarray(y[label_cols])
    for train_index, dev_index in sss.split(X, y):
        X_train, X_dev = [X[i] for i in train_index], [X[i] for i in dev_index]
        y_train, y_dev = y[train_index], y[dev_index]
        # class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
        # class_weight = [0.5, 0.5]
        # print(class_weight)
        gold_list, pred_list = one_fold(X_train, y_train, X_dev, y_dev)

    p = precision_score(gold_list, pred_list, average='macro')
    r = recall_score(gold_list, pred_list, average='macro')
    f1 = f1_score(gold_list, pred_list, average='macro')
    print(p, r, f1)
    # with open('preds', 'bw') as f:
    #     pickle.dump(preds, f)
    #
    # prd_1 = pd.DataFrame(preds, columns=y.columns)
    # submit = pd.concat([sub_id, prd_1], axis=1)
    # submit.to_csv('submit.csv', index=False)

    # p = precision_score(gold_list, pred_list, average='macro')
    # r = recall_score(gold_list, pred_list, average='macro')
    # f1 = f1_score(gold_list, pred_list, average='macro')
    # print(p, r, f1)


