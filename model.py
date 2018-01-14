"""
Copy from https://github.com/jiangqy/LSTM-Classification-Pytorch/blob/master/utils/LSTMClassifier.py
Original author: Qinyuan Jiang, 2017
"""
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import os


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.mask = None
        self.u_w = Variable(torch.randn(dim, 1)).cuda()

    def forward(self, input, context):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        u = F.tanh(self.linear_in(context))  # batch x dim x 1
        # u.view(u.size()[0] * u.size()[1], u.size()[2])
        # Get attention
        attn = F.softmax((u @ self.u_w).squeeze(2), dim=0).unsqueeze(1)
        h_tilde = torch.bmm(attn, context).squeeze(1)
        return h_tilde, attn


class AttentionLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, word2id,
                 label_size, batch_size):
        super(AttentionLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.pad_token_src = word2id['<pad>']
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad_token_src)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        # self.hidden = self.init_hidden()
        self.attention_layer = SoftDotAttention(hidden_dim)
        self.label_size = label_size
        self.expand_size = 100
        self.expand_layer = nn.Linear(hidden_dim, label_size * self.expand_size)
        self.label_max_list = []
        for _ in range(label_size):
            self.label_max_list.append(nn.Linear(self.expand_size, 2).cuda())
        # loss
        #weight_mask = torch.ones(vocab_size).cuda()
        #weight_mask[word2id['<pad>']] = 0
        # self.loss_criterion = nn.BCELoss()

    def init_hidden(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad=False).cuda()
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad=False).cuda()
        return (h0, c0)

    def forward(self, x):
        embedded = self.embeddings(x)
        # = embeds.view(len(sentence), self.batch_size, -1)
        hidden = self.init_hidden(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        out, att = self.attention_layer(hidden, lstm_out)

        # global attention
        # y_pred = self.hidden2label(out) # lstm_out[:, -1:].squeeze(1)
        expand = self.expand_layer(out).view(-1, self.label_size, self.expand_size)
        sm_list = []
        for i in range(self.label_size):
            sm_list.append(F.softmax(self.label_max_list[i](expand[:, i, :]), dim=1))

        y_logit = torch.cat(sm_list, dim=1)
        # loss = self.loss_criterion(nn.Sigmoid()(y_pred), y)
        return y_logit

    def load_glove_embedding(self, id2word):
        """
        :param id2word:
        :return:
        """
        emb = np.zeros((self.vocab_size, self.emb_dim))
        with open('feature/glove.twitter.200d.pkl', 'br') as f:
            emb_dict = pickle.load(f)

        for idx in range(self.vocab_size):
            word = id2word[idx]
            if word in emb_dict:
                vec = emb_dict.syn0[emb_dict[word].index]
                emb[idx] = vec
            else:
                if word == '<pad>':
                    emb[idx] = np.zeros([self.emb_dim])
                else:
                    emb[idx] = np.random.uniform(-1, 1, self.emb_dim)
        self.embedding.weight = nn.Parameter(torch.FloatTensor(emb))

    def load_bog_embedding(self, word2id):
        """"
        :param word2id:
        :return:
        """
        # Create BOW embedding
        emb = np.eye(self.vocab_size)
        emb[word2id['<pad>']] = np.zeros([self.vocab_size])
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(emb))
        # self.word_embedding.weight.requires_grad = False
