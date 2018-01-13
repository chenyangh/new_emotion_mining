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
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

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
        self.last_layer = nn.Linear(hidden_dim, label_size * 100)
        # loss
        weight_mask = torch.ones(vocab_size).cuda()
        weight_mask[word2id['<pad>']] = 0
        self.loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)

    def init_hidden(self):
        h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        return (h0, c0)

    def forward(self,  x, y):
        embedded = self.embeddings(x)
        # = embeds.view(len(sentence), self.batch_size, -1)
        hidden = self.init_hidden()
        lstm_out, hidden = self.lstm(embedded, hidden)
        out, att = self.attention_layer(hidden, lstm_out)

        # global attention

        y = self.hidden2label(lstm_out[-1])
        return y

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
