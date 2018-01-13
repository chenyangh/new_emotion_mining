import unicodedata
import string
import re
import random
import gensim

from torch.autograd import Variable
import torch
import torch.nn as nn
import nltk
import numpy as np
import os


def build_vocab(fold_path, vocab_size, use_unk=True):
    word_count = {}
    word2id = {}
    id2word = {}
    with open(os.path.join(fold_path, 'train.csv'), 'r') as f:
        for line in f.readlines():
            tokens = line.split('\t')
            sent = tokens[1]
            for word in sent.split():
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
        if use_unk:
            word2id['<unk>'] = 1
            id2word[1] = '<unk>'
        n = len(word2id)
        word_list = word_list[:vocab_size - n]

        for word in word_list:
            word2id[word] = n
            id2word[n] = word
            n += 1
    return word2id, id2word


if __name__ == '__main__':
    a, b = build_vocab('data/Folds/fold_0', 20000)
    print(a, b)

