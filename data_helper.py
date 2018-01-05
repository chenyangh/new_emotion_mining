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
MIN_WORD_COUNT = 7
USE_CUDA = True
PAD_token = '<pad>'
UNKNOWN_TOKEN = '<unk>'
PAD_LEN = 30
WORD_EMBEDDING_VOCAB = 30000
EMBEDDING_DIM = 200


class DataHelper:
    def __init__(self):
        self.word2int = {}
        self.int2word = {}
        self.n_words = None
        self.word_embedding = None
        self.data_train = []
        self.data_test = []
        self.label_train = []
        self.label_test = []
        self.min_count = MIN_WORD_COUNT
        self.emb_dim = EMBEDDING_DIM
        self.batch_start = 0
        self.pad_len = PAD_LEN

    @staticmethod
    def normalize_string(text):
        # Clean the text, with the option to remove stopwords and to stem words.

        # Convert words to lower case and split them
        text = text.lower().strip()

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        # Return a list of words
        return (text)

    def load_train_data(self, fold_path):
        def pad_seq(seq):
            seq = [self.word2int[x] if x in self.word2int else '<pad>' for x in seq]
            seq += [PAD_token] * (PAD_LEN - len(seq))
            return seq
        # choose normalize method
        print('Start reading...')
        with open(os.path.join(fold_path, 'train.csv'), 'r') as f:
            self.data_train = [pad_seq(l.strip().split('\t')[1].split())
                               for l in f.readlines()]

    def load_bow_embedding(self, fold_path):
        def build_dcit(_fold_path):
            self.word2int[PAD_token] = 0
            self.int2word[0] = PAD_token
            self.word2int[UNKNOWN_TOKEN] = 1
            self.int2word[1] = UNKNOWN_TOKEN
            f = open(os.path.join(_fold_path, 'vocubulary.txt'), 'r')
            n_words = len(self.word2int)
            for line in f.readlines():
                word = line.strip()
                self.word2int[word] = n_words
                self.int2word[n_words] = word
                n_words += 1
            f.close()
            self.n_words = len(self.word2int)
        build_dcit(fold_path)  # Build vocabulary first

        # Create BOW embedding
        self.word_embedding = nn.Embedding(self.n_words, self.n_words)
        self.word_embedding.weight = nn.Parameter(torch.FloatTensor(np.eye(self.n_words)))
        self.word_embedding.weight.requires_grad = False
        self.load_train_data(fold_path)

    def load_word_embedding(self, fold_path):
        def build_dcit(_fold_path):
            self.word2int[PAD_token] = 0
            self.int2word[0] = PAD_token
            # self.word2int[UNKNOWN_TOKEN] = 1
            # self.int2word[1] = UNKNOWN_TOKEN
            f = open(os.path.join(_fold_path, 'vocubulary.txt'), 'r')
            n_words = len(self.word2int)
            for line in f.readlines():
                word = line.strip()
                self.word2int[word] = n_words
                self.int2word[n_words] = word
                n_words += 1
            f.close()
            self.n_words = len(self.word2int)
        build_dcit(fold_path)  # Build vocabulary first

        self.word_embedding = nn.Embedding(self.n_words, EMBEDDING_DIM)
        import pickle
        emb = np.zeros((self.n_words, EMBEDDING_DIM))
        with open('feature/fasttextModel', 'br') as f:
            model = pickle.load(f)
        embed_dict = model.vocab

        for idx in range(self.n_words):
            word = self.int2word[idx]
            if word in embed_dict:
                vec = model.syn0[embed_dict[word].index]
                emb[idx] = (vec - min(vec)) / np.add(max(vec), -min(vec)) * 2 - 1
            else:
                emb[idx] = np.random.uniform(-1, 1, EMBEDDING_DIM)
        self.word_embedding.weight = nn.Parameter(torch.FloatTensor(emb))
        self.word_embedding.weight.requires_grad = False
    # Pad a with the PAD symbol

    def sequence_batch(self, batch_size):
        start_point = self.batch_start
        end_point = start_point + batch_size
        num_data = len(self.data_train)
        if end_point <= num_data:
            chosen_data = self.data_train[start_point:end_point]
            self.batch_start = end_point
        else:
            chosen_data = self.data_train[start_point:num_data] + self.data_train[0:end_point - num_data]
            self.batch_start = end_point - num_data
        return chosen_data

if __name__ == '__main__':
    dh = DataHelper()
    dh.load_bow_embedding('data/Folds/fold_0/')
    dh.sequence_batch(2)
    tmp = 0
    # dh.load_emotion_data()
    #dh.load_word_embedding()