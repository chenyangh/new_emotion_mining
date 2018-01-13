import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import AttentionLSTMClassifier
from torch.utils.data import Dataset, DataLoader


MIN_WORD_COUNT = 7
USE_CUDA = True
PAD_token = '<pad>'
UNKNOWN_TOKEN = '<unk>'
PAD_LEN = 30
WORD_EMBEDDING_VOCAB = 30000
EMBEDDING_DIM = 200


class EmotionDataLoader(Dataset):
    def __init__(self, csv_file, tag_file, pad_len, word2int, max_size=None):
        tmp_csv = pd.read_csv(csv_file)
        self.source = tmp_csv['source']
        self.target = tmp_csv['target']

        self.pad_len = pad_len
        self.start_int = word2int['<s>']
        self.eos_int = word2int['</s>']
        self.pad_int = word2int['<pad>']
        assert len(self.tag) == len(self.source)
        if max_size is not None:
            self.source = self.source[:max_size]
            self.target = self.target[:max_size]
            self.tag = self.tag[:max_size]

    def __len__(self):
        return len(self.tag)

    def __getitem__(self, idx):
        # for src add <s> ahead
        src = [int(x) for x in self.source[idx].split()]
        if len(src) > self.pad_len:
            src = src[:self.pad_len]
        src = src + [self.pad_int] * (self.pad_len - len(src))

        # for trg add <s> ahead and </s> end
        trg = [int(x) for x in self.target[idx].split()]
        if len(trg) > self.pad_len - 2:
            trg = trg[:self.pad_len-2]
        trg = [self.start_int] + trg + [self.eos_int] + [self.pad_int] * (self.pad_len - len(trg) - 2)
        if not len(src) == len(trg) == self.pad_len:
            print(src, trg)
        assert len(src) == len(trg) == self.pad_len
        tag = self.tag[idx]
        return torch.LongTensor(src), torch.LongTensor(trg), torch.LongTensor([tag])


if __name__ == '__main__':


    weight_mask = torch.ones(dh.n_words).cuda()
    weight_mask[dh.word2int['<pad>']] = 0
    loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()

    model = AttentionLSTMClassifier().cuda()

