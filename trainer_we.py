import math
import numpy as np
import logging
import argparse
import os
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
from data_helper import *
from model import LSTMClassifier

dh = DataHelper()
dh.load_bow_embedding('data/Folds/fold_0/')


weight_mask = torch.ones(dh.n_words).cuda()
weight_mask[dh.word2int['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()


model = LSTMClassifier()