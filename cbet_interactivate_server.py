
import torch
from torch.autograd import Variable
from model_cpu import AttentionLSTMClassifier
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# stop_words = set(stopwords.words('english'))
NUM_CLASS = 9


def inference(t, word2id, model):

    t = t.lower()

    word_tokens = word_tokenize(t)
    # filtered_sentence = [w for w in word_tokens if w not in stop_words]
    text = ' '.join(word_tokens)
    tokens = text.split()
    tmp = [word2id[x] if x in word2id else word2id['<unk>'] for x in tokens]
    if len(tmp) == 0:
        tmp = [word2id['<empty>']]

    to_infer = torch.LongTensor([tmp, tmp])
    seq_len = torch.LongTensor([len(tmp), len(tmp)])
    y_pred = model(Variable(to_infer), seq_len)
    y_pred = y_pred.data.numpy()
    return y_pred[0]


def main():
    num_labels = NUM_CLASS
    vocab_size = 30000
    batch_size = 5
    embedding_dim = 200
    hidden_dim = 600
    label_cols = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'thankfulness', 'disgust', 'guilt']
    with open('checkpoint/some_data.pkl', 'br') as f:
        word2id, id2word = pickle.load(f)

    model = AttentionLSTMClassifier(embedding_dim, hidden_dim, vocab_size, word2id,
                                    num_labels, batch_size, use_att=False, soft_last=False)

    model.load_state_dict(torch.load(
        'checkpoint/cbet.model'
    ))
    model.cpu()

    import socket

    s = socket.socket()
    host = socket.gethostname()
    port = 12222
    s.bind((host, port))

    s.listen(5)
    print('Start listening from port', port)
    while True:
        c, addr = s.accept()
        print('Got connection from', addr)
        text = c.recv(1024).decode('utf-8')
        print(text)
        # text = 'I am happy, how about you, great night'
        y_pred = inference(text, word2id, model)
        response = ''
        for emo, prob in zip(label_cols, y_pred):
            if prob > 0:
                response += emo + ":" + str(prob) + '\n'
        q = response.encode('utf-8')
        c.send(q)


main()

