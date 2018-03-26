import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import numpy as np
from sklearn.ensemble import RandomForestClassifier


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


if __name__ == '__main__':
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


    sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    # golds = np.zeros((int(len(X)*0.1) + 1, y.shape[1]))
    # preds = np.zeros((int(len(X)*0.1) + 1, y.shape[1]))
    y = np.asarray(y[label_cols])
    train_index, dev_index = next(sss.split(X, y))
    X_train, X_dev = [X[i] for i in train_index], [X[i] for i in dev_index]
    y_train, y_dev = y[train_index], y[dev_index]

    clf = RandomForestClassifier(max_depth=2, random_state=0)

