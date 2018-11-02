import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def cbet_data(label_cols):
    data = pd.read_csv('data/CBET.csv')
    label = data[label_cols]

    stop_words = set(stopwords.words('english'))

    train_text = []
    for t in data['text'].fillna("fillna").values:
        t = t.lower()
        word_tokens = word_tokenize(t)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        train_text.append(' '.join(filtered_sentence))

    return train_text, label


if __name__ == '__main__':
    label_cols = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'thankfulness', 'disgust', 'guilt']
    X, y = cbet_data(label_cols)

    sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    y = np.asarray(y[label_cols])
    train_index, dev_index = next(sss.split(X, y))
    X_train, X_dev = [X[i] for i in train_index], [X[i] for i in dev_index]
    y_train, y_dev = y[train_index], y[dev_index]

    bag_of_words_len = 5000
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
                                 preprocessor=None,
                                 max_features=bag_of_words_len)

    x_train_fea = vectorizer.fit_transform(X_train)
    x_train_fea = x_train_fea.toarray()
    x_dev_fea = vectorizer.transform(X_dev)
    x_dev_fea = x_dev_fea.toarray()

    # clf = RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=-1)
    # clf = OneVsRestClassifier(SVC(kernel='linear'), n_jobs=-1)
    clf = OneVsRestClassifier(SVC(kernel='rbf', probability=True), n_jobs=-1)
    clf.fit(x_train_fea, y_train)

    y_dev_pred = clf.predict(x_dev_fea)
    result = clf.predict_proba(x_dev_fea)
    import pickle
    with open('result_prob.pkl', 'bw') as f:
        pickle.dump([result], f)

    f1 = f1_score(y_dev, y_dev_pred, average='macro')
    p = precision_score(y_dev, y_dev_pred, average='macro')
    r = recall_score(y_dev, y_dev_pred, average='macro')
    print(f1, p, r)

    with open('result_measure.txt', 'w') as f:
        f.write(str(f1) + ' ' + str(p) + ' ' + str(r) + '\n')
