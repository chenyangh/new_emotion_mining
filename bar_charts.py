import numpy as np
import matplotlib.pyplot as plt


def tec_dis():
    def tec_data():
        txt = []
        emo = []
        for line in open('data/TEC.txt', 'r').readlines():
            tokens = line.split()
            txt.append(' '.join(tokens[:-2]))
            emo.append(tokens[-1])
        return txt, emo

    txt, emo = tec_data()
    dic_count = {}
    for e in emo:
        if e in dic_count:
            dic_count[e] += 1
        else:
            dic_count[e] = 1
    performance = [dic_count[str(i)] for i in range(len(dic_count))]

    emotions = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust']
    ind = range(len(emotions))

    plt.bar(ind, performance, edgecolor='black', hatch="/")
    plt.xticks(ind, emotions)


def group_bar_char():
    data = [[42.48, 38.70, 40.50],
            # [47.07, 47.04, 47.05],
            [47.23, 47.17, 47.20],
            [50.27, 48.73, 49.49],
            # [45.31, 49.47, 44.29],
            [53.61, 52.69, 52.63]]
    data = np.asarray(data)
    fig, ax = plt.subplots()
    N = 4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2  # the width of the bars
    p1 = ax.bar(ind, data[:, 0], width, color='w', edgecolor='black', hatch="/")

    p2 = ax.bar(ind + width, data[:, 1], width, color='w', edgecolor='black', hatch="o")

    p3 = ax.bar(ind + width * 2, data[:, 2], width, color='w', edgecolor='black', hatch="\\")


    # ax.set_title('Scores by group and gender')
    ax.set_xticks(ind + width)
    plt.xticks(fontsize=14)
    ax.set_xticklabels(('Lexical', 'SVM(Informative BOW)', 'NB(Informative BOW)', 'LSTM-Att(WE)'))

    ax.legend((p1[0], p2[0], p3[0]), ('Precision', 'Recall', 'F1-score'), prop={'size': 12})
    ax.autoscale_view()

    plt.show()

group_bar_char()