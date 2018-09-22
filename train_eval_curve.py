bs1 = 600
bs2 = 50

train_size = 5328194
test_size = 280432

dir_path = 'tmp_fold/'


def parse_file(f, bs, averaged=False):
    lines = open(dir_path + f + '.txt', 'r').readlines()
    train_loss_list = []
    eval_loss_list = []
    for line in lines:
            if line.startswith('Training Loss'):
                train_loss = float(line.split()[-1])
                if averaged:
                    train_loss = train_loss * train_size / bs
                train_loss_list.append(train_loss)
            if line.startswith('Evaluation Loss'):
                eval_loss = float(line.split()[-1])
                if averaged:
                    eval_loss = eval_loss * test_size
                eval_loss_list.append(eval_loss)

    assert len(eval_loss_list) == len(train_loss_list)
    return train_loss_list, eval_loss_list


def draw(f, train_loss_list, eval_loss_list):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Create some mock data
    t = range(1, len(train_loss_list) + 1)
    data1 = train_loss_list
    data2 = eval_loss_list

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of epoch')
    ax1.set_ylabel('Training loss', color=color)
    ax1.plot(t, data1, color=color, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Evaluation loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()
    pp = PdfPages(dir_path + f + '_train_eval.pdf')
    pp.savefig(fig)
    pp.close()
    # plt.imsave


f_list = ['foo', 'bar']
for f in f_list:
    train_loss_list, eval_loss_list = parse_file(f, bs1)
    draw(f, train_loss_list, eval_loss_list)

f = 'persona'
train_loss_list, eval_loss_list = parse_file(f, bs1, averaged=True)
draw(f, train_loss_list, eval_loss_list)

# f = 'trans'
# train_loss_list, eval_loss_list = parse_file(f, bs2, averaged=True)
# draw(f, train_loss_list, eval_loss_list)
