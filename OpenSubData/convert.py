# sort s_given_t_dialogue_length2_6.txt | uniq -u | > data_6_remove_dup.txt
import pandas as pd
import numpy as np
vocab_dict = {}
for i, w in enumerate(open('movie_25000', 'r')):
    vocab_dict[i+1] = w.strip()

lines = open('data_6_remove_dup.txt').readlines()

foo_list = []
bar_list = []
for line in lines:
    foo, bar = line.split('|')
    foo_list.append(' '.join([vocab_dict[int(x)] for x in foo.split()]))
    bar_list.append(' '.join([vocab_dict[int(x)] for x in bar.split()]))
    if len(foo_list)>100:
        break

tag = np.zeros((len(foo_list)))
tag[:] = np.nan
df = pd.DataFrame(data={'source': foo_list, 'target': bar_list, 'tag': ['Nan' if x is not np.nan else x for x in tag]})
df = df[['source', 'target', 'tag']]
df.to_csv('data_6_remove_dup.csv', index=False)
