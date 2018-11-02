import pickle
import numpy as np
p_file = open('feature/glove.twitter.27B.200d.txt', 'r')

emb_dict = {}
for line in p_file.readlines():
    tokens = line.split()
    word = tokens[0]
    vec = np.asarray([float(x) for x in tokens[1:]])
    emb_dict[word] = vec
p_file.close()


with open('feature/glove.twitter.200d.pkl', 'bw') as f:
    emb_dict = pickle.dump(emb_dict, f)
