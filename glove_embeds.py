#This script processes and generates GloVe embeddings
#taken from: https://github.com/ajamjoom/Image-Captions/blob/master/glove_embeds.py
#run "wget -P glove.6B http://nlp.stanford.edu/data/glove.6B.zip"
#unzip
#pip install bcolz
#then run Run the glove_embeds.py file - this will generate glove_words.pkl in the glove.6B folder
import pickle
from pathconf import PathConfig

import os
import sys
sys.path.append('cocoapi/PythonAPI/')
#from vocabulary import Vocabulary

import numpy as np
import json
from scipy import misc
import bcolz
import torch

def generate_glove_vectors():
    words = []
    idx = 0
    w2i = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='glove.6B/6B.300.dat', mode='w')

    with open('glove.6B/glove.6B.300d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            w2i[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
        
    vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir='glove.6B/6B.300.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open('glove.6B/6B.300_words.pkl', 'wb'))
    pickle.dump(w2i, open('glove.6B/6B.300_idx.pkl', 'wb'))

    with open(PathConfig.vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    print('Loading vocab...')

    vectors = bcolz.open('glove.6B/6B.300.dat')[:]
    words = pickle.load(open('glove.6B/6B.300_words.pkl', 'rb'))
    w2i = pickle.load(open('glove.6B/6B.300_idx.pkl', 'rb'))

    print('glove is loaded...')

    glove = {w: vectors[w2i[w]] for w in words}
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0

    for i, word in enumerate(vocab.i2w):
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))

    pickle.dump(weights_matrix, open('glove.6B/glove_words.pkl', 'wb'), protocol=2)

    print('weights_matrix is created')

def load_glove_vectors():
    
    glove_vectors = pickle.load(open('glove.6B/glove_words.pkl', 'rb'))
    glove_vectors = torch.tensor(glove_vectors)

    return glove_vectors