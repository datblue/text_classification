import numpy as np
import pickle


vector_dir = 'embedding/vectors.npy'
word_dir = 'embedding/words.pl'

embedd_vectors = np.load(vector_dir)
with open(word_dir, 'rb') as handle:
    embedd_words = pickle.load(handle)
embedd_dim = np.shape(embedd_vectors)[1]
# gen embedding vector for unknown token
unknown_embedd = np.random.uniform(-0.01, 0.01, (1, embedd_dim))


def get_embedding(word):
    w = word.lower()
    try:
        embedd = embedd_vectors[embedd_words.index(w)]
    except:
        embedd = unknown_embedd
    return embedd