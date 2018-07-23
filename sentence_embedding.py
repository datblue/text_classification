# -*- encoding: utf-8 -*-

import sent2vec
import numpy as np
import my_map



model = sent2vec.Sent2vecModel()
model.load_model('sent2vec.bin')
emb_size = model.get_emb_size()


def create_vector_data(sentences, max_sentence):
    nsamples = 0
    y = []
    sens = []
    for cat in sentences:
        nsamples += len(sentences[cat])
        sens += sentences[cat]
        y += [my_map.name2label[cat] for _ in xrange(len(sentences[cat]))]

    X = np.zeros((nsamples, max_sentence, emb_size))
    for i, s in enumerate(sens):
        embs = model.embed_sentences(s)
        for j, emb in enumerate(embs):
            X[i, j] = emb

    return X, y
