
import os
import gensim
import time
import logging
from scipy import spatial
import csv
import itertools
import numpy as np



TOPN = 10

FNAMES = [
            'data_processed.txt_size=200_seed=7.model',
            'data_processed_20.txt_size=200_seed=7.model',
            'data_processed_40.txt_size=200_seed=7.model',
            'data_processed.txt_size=200_seed=2.model',
            'data_processed.txt_size=20_seed=7.model',
            'data_processed_20.txt_size=20_seed=7.model',
            'data_processed_40.txt_size=20_seed=7.model'
        ]

models = [(fname, gensim.models.Word2Vec.load(os.path.join("models", fname)))
          for fname in FNAMES]


def print_stats(arr):
    print("\tNumber of entries:", arr.size)
    print("\tmedian:", np.median(arr))
    print("\tmean:", np.mean(arr))
    print("\tvariance:", np.var(arr))
    print("\tstd deviation:", np.std(arr))
    print("\tmin:", np.amin(arr))
    print("\tmax:", np.amax(arr))


def compare_by_vecs(model0, model1):
    dists = np.array([spatial.distance.cosine(model0[key], model1[key])
                for key in model0.vocab if key in model1.vocab])

    print("Shared tokens:", dists.size)
    print("\nCosine distances between vectors corresponding to the same token in model0 and model1:")
    print_stats(dists)

def compare_by_closest(model0, model1):
    dists_sim = list()
    for key in model0.vocab:
        if key in model1.vocab:
            #sim_k, sim_w = zip(*models[0].most_similar([key]) )
            for (k,w0) in model0.most_similar([key], topn=TOPN):
                if k in model1.vocab:
                    w1 = model1.similarity(key,k)
                    dists_sim.append(np.abs(w0 - w1))

    dists_sim = np.array(dists_sim)
    print("\nDistances between each token in vocab and its %i most similar. Abs deviation of distances between models:" % TOPN)
    print_stats(dists_sim)


def print_similar_csect(token, model0, model1):
    print("\nMost similar to token '%s'" % token)
    print([(x,f) for x,f in model0.most_similar([token]) if x in model1.vocab])
    print([(x,f) for x,f in model1.most_similar([token]) if x in model0.vocab])

def print_compare(x0, x1):
    model0 = x0[1]
    model1 = x1[1]
    print("----------")
    print("model0:", x0[0])
    print("model1:", x1[0])
    print("vocab size 0:", len(model0.vocab))
    print("vocab size 1:", len(model1.vocab))
    #print(model0.vocab)
    if model0[next(iter(model0.vocab))].size == model1[next(iter(model1.vocab))].size:
        compare_by_vecs(model0, model1)
    compare_by_closest(model0, model1)

print_compare(models[0], models[1])
print_compare(models[0], models[2])
print_compare(models[0], models[3])
print_compare(models[0], models[4])
print_compare(models[4], models[5])
print_compare(models[4], models[6])



