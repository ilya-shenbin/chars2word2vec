from scipy import spatial
import csv
import itertools
import numpy as np
import random
import warnings

def check_skipped(model0, model1, topn=100, sample_size=None):
    ###########
    # model0 : etalon model
    ###########
    diff = list()
    vocab0 = set(model0.index2word)
    vocab1 = set(model1.index2word)
    vocab_diff = vocab0 - vocab1

    if sample_size is None:
        test_set = vocab_diff
    elif sample_size > len(vocab_diff):
        warnings.warn('Sample size is larger than vocab.')
        test_set = vocab_diff
    else:
        test_set = random.sample(list(vocab_diff), sample_size)

    for key in test_set:
        vec = model0[key]
        sim0 = set([k for k,v in model0.similar_by_vector(vec, topn=topn) if k in vocab1])
        sim1 = set([k for k,v in model1.similar_by_vector(vec, topn=len(sim0))])
        diff.append(len(sim0 & sim1) / len(sim0))

    return np.array(diff)


def eps_discrete(model0, model1, eps=.7, topn=10, sample_size=None):
    diff = list()
    vocab0 = set(model0.index2word)
    vocab1 = set(model1.index2word)
    vocab_common = vocab0 & vocab1

    if sample_size is None:
        test_set = vocab_common
    elif sample_size > len(vocab_common):
        warnings.warn('Sample size is bigger than vocab.')
        test_set = vocab_common
    else:
        test_set = random.sample(list(vocab_common), sample_size)

    for key in test_set:
        topn_ = topn
        sim0 = model0.most_similar([key], topn=topn)
        if sim0[-1][1] > eps:
            warnings.warn('Last element is closer than eps: %f. Starting iterations.'
                    % sim0[-1][1])

        while sim0[-1][1] > eps:
            topn_ *= 10
            if topn_ > len(vocab0):
                warnings.warn('Epsilon is too small.')
                break
            sim0 = model0.most_similar([key], topn=topn_)
        sim1 = model1.most_similar([key], topn=topn_)
        if sim1[-1][1] > eps:
            warnings.warn('Last element is closer than eps: %f. Starting iterations.'
                    % sim0[-1][1])
        while sim1[-1][1] > eps:
            topn_ *= 10
            if topn_ > len(vocab0):
                warnings.warn('Epsilon is too small.')
                break
            sim1 = model1.most_similar([key], topn=topn_)

        sim_e0 = set([k for k,v in sim0 if v > eps and k in vocab1])
        sim_e1 = set([k for k,v in sim1 if v > eps and k in vocab0])

        len_sd  = len(sim_e0 ^ sim_e1)
        len_un  = len(sim_e0 | sim_e1)
        if len_un > 0:
            diff.append(len_sd/len_un)

    return np.array(diff)


def by_topn_discrete(model0, model1, eps=.7, topn=10, sample_size=None):
    diff = list()
    vocab0 = set(model0.index2word)
    vocab1 = set(model1.index2word)
    vocab_common = vocab0 & vocab1

    if sample_size is None:
        test_set = vocab_common
    elif sample_size > len(vocab_common):
        warnings.warn('Sample size is bigger than vocab.')
        test_set = vocab_common
    else:
        test_set = random.sample(list(vocab_common), sample_size)

    for key in test_set:
        topn_ = topn * 2
        sim0 = [k for k,v in model0.most_similar([key], topn=topn_) if k in vocab1]
        if len(sim0) < topn:
            warnings.warn('Starting iterations.')

        while len(sim0) < topn:
            topn_ *= 10
            if topn_ > len(vocab0):
                warnings.warn('Topn is too big.')
                break
            sim0 = [k for k,v in model0.most_similar([key], topn=topn_) if k in vocab1]

        topn_ = topn * 2
        sim1 = [k for k,v in model1.most_similar([key], topn=topn_) if k in vocab0]
        if len(sim1) < topn:
            warnings.warn('Starting iterations.')

        while len(sim1) < topn:
            topn_ *= 10
            if topn_ > len(vocab0):
                warnings.warn('Topn is too big.')
                break
            sim1 = [k for k,v in model1.most_similar([key], topn=topn_) if k in vocab0]

        sim0 = set(sim0[:topn])
        sim1 = set(sim1[:topn])
        len_sd  = len(sim0 ^ sim1)
        len_un  = len(sim0 | sim1)
        if len_un > 0:
            diff.append(len_sd/len_un)

    return np.array(diff)


def eps_continuous(model0, model1, eps=.7, topn=100):
    diff = list()

    for key in model0.index2word:
        if key in model1.index2word:
            sim0 = dict([(k,v) for k,v in model0.most_similar([key], topn=topn)
                if (v > eps) and (k in model1.index2word)])
            sim1 = dict([(k,v) for k,v in model1.most_similar([key], topn=topn)
                if (v > eps) and (k in model0.index2word)])
            for k in sim0.keys():
                diff.append(np.abs(model0.similarity(key,k) - model1.similarity(key,k)))
    return np.array(diff)


def distance_by_vecs(model0, model1, sample_size=None):
    ###########
    # Cosine distances between vectors corresponding to the same token in model0 and model1:
    #
    #
    ###########
    vocab0 = set(model0.index2word)
    vocab1 = set(model1.index2word)
    vocab_common = vocab0 & vocab1

    if sample_size is None:
        test_set = vocab_common
    elif sample_size > len(vocab_common):
        warnings.warn('Sample size is bigger than vocab.')
        test_set = vocab_common
    else:
        test_set = random.sample(list(vocab_common), sample_size)

    return np.array([spatial.distance.cosine(model0[key], model1[key]) for key in test_set])


def compare_by_closest(model0, model1, topn=10):
    dists_sim = list()
    for key in model0.vocab:
        if key in model1.vocab:
            #sim_k, sim_w = zip(*models[0].most_similar([key]) )
            for (k,w0) in model0.most_similar([key], topn=topn):
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


def form_questions2(model, length=1000, treshold=.7, seed=None, restrict_vocab=None):
    res = list()
    random.seed(seed)
    if restrict_vocab is None:
        vocab = model.index2word
    else:
        vocab = model.index2word[:restrict_vocab]
    while len(res) < length:
        word1, word2 = random.sample(vocab, 2)
        word4, acc = model.most_similar(positive=[word2], negative=[word1],
                topn=1, restrict_vocab=restrict_vocab)[0]
        if acc > treshold:
            print('%s - %s =' % (word2, word1), word4, acc)
            res.append((word1, word2, word4, acc))
    return res

def form_questions(model, length=1000, treshold=.7, seed=None, restrict_vocab=None):
    res = list()
    random.seed(seed)
    if restrict_vocab is None:
        vocab = model.index2word
    else:
        vocab = model.index2word[:restrict_vocab]
    while len(res) < length:
        word1, word2, word3 = random.sample(vocab, 3)
        word4, acc = model.most_similar(positive=[word3, word2], negative=[word1],
                topn=1, restrict_vocab=restrict_vocab)[0]
        if acc > treshold:
            print('%s + %s - %s =' % (word2, word3, word1), word4, acc)
            res.append((word1, word2, word3, word4))
    return res

def check_vect_props(model0, model1, treshold=.5, size=10000, topn=100, seed=None, restrict_vocab=None):
    random.seed(seed)
    vocab0 = set(model0.index2word)
    vocab1 = set(model1.index2word)
    vocab_common = vocab0 & vocab1

    res = list()
    while len(res) < size:
#    for i in range(size):
        w1, w2, w3 = random.sample(vocab_common, 3)
        sim0 = [(k,v) for k,v in model0.most_similar(positive=[w2, w3],
                                      negative=[w1],
                                      topn=topn,
                                      restrict_vocab=restrict_vocab)
                if k in vocab_common and v > treshold]
        sim1 = [(k,v) for k,v in model1.most_similar(positive=[w2, w3],
                                      negative=[w1],
                                      topn=topn,
                                      restrict_vocab=restrict_vocab)
                if k in vocab_common and v > treshold]
        sim_e0 = set([k for k,v in sim0])
        sim_e1 = set([k for k,v in sim1])

        #print(len(sim_e0), len(sim_e1))

        len_sd  = len(sim_e0 ^ sim_e1)
        len_un  = len(sim_e0 | sim_e1)
        if len_un > 0:
            res.append(len_sd/len_un)
            print('%s + %s - %s =' % (w2,w3,w1), sim_e0)
#        else:
#            res.append(np.nan)

    return np.array(res)





