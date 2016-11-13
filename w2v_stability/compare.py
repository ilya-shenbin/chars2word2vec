import os
import gensim
import time
import logging
import numpy as np
import metrics
import pickle




MODELS_DIR = 'models_mid_100'

fname = 'data_processed_mid.txt_size=200_seed=1194528071.model'

FNAMES = [
    'data_processed_mid.txt_size=200_seed=3024480899.model',
    'data_processed_mid_2.txt_size=200_seed=1194528071.model',
    'data_processed_mid_4.txt_size=200_seed=1194528071.model',
    'data_processed_mid_6.txt_size=200_seed=1194528071.model',
    'data_processed_mid_8.txt_size=200_seed=1194528071.model',
    'data_processed_mid_10.txt_size=200_seed=1194528071.model',
    'data_processed_mid_12.txt_size=200_seed=1194528071.model',
    'data_processed_mid_14.txt_size=200_seed=1194528071.model',
    'data_processed_mid_16.txt_size=200_seed=1194528071.model',
    'data_processed_mid_18.txt_size=200_seed=1194528071.model',
    'data_processed_mid_20.txt_size=200_seed=1194528071.model',
    'data_processed_mid_22.txt_size=200_seed=1194528071.model',
    'data_processed_mid_24.txt_size=200_seed=1194528071.model',
    'data_processed_mid_26.txt_size=200_seed=1194528071.model',
    'data_processed_mid_28.txt_size=200_seed=1194528071.model',
        ]



"""
MODELS_DIR = 'models_mid_sg'

fname = 'data_processed_mid.txt_size=200_seed=1739656695.model'

FNAMES = [
    'data_processed_mid.txt_size=200_seed=2211067460.model',
    'data_processed_mid_2.txt_size=200_seed=1739656695.model',
    'data_processed_mid_4.txt_size=200_seed=1739656695.model',
    'data_processed_mid_6.txt_size=200_seed=1739656695.model',
    'data_processed_mid_8.txt_size=200_seed=1739656695.model'
    'data_processed_mid_10.txt_size=200_seed=1739656695.model',
    'data_processed_mid_12.txt_size=200_seed=1739656695.model',
    'data_processed_mid_14.txt_size=200_seed=1739656695.model',
    'data_processed_mid_16.txt_size=200_seed=1739656695.model',
    'data_processed_mid_18.txt_size=200_seed=1739656695.model',
    'data_processed_mid_20.txt_size=200_seed=1739656695.model',
    'data_processed_mid_22.txt_size=200_seed=1739656695.model',
    'data_processed_mid_24.txt_size=200_seed=1739656695.model',
    'data_processed_mid_26.txt_size=200_seed=1739656695.model',
    'data_processed_mid_28.txt_size=200_seed=1739656695.model',
        ]
MODELS_DIR = 'models_mid'

fname = 'data_processed_mid.txt_size=200_seed=2861750525.model'

FNAMES = [
    'data_processed_mid.txt_size=200_seed=1390715074.model',
    'data_processed_mid_2.txt_size=200_seed=2861750525.model',
    'data_processed_mid_4.txt_size=200_seed=2861750525.model',
    'data_processed_mid_6.txt_size=200_seed=2861750525.model',
    'data_processed_mid_8.txt_size=200_seed=2861750525.model',
    'data_processed_mid_10.txt_size=200_seed=2861750525.model',
    'data_processed_mid_12.txt_size=200_seed=2861750525.model',
    'data_processed_mid_14.txt_size=200_seed=2861750525.model',
    'data_processed_mid_16.txt_size=200_seed=2861750525.model',
    'data_processed_mid_18.txt_size=200_seed=2861750525.model',
    'data_processed_mid_20.txt_size=200_seed=2861750525.model',
    'data_processed_mid_22.txt_size=200_seed=2861750525.model',
    'data_processed_mid_24.txt_size=200_seed=2861750525.model',
    'data_processed_mid_26.txt_size=200_seed=2861750525.model',
    'data_processed_mid_28.txt_size=200_seed=2861750525.model',
        ]
"""

model0 = gensim.models.Word2Vec.load(os.path.join(MODELS_DIR, fname))

print(model0.most_similar(positive=['пять','шесть'], negative=['один']))
print(model0.most_similar(positive=['пять','шесть'], negative=['сто']))

metrics.eps_discrete(model0, model0, eps=.01)
metrics.form_questions(model0, treshold=.75,restrict_vocab=10000)


# def check_skipped(model0, model1, topn=100, sample_size=None):
# def eps_discrete(model0, model1, eps=.7, topn=10, sample_size=None):
# def distance_by_vecs(model0, model1):
# def check_vect_props(model0, model1, treshold=.9, size=10000, seed=None, restrict_vocab=None):

check_skipped = list()
eps_discrete = list()
dist_vec = list()
vec_props = list()

for f in FNAMES:
    print(f)
    ts0 = time.clock()
    model1 = gensim.models.Word2Vec.load(os.path.join(MODELS_DIR, f))

    ts = time.clock()
    print('\tcheck_skipped')
    check_skipped.append(metrics.check_skipped(model0, model1, topn=100, sample_size=None))
    print('\t\tdone in %f sec' % (time.clock() - ts))

    ts = time.clock()
    print('\teps_discrete')
    eps_discrete.append(metrics.eps_discrete(model0, model1, eps=.7, topn=20, sample_size=None))
    print('\t\tdone in %f sec' % (time.clock() - ts))

    ts = time.clock()
    print('\tdist_vec')
    dist_vec.append(metrics.distance_by_vecs(model0, model1))
    print('\t\tdone in %f sec' % (time.clock() - ts))

#    ts = time.clock()
#    print('\tvec_props')
#    vp = metrics.check_vect_props(model0, model1, treshold=.8, size=1000, seed=None, restrict_vocab=100000)
#    print(vp)
#    vec_props.append(vp)
#    print('\t\tdone in %f sec' % (time.clock() - ts))

    print('Done in %f sec' % (time.clock() - ts0))


with open( '100_check_skipped.pickle', 'wb') as f:
    pickle.dump(check_skipped,f)
with open( '100_eps_discrete.pickle', 'wb') as f:
    pickle.dump(eps_discrete, f)
with open('100_dist_vec.pickle', 'wb') as f:
    pickle.dump(dist_vec, f)
#with open('vec_props.pickle', 'wb') as f:
#    pickle.dump(vec_props, f)








"""
import random
models = [gensim.models.Word2Vec.load(os.path.join("models_mid", fname))
              for fname in FNAMES]

vocab0 = set(models[0].index2word)
vocab1 = set(models[1].index2word)
vocab_diff = vocab0 - vocab1
print(len(vocab_diff))

key = list(vocab_diff)[100]
print(key)
vec = models[0][key]
#print(models[0].most_similar([key]))
sim0 = set([k for k,v in models[0].similar_by_vector(vec, topn=100) if k in vocab1])
sim1 = set([k for k,v in models[1].similar_by_vector(vec, topn=len(sim0))])
print(len(sim0), len(sim1), len(sim0 & sim1))

print(sim0)
print(sim1)

#print(models[1].most_similar(positive=['муж','она'], negative=['он']))
#print(models[0].most_similar(positive=['муж','она'], negative=['он']))
#print(models[1].most_similar(positive=['девочка']))

#t =  m_vect(models[0], models[1], treshold=.7, size=10000, seed=None, restrict_vocab=10000)
#print("%i (%i) %f (%f)" % (t.size, np.count_nonzero(~np.isnan(t)), np.nanmean(t), np.nanvar(t)))
"""


