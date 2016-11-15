import os
import time
import numpy as np
import pickle
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt




def get_stats(arr):
    return np.array([
            np.nanmean(arr),
            np.nanvar(arr),
            np.nanmedian(arr),
            np.nanstd(arr),
            arr.shape[0]
        ])


with open( 'sg_check_skipped.pickle', 'rb') as f:
    sg_cs = pickle.load(f)
with open( 'sg_eps_discrete.pickle', 'rb') as f:
    sg_ed = pickle.load(f)
with open('sg_dist_vec.pickle', 'rb') as f:
    sg_dv = pickle.load(f)
with open('sg_topn_discrete.pickle', 'rb') as f:
    sg_tn = pickle.load(f)
with open( 'check_skipped.pickle', 'rb') as f:
    cs = pickle.load(f)
with open( 'eps_discrete.pickle', 'rb') as f:
    ed = pickle.load(f)
with open('dist_vec.pickle', 'rb') as f:
    dv = pickle.load(f)
with open('topn_discrete.pickle', 'rb') as f:
    tn = pickle.load(f)
with open('100_check_skipped.pickle', 'rb') as f:
    cs_100 = pickle.load(f)
with open('100_eps_discrete.pickle', 'rb') as f:
    ed_100 = pickle.load(f)
with open('100_dist_vec.pickle', 'rb') as f:
    dv_100 = pickle.load(f)
with open('100_topn_discrete.pickle', 'rb') as f:
    tn_100 = pickle.load(f)

cs_stats = np.array([get_stats(k) for k in cs])
ed_stats = np.array([get_stats(k) for k in ed])
dv_stats = np.array([get_stats(k) for k in dv])
tn_stats = np.array([get_stats(k) for k in tn])

sg_cs_stats = np.array([get_stats(k) for k in sg_cs])
sg_ed_stats = np.array([get_stats(k) for k in sg_ed])
sg_dv_stats = np.array([get_stats(k) for k in sg_dv])
sg_tn_stats = np.array([get_stats(k) for k in sg_tn])

cs_100stats = np.array([get_stats(k) for k in cs_100])
ed_100stats = np.array([get_stats(k) for k in ed_100])
dv_100stats = np.array([get_stats(k) for k in dv_100])
tn_100stats = np.array([get_stats(k) for k in tn_100])

def plot_it(x, arr_y, labels=None, fname=None, figtext='', loc=1):
    if labels is not None:
        assert len(arr_y) == len(labels)
    else:
        labels = [None] * len(arr_y)

    plt.figure()
    plt.gca().set_position((.1, .3, .8, .7)) # to make a bit of room for extra text

    for y,lab in zip(arr_y, labels):
        plt.errorbar(x, y[:,0], yerr=y[:,1], fmt='-o', label=lab)

    plt.legend(loc=loc)
    plt.xlim((-1, 30))
    #plt.figtext(0,0,figtext)
    #plt.plot(x, y[:,2], '-o', label='median')

    plt.figtext(.02, .02, figtext,fontsize=16)
    plt.xlabel("% skipped")

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

ed_capt = """$m(model_0, model_1) = \\frac{1}{|Vocab_0\\cap Vocab_1|}\\sum_{k\in Vocab_0\\cap Vocab_1} \\frac{|M_0(k) \\Delta M_1(k)|}{|M_0(k) \\cup M_1(k)|}$,
where $M_{0,1}(k)$ is a set of words from $Vocab_0\\cap Vocab_1$ with cosine
similarity to k greater than eps, for eps = 0.7"""

cs_capt = """$m(model_0, model_1) = \\frac{1}{|Vocab_0 - Vocab_1|}\\sum_{k\in Vocab_0 - Vocab_1} \\frac{|M_0(k) \\cap M_1(k)|}{|M_0(k)|}$,
where $M_0(k)$ is a set of topn=100 most similar to $k$ words from
$Vocab_0\\cap Vocab_1$ in model0, $M_1(k)$ is a set of words in model1,
most similar to vector $V_0(k)$ from model0."""

dv_capt = """Mean cosine similarity between vectors in model0 and model1
for each token from $Vocab_0\\cap Vocab_1$."""

tn_capt = """$m(model_0, model_1) = \\frac{1}{|Vocab_0\\cap Vocab_1|}\\sum_{k\in Vocab_0\\cap Vocab_1} \\frac{|M_0(k) \\Delta M_1(k)|}{|M_0(k) \\cup M_1(k)|}$,
where $M_{0,1}(k)$ is a set of topn words from $Vocab_0\\cap Vocab_1$,
for topn = 20"""


#print(ed_capt)
plot_it(np.arange(0,30,2), [tn_stats, sg_tn_stats, tn_100stats],
        labels=['CBOW, min_count=5', 'sg, min_count=5', 'CBOW, min_count=100'],
        figtext=tn_capt, fname='_plot_tn.png', loc=4)
plot_it(np.arange(0,30,2), [ed_stats, sg_ed_stats, ed_100stats],
        labels=['CBOW, min_count=5', 'sg, min_count=5', 'CBOW, min_count=100'],
        figtext=ed_capt, fname='_plot_ed.png', loc=4)
plot_it(np.arange(0,30,2), [cs_stats, sg_cs_stats, cs_100stats],
        labels=['CBOW, min_count=5', 'sg, min_count=5', 'CBOW, min_count=100'],
        figtext=cs_capt, fname='_plot_cs.png')
plot_it(np.arange(0,30,2), [dv_stats, sg_dv_stats, dv_100stats],
        labels=['CBOW, min_count=5', 'sg, min_count=5', 'CBOW, min_count=100'],
        figtext=dv_capt, fname='_plot_dv.png')

#print(dv_stats)
