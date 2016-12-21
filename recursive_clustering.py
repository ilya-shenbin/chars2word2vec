import random
import numpy as np
import os
import pickle
from gensim.models import Word2Vec
from sklearn.cluster.k_means_ import MiniBatchKMeans
from sklearn.cluster.birch import Birch
from sklearn.cluster.dbscan_ import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster.mean_shift_ import MeanShift
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.externals.joblib.memory import Memory
import matplotlib.pyplot as plt

#model_fname = 'models_mid_sg/data_processed_mid.txt_size=200_seed=1739656695.model'

model_fname = os.path.join('models_ok', 'ok-20161206.w2v.bi.300.bin')
model_fname = os.path.join('models_ok', 'ok-20161206.w2v.300.bin')
model_fname = os.path.join('models_ok', 'ok-20161123.w2v.200.bin')
model_fname = os.path.join('models_ok', 'ok-20161123.w2v.100.bin')
model_fname = os.path.join('models_ok', 'ok-20161206.w2v.300.bin')
model = Word2Vec.load(model_fname)

# model_fname = 'news.model.bin.gz'
# model = Word2Vec.load_word2vec_format(
#             'models_rusvec/news.model.bin.gz'
#             , binary=True
#             , encoding='utf-8')

print('data shape: ', model.syn0.shape)

p = np.random.permutation(len(model.index2word))
keys = [model.index2word[k] for k in p]
vals = model.syn0[p]
norm = np.linalg.norm(vals, axis=1)
vals = vals / norm[:, None]

# for i in p:
#     assert np.allclose(model[keys[i]], vals[i])

clr_sign = "MiniBatchKMeansi_test_norm"

def population_hist(population, fname=None):

    plt.hist(population, bins=10**np.linspace(0, 4, 100))
    plt.xscale('log')
    #plt.yscale('log', nonposy='clip')
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


class Tree():
    def __init__(self, X, ids, center, max_size=100):
        print('Creating node with len(data) =', len(ids))
        self.ids = None
        self.center = None
        self.children = []
        self.X = X

        if ids is not None:
            if len(ids) > max_size:
                n_clusters = -(-len(ids) // max_size)     # ceiling
                clusters_map, centers = self.cluster(ids, n_clusters)
                for k in clusters_map.keys():

                    # in order to support DBSCAN
                    if k != -1:
                        self.children.append(Tree(X, clusters_map[k], centers[k], max_size=max_size))
            else:
                self.ids = ids
                self.center = center

    def cluster(self, ids, n_clusters):
        X = self.X[ids]

        if X.shape[0] > 1000:
            clr = MiniBatchKMeans(n_clusters=n_clusters, verbose=0, init_size=10*n_clusters)
            clusters = clr.fit(X)
            labels = clusters.labels_
            centers = clusters.cluster_centers_
        else:
            clr = KMeans(n_clusters=n_clusters, verbose=0)
            clusters = clr.fit(X)
            labels = clusters.labels_
            centers = clusters.cluster_centers_

            # #too many small clusters. Also, works as shit.
            # clr = Birch(n_clusters=n_clusters)
            # clusters = clr.fit(X)
            # labels = clusters.labels_
            # centers = clusters.subcluster_centers_
            #
            # #does not work properly
            # clr = DBSCAN(algorithm='kd_tree', eps=1.0)
            # clusters = clr.fit(X)
            # labels = clusters.labels_
            # centers = clusters.components_
            #
            # #too many small clusters. Also, works as shit.
            # clr =AffinityPropagation()
            # clusters = clr.fit(X)
            # labels = clusters.labels_
            # centers = clusters.cluster_centers_

        clusters_map = {}

        for id in range(len(labels)):
            if not labels[id] in clusters_map:
                clusters_map[labels[id]] = [id]
            else:
                clusters_map[labels[id]] += [id]
        return clusters_map, centers

    def get_clusters(self):
        clusters = []
        if len(self.children) > 0:
            for child in self.children:
                clusters += child.get_clusters()
        elif (self.ids is not None) and (self.center is not None):
            clusters.append((self.ids, self.center))
        else:
            print('get_clusters error: WTF?!?')
        return clusters

# recursively build tree
root = Tree(vals, list(range(vals.shape[0])), None, max_size=200)

# get clusters in form (list of word indexes, center)
clusters = root.get_clusters()

centers  = np.empty((0, model.syn0.shape[1]))
id_label = []
key_label = []
label_center = []
clusters_map = {}

label_cnt = 0
for ids, center in clusters:
    centers = np.concatenate((centers, center[None,...]))
    label_center.append((label_cnt, center))

    current_keys = [keys[i] for i in ids]
    labels = len(ids) * [label_cnt]
    key_label += zip(current_keys, labels)

    clusters_map[label_cnt] = ids

    label_cnt += 1


assert centers.shape[0] == len(clusters)
print('centers.shape =', centers.shape)
ofname = os.path.join("results",
                       os.path.basename(model_fname) + '_' +  clr_sign + ".txt")

np.save(ofname[:-4] + '_centers.npy', centers)

with open(ofname[:-4] + '_word-labels.pickle', 'wb') as f:
    pickle.dump(key_label, f)


population = np.array([len(v) for v in clusters_map.values()])

print('population')
print('\tmax:', np.amax(population))
print('\tmin:', np.amin(population))
print('\tmedian:', np.median(population))
population_hist(population, fname=ofname + '.png')

print("writing stuff to file:", ofname)
with open(ofname, "w") as f:
    for cluster_id, ids in clusters_map.items():
        f.write("Cluster " + str(cluster_id) + "\n")
        for id in ids:
            f.write(keys[id] + "\t")
        f.write("\n")

