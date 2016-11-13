import gensim
import time
import logging
import random
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

PROCESSED_DATA_PATH = "data_processed_mid"
#FNAMES = ["data_processed.txt", "data_processed_20.txt", "data_processed_40.txt"]
#FNAMES = ["data_processed_40.txt"]
SIZE = 200
MIN_COUNT = 100
SG = 0
MODELS_DIR = 'models_mid_100'


FNAMES = [f for f in os.listdir(PROCESSED_DATA_PATH) if f.endswith('.txt')]
FNAMES0 = ["data_processed_mid.txt"]
#FNAMES += ["data_processed_" + str(c) + ".txt" for c in range(5,41,5)]


# for seed in random.sample(range(2**32),1):
#     f = FNAMES0[0]
#     sentences = gensim.models.word2vec.LineSentence(os.path.join(PROCESSED_DATA_PATH, f))
#     model = gensim.models.Word2Vec(sentences, seed=seed, size=SIZE, iter=10)
#     model.save('models_mid/' + f + '_size=' + str(SIZE) + '_seed=' + str(seed) + '.model')
#

for seed in random.sample(range(2**32),1):
    for f in FNAMES0:
        sentences = gensim.models.word2vec.LineSentence(os.path.join(PROCESSED_DATA_PATH, f))
        model = gensim.models.Word2Vec(sentences, seed=seed, size=SIZE,
                sg=SG, min_count=MIN_COUNT, iter=10)
        fn = f + '_size=' + str(SIZE) + '_seed=' + str(seed) + '.model'
        model.save(os.path.join(MODELS_DIR, fn))

for seed in random.sample(range(2**32),1):
    for f in FNAMES:
        sentences = gensim.models.word2vec.LineSentence(os.path.join(PROCESSED_DATA_PATH, f))
        model = gensim.models.Word2Vec(sentences, seed=seed, size=SIZE,
                sg=SG, min_count=MIN_COUNT, iter=10)
        fn = f + '_size=' + str(SIZE) + '_seed=' + str(seed) + '.model'
        model.save(os.path.join(MODELS_DIR, fn))
