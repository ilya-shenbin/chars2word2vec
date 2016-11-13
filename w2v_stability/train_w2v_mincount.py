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
MODELS_DIR = 'models_mid_mincount'


FNAMES0 = ["data_processed_mid.txt"]

for seed in random.sample(range(2**32), 2):
    for f in FNAMES0:
        for min_count in range(10, 110, 10):
            sentences = gensim.models.word2vec.LineSentence(
                    os.path.join(PROCESSED_DATA_PATH, f))
            model = gensim.models.Word2Vec(sentences, seed=seed, size=SIZE,
                    sg=SG, min_count=min_count, iter=10)
            fn = f + 'cnt=' + str(min_count) + '_size=' + str(SIZE) + '_seed=' + str(seed) + '.model'
            model.save(os.path.join(MODELS_DIR, fn))

