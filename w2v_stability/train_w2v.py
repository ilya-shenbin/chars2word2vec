import gensim
import time
import logging
import random
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

PROCESSED_DATA_PATH = "data_processed"
FNAMES = ["data_processed.txt", "data_processed_20.txt", "data_processed_40.txt"]
#FNAMES = ["data_processed_40.txt"]
SIZE = 200


seed = random.randint(1, 10)
for f in FNAMES:
    sentences = gensim.models.word2vec.LineSentence(os.path.join(PROCESSED_DATA_PATH, f))
    model = gensim.models.Word2Vec(sentences, seed=seed, size=SIZE, iter=10)
    model.save('models/' + f + '_size=' + str(SIZE) + '_seed=' + str(seed) + '.model')
