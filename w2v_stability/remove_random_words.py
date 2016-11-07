#
# input:  plain text
# output: plain text with words randomly replaced by unique tokens
#
import time
import os
import nltk
import csv
import random
import numpy as np
import numpy.random as rnd
import operator

SKIP_RATE = .4

PROCESSED_DATA_PATH = "data_processed"
PLAIN_TEXT_FNAME = os.path.join(PROCESSED_DATA_PATH, "data_processed.txt")

CSV_FNAME = os.path.splitext(PLAIN_TEXT_FNAME)[0] + ".csv"
OFNAME = os.path.splitext(PLAIN_TEXT_FNAME)[0] + "_" + str(round(100 * SKIP_RATE)) + ".txt"
OF_CSV_NAME = os.path.splitext(OFNAME)[0] + ".csv"
OF_SKIPPED_WORDS_NAME = os.path.splitext(OFNAME)[0] + "_skipped.csv"


class NewUniqueToken:
    def __init__(self, seed=4294967296):
        self.seed = seed
        self.it = 0

    def next(self):
        self.it += 1
        return str(self.it ^ self.seed)


with open(CSV_FNAME, 'r') as f:
    dict_freqs = {rows[0]:int(rows[1]) for rows in csv.reader(f)}

print("Number of tokens:", len(dict_freqs))
print("Number of tokens with freq > 10:",
        sum(1 for x in dict_freqs.keys() if dict_freqs[x] > 10))

num_words_to_skip = round(SKIP_RATE * len(dict_freqs))

def uniform_sample(dict_freqs):
    skip_items = sorted(random.sample(dict_freqs.items(),
                                      num_words_to_skip),
                        key=operator.itemgetter(1),
                        reverse=True)
    return skip_items

def proportional_sample(dict_freqs):
    tokens, freqs = map(np.array, zip(*dict_freqs.items()))
    probs = freqs/np.sum(freqs)
    skip_words = set(rnd.choice(tokens, p=probs,
                    size=num_words_to_skip))
    while len(skip_words) < num_words_to_skip:
        skip_words |= {rnd.choice(tokens, p=probs)}
    skip_items = sorted([(x, dict_freqs[x]) for x in skip_words],
                        key=operator.itemgetter(1),
                        reverse=True)
    return skip_items

def alpha_sample(dict_freqs, alpha):
    tokens, freqs = map(np.array, zip(*dict_freqs.items()))
    probs = np.exp(alpha * np.log(freqs))
    probs = probs/np.sum(probs)
    skip_words = set(rnd.choice(tokens, p=probs,
                    size=num_words_to_skip))
    while len(skip_words) < num_words_to_skip:
        skip_words |= {rnd.choice(tokens, p=probs)}
    skip_items = sorted([(x, dict_freqs[x]) for x in skip_words],
                        key=operator.itemgetter(1),
                        reverse=True)
    return skip_items

print("Sampling..")
ts = time.clock()
# skip_items = uniform_sample(dict_freqs)
skip_items = alpha_sample(dict_freqs, 0.1)
skip_words = set([x[0] for x in skip_items])
print("Done in %f s" % (time.clock() - ts))

print("Number of randomly skipped tokens:", len(skip_words))
print("Number of skipped tokens with freq > 10:", sum(1 for x in skip_words if dict_freqs[x] > 10))

print("Writing dict")
ts = time.clock()
with open(OF_SKIPPED_WORDS_NAME, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(skip_items)

with open(OF_CSV_NAME, 'w') as f:
    writer = csv.writer(f)
    dict_sorted = sorted(
           [
               (k,v) for k,v in dict_freqs.items() if k not in skip_words
           ],
           key=operator.itemgetter(1),
           reverse=True)
    writer.writerows(dict_sorted)

print("Done in %f s" % (time.clock() - ts))

ut = NewUniqueToken()

with open(OFNAME, 'w') as of:
    with open(PLAIN_TEXT_FNAME, 'r') as f:
        for line in f:
            for token in line.strip('\n').split(' '):
                if token in skip_words:
                    of.write("%s " % ut.next())
                else:
                    of.write("%s " % token)
            of.write('\n')

print("Saved to", OFNAME)



