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

# SKIP_RATE = .4
#
# PROCESSED_DATA_PATH = "data_processed"
# PLAIN_TEXT_FNAME = os.path.join(PROCESSED_DATA_PATH, "data_processed.txt")
#
# CSV_FNAME = os.path.splitext(PLAIN_TEXT_FNAME)[0] + ".csv"
# OFNAME = os.path.splitext(PLAIN_TEXT_FNAME)[0] + "_" + str(round(100 * SKIP_RATE)) + ".txt"
# OF_CSV_NAME = os.path.splitext(OFNAME)[0] + ".csv"
# OF_SKIPPED_WORDS_NAME = os.path.splitext(OFNAME)[0] + "_skipped.csv"
#


class RemRand:
    def __init__(self, seed=1):
        self.seed = seed
        self.ut = self.NewUniqueToken()

    class NewUniqueToken:
        def __init__(self, seed=4294967296):
            self.seed = seed
            self.it = 0

        def next(self):
            self.it += 1
            return str(self.it ^ self.seed)

    def uniform_sample(self, dict_freqs, num_words_to_skip):
        random.seed(self.seed)
        skip_items = sorted(random.sample(dict_freqs.items(),
                                          num_words_to_skip),
                            key=operator.itemgetter(1),
                            reverse=True)
        return skip_items

    def proportional_sample(self, dict_freqs, num_words_to_skip):
        np.random.seed(self.seed)
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

    def alpha_sample(self, dict_freqs, num_words_to_skip, alpha):
        np.random.seed(self.seed)
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

    def process(self, ifname, ifname_csv, ofname,
                ofname_csv, ofname_csv_skipped, skip_rate=.05):

        with open(ifname_csv, 'r') as f:
            dict_freqs = {rows[0]:int(rows[1]) for rows in csv.reader(f)}

        num_words_to_skip = round(skip_rate * len(dict_freqs))

        skip_items = self.alpha_sample(dict_freqs, num_words_to_skip, 0.1)
        skip_words = set([x[0] for x in skip_items])


        with open(ofname_csv_skipped, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(skip_items)

        with open(ofname_csv, 'w') as f:
            writer = csv.writer(f)
            dict_sorted = sorted(
                   [
                       (k,v) for k,v in dict_freqs.items() if k not in skip_words
                   ],
                   key=operator.itemgetter(1),
                   reverse=True)
            writer.writerows(dict_sorted)

        with open(ofname, 'w') as of:
            with open(ifname, 'r') as f:
                for line in f:
                    for token in line.strip('\n').split(' '):
                        if token in skip_words:
                            of.write("%s " % self.ut.next())
                        else:
                            of.write("%s " % token)
                    of.write('\n')

        print("Saved to", ofname)
        return ofname, ofname_csv


