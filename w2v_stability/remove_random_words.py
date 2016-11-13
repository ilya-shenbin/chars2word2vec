#
# input:  plain text
# output: plain text with words randomly replaced by unique tokens
#
import time
import os
#import nltk
#import csv
#import random
#import numpy as np
#import numpy.random as rnd
#import operator

from RemRand import RemRand

SKIP_RATE = .02

PROCESSED_DATA_PATH = "data_processed_mid"
PLAIN_TEXT_FNAME = os.path.join(PROCESSED_DATA_PATH, "data_processed_mid.txt")


rr = RemRand()

ifname = PLAIN_TEXT_FNAME
ifname_csv = os.path.splitext(PLAIN_TEXT_FNAME)[0] + ".csv"

for i in range(1, 15):
    current_skip = i * SKIP_RATE
    print("Starting step",i)
    ts = time.clock()
    ofname = os.path.splitext(PLAIN_TEXT_FNAME)[0] + "_" + str(round(100 * current_skip)) + ".txt"
    ofname_csv = os.path.splitext(ofname)[0] + ".csv"
    ofname_csv_skipped = os.path.splitext(ofname)[0] + "_skipped.csv"

    #ifname, ifname_csv =
    nm, nmcsv = rr.process( ifname=ifname,
                                ifname_csv=ifname_csv,
                                ofname=ofname,
                                ofname_csv=ofname_csv,
                                ofname_csv_skipped=ofname_csv_skipped,
                                skip_rate=current_skip
                                )
    print("Done in", time.clock() - ts, nm)


