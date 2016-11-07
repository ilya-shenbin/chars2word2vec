#
# removes punctuation, digits etc from each file
# in directory and saves the result into a single
# text file.
#

import re
import nltk
import csv
import os


RAW_DATA_PATH = "data_raw"
PROCESSED_DATA_PATH = "data_processed"
OFNAME = os.path.join(PROCESSED_DATA_PATH, "data_processed.txt")
CSV_NAME = os.path.splitext(OFNAME)[0] + ".csv"

freqs = nltk.FreqDist()

with open(OFNAME, 'w') as output_file:
    for filename in os.listdir(RAW_DATA_PATH):
        with open(os.path.join(RAW_DATA_PATH, filename), 'r') as f:
            text = f.read()
            tokens = [e.lower() for e in map(str.strip,
                                             re.split("([^\u0400-\u0500]+)",
                                                      text))
                        if len(e) > 0 and not re.match("[^\u0400-\u0500]", e)]

            freqs = freqs +  nltk.FreqDist(tokens)

            out_str = ''.join(["%s " % token for token in tokens])
            output_file.write(out_str)

#            for token in tokens:
#                output_file.write("%s " % token)

            output_file.write('\n')

with open(CSV_NAME, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(freqs.items())
