import numpy as np
import random
from gensim.models import Word2Vec

model = Word2Vec.load_word2vec_format(
            'news.model.bin.gz'
            , binary=True
            , encoding='utf-8'
        )

def get_char_id(char):
    if ord('а') <= ord(char) <= ord('я'):
        return ord(char) - ord('а')
    if char == '-':
        return 33
    if char == '^':
        return 34
    if char == '$':
        return 35
    #raise Exception("Unsupported character", char)

def word_to_ohe_vector(word):
    result = np.zeros(len(word) * letters_count)
    result = result.reshape(len(word), letters_count)
    for i in range(len(word)):
        result[i, get_char_id(word[i])] = 1
    return result

def delete_tag(word):
    return word[ : len(word) - word[::-1].find('_') - 1 ]

def is_suitable_word(word):
    return len(word) <= max_word_len and not None in [get_char_id(char) for char in word]

max_word_len = 15
letters_count = 34

def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]

def most_similar_neighbor(word, topn, w2v_model):
    top = w2v_model.most_similar(positive=[word], topn=topn)
    
    len_array = [len(longest_common_substring(delete_tag(word), delete_tag(neighbor))) for neighbor, _ in top]
    
    i = np.argmax(len_array)
    
    if len_array[i] > 2 and is_suitable_word(delete_tag(top[i][0])):
        return top[i][0]

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, Embedding, \
                         Convolution1D, GlobalMaxPooling1D, Lambda, Permute, merge
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.optimizers import Adam, RMSprop

import tensorflow as tf
tf.python.control_flow_ops = tf

input_1 = Input(shape=(max_word_len, letters_count, ))
input_2 = Input(shape=(max_word_len, letters_count, ))

#struct_shared = Convolution1D(nb_filter=1000, filter_length=4, border_mode='valid', activation='relu')
struct_shared = Bidirectional(LSTM(100))

struct_1 = struct_shared(input_1)
struct_2 = struct_shared(input_2)

#struct_1 = GlobalMaxPooling1D()(struct_1)
#struct_2 = GlobalMaxPooling1D()(struct_2)

struct_merged = merge([struct_1, struct_2], mode='sum', concat_axis=-1)

dropout_1 = Dropout(0.1)(struct_merged)

dense_1 = Dense(200, activation='sigmoid')(dropout_1)

dropout_2 = Dropout(0.1)(dense_1)

dense_2 = Dense(1, activation='sigmoid')(dropout_2)

siamese_model = Model(input=[input_1, input_2], output=dense_2)
siamese_model.summary()
siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['binary_accuracy'])
#siamese_model.fit([X_1, X_2], Y, batch_size=20, nb_epoch=1, validation_split=0)

from keras.preprocessing import sequence

def myGenerator(w2v_model, batch_size=20):
    word_array = []
    vocab_ohe = {}
    for word in w2v_model.vocab:
        word_no_tag = delete_tag(word)
        if is_suitable_word(word_no_tag):
            word_array.append(word_no_tag)
            vocab_ohe[word_no_tag] = word_to_ohe_vector(word_no_tag)
    
    vocab = list(w2v_model.vocab)
     
    while 1:
        X_1_batch = []
        X_2_batch = []
        Y_batch = []
        
        while len(Y_batch) < batch_size:
            word = random.choice(vocab)
            
            word_no_tag = delete_tag(word)
            if not is_suitable_word(delete_tag(word_no_tag)):
                continue
            word_ohe = word_to_ohe_vector(word_no_tag)
            
            msn = most_similar_neighbor(word, 20, w2v_model)
            if msn is None:
                continue
            
            X_1_batch.append(word_ohe)
            X_2_batch.append(vocab_ohe[delete_tag(msn)])
            Y_batch.append(1)

            X_1_batch.append(word_ohe)
            X_2_batch.append(vocab_ohe[random.choice(word_array)])
            Y_batch.append(0)
                
        Y_batch = np.array(Y_batch)
        X_1_batch = sequence.pad_sequences(X_1_batch, maxlen=max_word_len)
        X_2_batch = sequence.pad_sequences(X_2_batch, maxlen=max_word_len)
                
        yield [X_1_batch, X_2_batch], Y_batch
            
siamese_model.fit_generator(myGenerator(model), samples_per_epoch=60000, nb_epoch=5)
