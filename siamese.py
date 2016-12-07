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

max_word_len = 12
letters_count = 34



from keras.preprocessing import sequence

def myGenerator(w2v_model, batch_size=20):
    word_array = []
    vocab_ohe = {}
    for word in model.vocab:
        word_no_tag = delete_tag(word)
        if is_suitable_word(word_no_tag):
            word_array.append(word_no_tag)
            vocab_ohe[word_no_tag] = word_to_ohe_vector(word_no_tag)
    
    vocab = list(w2v_model.vocab)
     
    while 1:
        X_1_batch = []
        X_2_batch = []
        Y_batch = []
        
        samples = 5

        for _ in range(batch_size):
            word = random.choice(vocab)
            
            word_no_tag = delete_tag(word)
            if not is_suitable_word(delete_tag(word_no_tag)):
                continue
            word_ohe = word_to_ohe_vector(word_no_tag)

            for neighbor, _ in model.most_similar(positive=[word], topn=samples):
                if not is_suitable_word(delete_tag(neighbor)):
                    continue
                X_1_batch.append(word_ohe)
                X_2_batch.append(vocab_ohe[delete_tag(neighbor)])
                Y_batch.append(1)

            for _ in range(samples):
                X_1_batch.append(word_ohe)
                X_2_batch.append(vocab_ohe[random.choice(word_array)])
                Y_batch.append(0)
                
        Y_batch = np.array(Y_batch)
        X_1_batch = sequence.pad_sequences(X_1_batch, maxlen=max_word_len)
        X_2_batch = sequence.pad_sequences(X_2_batch, maxlen=max_word_len)
                
        yield [X_1_batch, X_2_batch], Y_batch



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

cnn_shared = Convolution1D(nb_filter=500, filter_length=3, border_mode='valid', activation='relu')

cnn_1 = cnn_shared(input_1)
cnn_2 = cnn_shared(input_2)

cnn_1 = GlobalMaxPooling1D()(cnn_1)
cnn_2 = GlobalMaxPooling1D()(cnn_2)

cnn_merged = merge([cnn_1, cnn_2], mode='sum', concat_axis=-1)

dropout_1 = Dropout(0.3)(cnn_merged)

dense_1 = Dense(50, activation='sigmoid')(dropout_1)
dense_2 = Dense(1, activation='sigmoid')(dense_1)

siamese_model = Model(input=[input_1, input_2], output=dense_2)
siamese_model.summary()
siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['binary_accuracy'])       
siamese_model.fit_generator(myGenerator(model), samples_per_epoch=60000, nb_epoch=2)
