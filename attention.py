import numpy as np
from gensim.models import Word2Vec

model = Word2Vec.load_word2vec_format(
            'ruscorpora.model.bin.gz'
            , binary=True
            , encoding='utf-8'
        )

max_word_len = 18
letters_count = 34
w2v_dim = 300
vocab_size = len(model.vocab)

def get_char_id(char):
    if ord('а') <= ord(char) <= ord('я'):
        return ord(char) - ord('а')
    if char == '-':
        return 33
    #raise Exception("Unsupported character", char)

def word_to_ohe_vector(word):
    result = np.zeros(len(word) * letters_count)
    result = result.reshape(len(word), letters_count)
    for i in range(len(word)):
        result[i, get_char_id(word[i])] = 1
    return result

def delete_tag(word):
    return word[ : len(word) - word[::-1].find('_') - 1 ]

word_array = []
vec_array = []

for word in model.vocab:
    word_no_tag = delete_tag(word)
    if len(word_no_tag) <= max_word_len and not None in [get_char_id(char) for char in word_no_tag]:
        word_array.append(word_no_tag)
        vec_array.append(model[word])

X = np.array([word_to_ohe_vector(word) for word in word_array])
Y = np.array(vec_array)

from keras.preprocessing import sequence
X = sequence.pad_sequences(X, maxlen=max_word_len)



from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, Embedding, \
                         Convolution1D, GlobalMaxPooling1D, Lambda, Permute, merge
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed, Bidirectional

import tensorflow as tf
tf.python.control_flow_ops = tf

inputs = Input(shape=(max_word_len, letters_count, ))
lstm_fwd = LSTM(100, return_sequences=True)(inputs)
lstm_bwd = LSTM(100, return_sequences=True, go_backwards=True)(inputs)
blstm = merge([lstm_fwd, lstm_bwd], mode='concat')

attention_dense = TimeDistributed(Dense(50, activation='tanh'))(blstm)
attention_out = TimeDistributed(Dense(1, activation='linear'))(attention_dense)
attention_softmax = Activation('softmax')(attention_out)

attention_dot = merge([attention_softmax, blstm], mode='dot', dot_axes=(1,1))

lambda_1 = Lambda(lambda x: x[:,0,:])(attention_dot)

dense_1 = Dense(50, activation='sigmoid')(lambda_1)
dense_2 = Dense(w2v_dim)(dense_1)

lstm_att_model = Model(input=inputs, output=dense_2)
lstm_att_model.summary()
lstm_att_model.compile(loss='cosine_proximity', optimizer='adam', metrics=['cosine_proximity', 'mse'])
lstm_att_model.fit(X, Y, batch_size=20, nb_epoch=1, validation_split=0)



word_to_pedict = [word_to_ohe_vector('апельсин')]
b = lstm_att_model.predict(sequence.pad_sequences(word_to_pedict, maxlen=max_word_len))
model.most_similar(positive=b, topn=20)