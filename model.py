import numpy as np
from gensim.models import Word2Vec

import tensorflow as tf
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Embedding, Activation, BatchNormalization, \
                         Convolution1D, GlobalMaxPooling1D, SpatialDropout1D, \
                         Convolution2D, LocallyConnected2D, LocallyConnected1D,\
                         Lambda, Permute, Reshape, RepeatVector, merge
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.regularizers import l1, l2
from keras.constraints import maxnorm, nonneg, unitnorm
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import sequence
from keras.models import load_model
from keras import metrics


model = Word2Vec.load_word2vec_format('PubMed-w2v.bin', binary=True)

evaluate_step = 2
max_word_len = 15
letters_count = 30
w2v_dim = 200
test_size = 100000
vocab_size = len(model.vocab)

def get_char_id(char):
    if ord('a') <= ord(char) <= ord('z'):
        return ord(char) - ord('a')
    if ord('A') <= ord(char) <= ord('Z'):
        return ord(char) - ord('A')
    if char == '-':
        return 26
    if ord('0') <= ord(char) <= ord('9'):
        return 27
    if char == '(' or char == ')' or char == '[' or char == ']':
        return 28
    if char == '.' or char == ',':
        return 29


def word_to_ohe_vector(word):
#     word = '^' + word + '$'
    result = np.zeros(len(word) * letters_count)
    result = result.reshape(len(word), letters_count)
    for i in range(len(word)):
        result[i, get_char_id(word[i])] = 1
    return result

def get_mask(word):
    return np.array([1 if i >= max_word_len - len(word) else 0 for i in range(max_word_len)])


word_array = []
vec_array = []
mask_array = []

for word in model.vocab:
    if len(word) <= max_word_len and not None in [get_char_id(char) for char in word]:
        word_array.append(word)
        vec_array.append(model[word][:w2v_dim])
        mask_array.append(get_mask(word))

X = np.array([word_to_ohe_vector(word) for word in word_array])
X = sequence.pad_sequences(X, maxlen=max_word_len)
Y = np.array(vec_array)
X_mask = np.array(mask_array)

words_train, words_test = word_array[test_size:], word_array[:test_size]
X_train, X_test = X[test_size:], X[:test_size]
X_mask_train, X_mask_test = X_mask[test_size:], X_mask[:test_size]
Y_train, Y_test = Y[test_size:], Y[:test_size]

def mse_plus_cos(y_true, y_pred):
    return 10 * metrics.cosine_proximity(y_true, y_pred) + metrics.mean_squared_error(y_true, y_pred)


inputs = Input(shape=(max_word_len, letters_count, ))
mask = Input(shape=(max_word_len, ))

conv1d = Bidirectional(LSTM(output_dim=1000, return_sequences=True, dropout_W=0.1, dropout_U=0.1))(inputs)
conv1d = Bidirectional(LSTM(output_dim=500, return_sequences=True, dropout_W=0.1, dropout_U=0.1))(conv1d)
conv_output_dim = 1000

conv1d = Dropout(0.02)(conv1d)
conv1d = merge([conv1d, mask], mode=lambda x: tf.einsum('dij,di->dij', x[0], x[1]), output_shape=(max_word_len, conv_output_dim))

conv1d = merge([conv1d, mask], mode=lambda x: tf.einsum('dij,di->dij', x[0], x[1]), output_shape=(max_word_len, conv_output_dim))

attent = TimeDistributed(Dense(50, activation='relu'))(conv1d)
attent = TimeDistributed(Dense(1, activation='linear'))(attent)
attent = Activation('sigmoid')(attent)

attention_dot = merge([attent, conv1d], mode='dot', dot_axes=(1,1))
attention_dot = Lambda(lambda x: x[:,0,:])(attention_dot)

maxpool_1 = GlobalMaxPooling1D()(conv1d)

maxpool_and_att = merge([maxpool_1, attention_dot], mode='concat')

dense_2 = Dense(w2v_dim)(maxpool_and_att) # maxpool_1 / attention_dot / maxpool_and_att

gcnn_model = Model(input=[inputs, mask], output=dense_2)
gcnn_model.summary()
gcnn_model.compile(loss=mse_plus_cos, optimizer='adam', metrics=['cosine_proximity', 'mse'])


c2w2v_model = gcnn_model

model_name_str = 'gcnn_model'
start_iter = 1

if start_iter > 1:
    c2w2v_model.load_weights(model_name_str + '_' + str(start_iter-1) + '.h5')

def evaluate():
    test_size = 1000
    Y_pred = c2w2v_model.predict([X_test[:test_size], X_mask_test[:test_size]], batch_size=100)

    topw = 0.
    topl = 0.
    top = 0.
    for i in range(test_size):
        if i % 1000 == 0:
            print(i)
        most_similar = model.most_similar(positive=[Y_pred[i]], topn=100)
        for rank, (neigh_word, _) in enumerate(most_similar):
            if neigh_word == words_test[i]:
                topw += 1. / (rank+1)
                topl += 1. / np.log(2.71 + rank)
                top += 1.
                break
                
    return topw, topl, top

for ep in range(start_iter, 50 + 1):
    print("--- ### EPOCH {} ### ---".format(ep))
    c2w2v_model.fit([X_train, X_mask_train], Y_train, batch_size=50, nb_epoch=1,
                    validation_data=([X_test, X_mask_test], Y_test)
                   )
    c2w2v_model.save(model_name_str + '_' + str(ep) + '.h5')
    if ep % evaluate_step == evaluate_step - 1:
        print(evaluate())
