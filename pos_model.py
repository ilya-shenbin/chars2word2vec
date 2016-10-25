import numpy as np
import pickle

with open('pos_dataset', 'rb') as f:
    dictionary = pickle.load(f)

max_word_len = 33
letters_count = 34
tags_count = 14


def get_char_id(char):
    if ord('а') <= ord(char) <= ord('я'):
        return ord(char) - ord('а')
    if char == '-':
        return 33


def word_to_ohe_vector(word):
    result = np.zeros(len(word) * letters_count)
    result = result.reshape(len(word), letters_count)
    for i in range(len(word)):
        result[i, get_char_id(word[i])] = 1

    return result


def tag_id_to_ohe_vector(tag_id):
    result = np.zeros(tags_count)
    result[tag_id] = 1
    return result


X = np.array([word_to_ohe_vector(word) for word in dictionary.keys()])
Y = np.array([tag_id_to_ohe_vector(word) for word in dictionary.values()])
from keras.preprocessing import sequence
X = sequence.pad_sequences(X, maxlen=max_word_len)


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

rnn_model = Sequential()
rnn_model.add(LSTM(output_dim=100, input_dim=letters_count))
rnn_model.add(Dense(tags_count))
rnn_model.add(Activation('softmax'))

rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

rnn_model.fit(X, Y, batch_size=20, nb_epoch=2, validation_split=0.2)
# Train on 293635 samples, validate on 73409 samples
# Epoch 1/2
# 293635/293635 [==============================] - 569s - loss: 0.3704 - acc: 0.8679 - val_loss: 0.2605 - val_acc: 0.9054
# Epoch 2/2
# 293635/293635 [==============================] - 600s - loss: 0.2220 - acc: 0.9250 - val_loss: 0.2068 - val_acc: 0.9306


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Convolution1D, GlobalMaxPooling1D

cnn_model = Sequential()

cnn_model.add(Convolution1D(input_dim=letters_count,
                            input_length=max_word_len,
                            nb_filter=200,
                            filter_length=3,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
cnn_model.add(GlobalMaxPooling1D())

cnn_model.add(Dense(100))
cnn_model.add(Activation('relu'))

cnn_model.add(Dense(tags_count))
cnn_model.add(Activation('softmax'))

cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn_model.fit(X, Y, batch_size=20, nb_epoch=2, validation_split=0.2)
# Train on 293635 samples, validate on 73409 samples
# Epoch 1/2
# 293635/293635 [==============================] - 58s - loss: 0.2901 - acc: 0.8993 - val_loss: 0.2224 - val_acc: 0.9245
# Epoch 2/2
# 293635/293635 [==============================] - 59s - loss: 0.2088 - acc: 0.9282 - val_loss: 0.2011 - val_acc: 0.9309