import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from keras.preprocessing import sequence

w2v_model = KeyedVectors.load('ok/ok-20161206.w2v.300.bin')

evaluate_step = 5
max_word_len = 12
letters_count = 34 + 1
w2v_dim = 300
test_size = 10000
vocab_size = len(w2v_model.vocab)


def get_char_id(char):
    if ord('а') <= ord(char) <= ord('я'):
        return ord(char) - ord('а')
    if char == '-':
        return 33
    if char == '$':
        return 34


def word_to_ohe_vector(word, add_end_token=False):
    # word = '^' + word + '$'
    if add_end_token:
        word = word + '$'
    result = np.zeros(len(word) * letters_count)
    result = result.reshape(len(word), letters_count)
    for i in range(len(word)):
        result[i, get_char_id(word[i])] = 1
    return result

def word_to_num_vector(word, add_end_token=False):
    # word = '^' + word + '$'
    if add_end_token:
        word = word + '$'
    result = np.zeros(len(word))
    for i in range(len(word)):
        result[i] = get_char_id(word[i])
    return result

def get_encoder_mask(word):
    return np.array([1 if i >= max_word_len - len(word) else 0 for i in range(max_word_len)])

def get_decoder_mask(word):
    return np.array([1 if i < len(word) else 0 for i in range(max_word_len)])


word_array = []
vec_array = []
mask_array = []
length_array = []

for word in w2v_model.vocab:
    if len(word) <= max_word_len and not None in [get_char_id(char) for char in word]:
        word_array.append(word)
        vec_array.append(w2v_model[word][:w2v_dim])
        mask_array.append(get_decoder_mask(word))
        length_array.append(len(word))


Xe = sequence.pad_sequences(
    np.array([word_to_ohe_vector(word) for word in word_array]),
    maxlen=max_word_len,
    padding='pre')

Xd = sequence.pad_sequences(
    np.array([word_to_num_vector(word) for word in word_array]),
    maxlen=max_word_len,
    padding='post',
    value=get_char_id('$'))

Y = np.array(vec_array)
X_mask = np.array(mask_array)
X_length = np.array(length_array, dtype='int32')

def train_test_split(array, test_size):
    return array[test_size:], array[:test_size]

words_train, words_test = train_test_split(word_array, test_size)
Xe_train, Xe_test = train_test_split(Xe, test_size)
Xd_train, Xd_test = train_test_split(Xd, test_size)
X_mask_train, X_mask_test = train_test_split(X_mask, test_size)
X_length_train, X_length_test = train_test_split(X_length, test_size)
Y_train, Y_test = train_test_split(Y, test_size)


def mse_plus_cos(y_true, y_pred):
    return 10 * metrics.cosine_proximity(y_true, y_pred) + metrics.mean_squared_error(y_true, y_pred)


def evaluate(Y_pred):
    topw = 0.
    topl = 0.
    top = 0.
    for i in range(test_size):
        if i % 1000 == 0:
            #print(i)
            pass
        most_similar = w2v_model.most_similar(positive=[Y_pred[i]], topn=100)
        for rank, (neigh_word, _) in enumerate(most_similar):
            if neigh_word == words_test[i]:
                topw += 1. / (rank+1)
                topl += 1. / np.log(2.71 + rank)
                top += 1.
                break
                
    return topw, topl, top


import functools
import sets
import tensorflow as tf
import datetime


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceClassification:

    def __init__(self, data, length, target, decoder_output, dropout, num_hidden=1000, num_layers=1):
        self.data = data
        self.length = length
        self.target = target
        self.decoder_output = decoder_output
        self.dropout = dropout
        self.batch = batch
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.encoder
        self.w2v_predictor
        self.decoder
        self.decoder_optimize
        self.w2v_optimize
        self.w2v_error

    @lazy_property
    def encoder(self):
        # Recurrent network.
        cell_fw = tf.contrib.rnn.GRUCell(self._num_hidden)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout)
        cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw] * self._num_layers)
        
        # cell_bw = tf.contrib.rnn.GRUCell(self._num_hidden)
        # cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout)
        # cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw] * self._num_layers)
        
        # (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.data, sequence_length=self.length, dtype=tf.float32)
        # output = tf.concat((output_fw, output_bw), axis=2)
        
        output, _ = tf.nn.dynamic_rnn(cell_fw, self.data, sequence_length=self.length, dtype=tf.float32)
        
        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        
        return last

    @lazy_property
    def w2v_predictor(self):
        weight, bias = self._weight_and_bias(self._num_hidden, int(self.target.get_shape()[1]))
        w2v = tf.matmul(self.encoder, weight) + bias
        return w2v
    
    @lazy_property
    def decoder(self):
        cell = tf.contrib.rnn.GRUCell(310)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
        
        bottleneck = tf.layers.dense(self.encoder, 10)
        bottleneck_and_w2v = tf.concat((bottleneck, self.target), axis=1)
        
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=tf.zeros(shape=(self.batch, max_word_len, 310)),
            sequence_length=self.length,
            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=training_helper,
            initial_state=bottleneck_and_w2v,
            output_layer=Dense(letters_count)) 

        train_dec_outputs, _ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=max_word_len)
        
        return train_dec_outputs

    @lazy_property
    def w2v_cost(self):
        # cross_entropy = -tf.reduce_sum(self.target * tf.log(self.encoder))
        # return cross_entropy
        y_true = tf.nn.l2_normalize(self.target, dim=-1)
        y_pred = tf.nn.l2_normalize(self.w2v_predictor, dim=-1)
        return -tf.reduce_mean(y_true * y_pred, axis=-1)

    @lazy_property
    def decoder_cost(self):
        
        print(tf.identity(self.decoder.rnn_output, name='logits').get_shape())        
        print(self.decoder_output.get_shape())
        print(tf.zeros_like(self.decoder_output, dtype='float32').get_shape())
        
        loss = tf.contrib.seq2seq.sequence_loss(
                        tf.identity(self.decoder.rnn_output, name='logits'),
                        self.decoder_output,
                        tf.ones_like(self.decoder_output, dtype='float32'))
        return loss
    
    @lazy_property
    def decoder_optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.decoder_cost)
    
    @lazy_property
    def w2v_optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.w2v_cost)
    
    @lazy_property
    def w2v_error(self):
        # mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.encoder, 1))
        # return tf.reduce_mean(tf.cast(mistakes, tf.float32))
        y_true = tf.nn.l2_normalize(self.target, dim=-1)
        y_pred = tf.nn.l2_normalize(self.w2v_predictor, dim=-1)
        return -tf.reduce_mean(y_true * y_pred)

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)



tf.reset_default_graph()
batch_size = 100
train_size, rows, row_size = Xe_train.shape
num_classes = Y_train.shape[1]
epoch_count = 100
subepoch_count = 1

data = tf.placeholder(tf.float32, [None, rows, row_size])
length = tf.placeholder(tf.int32, [None])
target = tf.placeholder(tf.float32, [None, num_classes])
decoder_output = tf.placeholder(tf.int32, [None, rows])
dropout = tf.placeholder(tf.float32)
batch = tf.placeholder(tf.int32)

model = SequenceClassification(data, length, target, decoder_output, dropout)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(subepoch_count * epoch_count):
    dt_start = datetime.datetime.utcnow()

    for _ in range(93487 // batch_size // subepoch_count):
        batch_idx = np.random.choice(train_size, batch_size)
#         sess.run(model.w2v_optimize, {data: Xe_train[batch_idx],
#                                       #length: X_length_train[batch_idx],
#                                       length: np.ones(batch_size) * max_word_len,
#                                       target: Y_train[batch_idx],
#                                       dropout: 0.9,
#                                       batch: batch_size})
        
        sess.run(model.decoder_optimize, {data: Xe_train[batch_idx],
                                          #length: X_length_train[batch_idx],
                                          length: np.ones(batch_size) * max_word_len,
                                          target: Y_train[batch_idx],
                                          decoder_output: Xd_train[batch_idx],
                                          dropout: 0.9,
                                          batch: batch_size})

    dt_end = datetime.datetime.utcnow()

    decoder_cost = sess.run(model.decoder_cost, {data: Xe_test,
                                                 length: np.ones(test_size) * max_word_len,
                                                 target: Y_test,
                                                 decoder_output: Xd_test,
                                                 dropout: 1,
                                                 batch: test_size})
    
    w2v_cost = sess.run(model.w2v_error, {data: Xe_test,
                                          length: np.ones(test_size) * max_word_len,
                                          target: Y_test,
                                          dropout: 1,
                                          batch: test_size})
    
    print('Epoch {:2d}; w2v_loss {:1.5f}; rae_loss {:1.5f}; time {:3.0f} sec'.format(
            epoch + 1, w2v_cost, decoder_cost, (dt_end-dt_start).total_seconds()
        ))

    if epoch % evaluate_step == evaluate_step - 1:
        Y_pred = sess.run(model.w2v_predictor, {data: Xe_test,
                                                length: np.ones(test_size) * max_word_len,
                                                target: Y_test,
                                                dropout: 1,
                                                batch: test_size})
        #print(evaluate(Y_pred))