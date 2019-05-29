"""
Adapted from https://github.com/Jianbo-Lab/L2X

The code for constructing the Hierarchical LSTM is based on that of Richard
Liao at https://github.com/richliao/textClassifier

"""

import argparse
import pickle as pkl
import os
import time

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Multiply, Lambda, Activation, Conv1D
from keras.layers import GlobalMaxPooling1D, Dropout, LSTM, Bidirectional
from keras.layers import TimeDistributed, Embedding
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from subsets.L2X.imdb_sent.make_data import load_data, create_dataset_from_score
from subsets.sample_subsets import sample_subset

# Set parameters
tf.set_random_seed(10086)
np.random.seed(10086)
MAX_SENT_LENGTH = 100
MAX_SENTS = 15
EMBEDDING_DIM = 100
MAX_NUM_WORDS = 20000
BATCHSIZE = 100
k = 1  # Number of selected sentences by L2X.

# ##################################
# #########Original Model###########
# ##################################


def create_original_model(word_index):
    """
    Build the original model to be explained.

    """
    with tf.variable_scope('prediction_model'):
        embedding_layer = Embedding(MAX_NUM_WORDS + 1,
                                    EMBEDDING_DIM,
                                    input_length=MAX_SENT_LENGTH,
                                    name='embedding',
                                    trainable=True)

        sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(LSTM(100, name='lstm'),
                               name='bidirectional')(embedded_sequences)
        sentEncoder = Model(sentence_input, l_lstm)

        review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(LSTM(100, name='lstm2'),
                                    name='bidirectional2')(review_encoder)
        preds = Dense(2, activation='softmax', name='dense')(l_lstm_sent)
        model = Model(review_input, preds)

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        return model


def generate_original_preds(train=True):
    """
    Generate the predictions of the original model on training
    and validation datasets.

    The original model is also trained if train = True.

    """
    print('Loading data...')
    dataset = load_data()
    word_index = dataset['word_index']
    x_train, x_val, y_train, y_val = (
        dataset['x_train'], dataset['x_val'],
        dataset['y_train'], dataset['y_val'])

    print('Creating model...')
    model = create_original_model(word_index)

    if train:
        if 'models' not in os.listdir('.'):
            os.mkdir('models')

        filepath = "models/original.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='val_acc', verbose=1,
            save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  callbacks=callbacks_list, epochs=5, batch_size=BATCHSIZE)

    weights_name = 'original.hdf5'
    model.load_weights('./models/' + weights_name, by_name=True)

    pred_train = model.predict(x_train, verbose=1, batch_size=1000)
    pred_val = model.predict(x_val, verbose=1, batch_size=1000)

    np.save('data/pred_train.npy', pred_train)
    np.save('data/pred_val.npy', pred_val)

# ##################################
# #########     L2X      ###########
# ##################################


# Define various Keras layers.
Mean = Lambda(lambda x: K.sum(x, axis=1) / float(k),
              output_shape=lambda x: [x[0], x[2]])


class Concatenate(Layer):
    """
    Layer for concatenation.

    """
    def __init__(self, **kwargs):
        super(Concatenate, self).__init__(**kwargs)

    def call(self, inputs):
        input1, input2 = inputs
        input1 = tf.expand_dims(input1, axis=-2)
        # [batchsize, 1, input1_dim]
        dim1 = int(input2.get_shape()[1])
        input1 = tf.tile(input1, [1, dim1, 1])
        return tf.concat([input1, input2], axis=-1)

    def compute_output_shape(self, input_shapes):
        input_shape1, input_shape2 = input_shapes
        input_shape = list(input_shape2)
        input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
        input_shape[-2] = int(input_shape[-2])
        return tuple(input_shape)


class SampleSubset(Layer):
    """
    Layer for continuous approx of subset sampling

    """
    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(SampleSubset, self).__init__(**kwargs)

    def call(self, logits):
        # logits: [BATCH_SIZE, d, 1]
        logits = tf.squeeze(logits, 2)
        samples = sample_subset(logits, self.k, self.tau0)

        # Explanation Stage output.
        threshold = tf.expand_dims(
            tf.nn.top_k(logits, self.k, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(
            tf.greater_equal(logits, threshold), tf.float32)
        output = K.in_train_phase(samples, discrete_logits)
        return tf.expand_dims(output, -1)

    def compute_output_shape(self, input_shape):
        return input_shape


class SampleConcrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables.

    """
    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(SampleConcrete, self).__init__(**kwargs)

    def call(self, logits):
        logits_ = K.permute_dimensions(logits, (0, 2, 1))
        # [batchsize, 1, MAX_SENTS]
        uniform = [tf.random_uniform(tf.shape(logits),
                   minval=np.finfo(tf.float32.as_numpy_dtype).tiny,
                   maxval=1.0) for i in range(self.k)]
        uniform = K.permute_dimensions(tf.concat(uniform, axis=2), (0, 2, 1))

        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis=1)
        samples = K.expand_dims(samples, -1)

        discrete_logits = K.one_hot(K.argmax(logits_, axis=-1), num_classes=MAX_SENTS)
        discrete_logits = K.permute_dimensions(discrete_logits, (0, 2, 1))
        return K.in_train_phase(samples, discrete_logits)

    def compute_output_shape(self, input_shape):
        return input_shape


def construct_gumbel_selector(review_input, max_sent_length, embedding_dim, max_sents, word_index):
    """
    Build the L2X model for selecting sentences.

    """
    sentence_input = Input(shape=(max_sent_length,), dtype='int32')
    embedding_layer = Embedding(MAX_NUM_WORDS + 1,
                                embedding_dim,
                                input_length=max_sent_length,
                                name='embedding',
                                trainable=True)

    embedded_sequences = embedding_layer(sentence_input)
    net = Dropout(0.2)(embedded_sequences)
    net = Conv1D(250,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1)(net)
    net = GlobalMaxPooling1D()(net)
    sentEncoder = Model(sentence_input, net)

    # [batch_size, max_sents, 100]
    review_encoder = TimeDistributed(sentEncoder)(review_input)

    net = review_encoder
    first_layer = Conv1D(
        100, 3, padding='same', activation='relu', strides=1, name='conv1_gumbel')(net)

    # global info
    net_new = GlobalMaxPooling1D(name='new_global_max_pooling1d_1')(first_layer)

    global_info = Dense(100, name='new_dense_1', activation='relu')(net_new)

    # local info
    net = Conv1D(50, 3, padding='same', activation='relu', strides=1, name='conv2_gumbel')(first_layer)
    local_info = Conv1D(50, 3, padding='same', activation='relu', strides=1, name='conv3_gumbel')(net)
    combined = Concatenate()([global_info, local_info])
    net = Dropout(0.2, name='new_dropout_2')(combined)
    net = Conv1D(50, 1, padding='same', activation='relu', strides=1, name='conv_last_gumbel')(net)

    logits_T = Conv1D(1, 1, padding='same', activation=None, strides=1, name='conv4_gumbel')(net)

    return logits_T


def L2X(train=True, task='l2x', tau=0.01):
    """
    Generate scores on features on validation by L2X.

    Train the L2X model with variational approaches
    if train = True.

    """
    print('Loading dataset...')
    dataset = load_data()
    word_index = dataset['word_index']
    x_train, x_val, y_train, y_val = dataset['x_train'], dataset['x_val'], dataset['y_train'], dataset['y_val']
    with open('./data/word_index.pkl', 'rb') as f:
        word_index = pkl.load(f)

    print('Creating model...')

    # P(S|X)
    with tf.variable_scope('selection_model'):
        review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
        logits_T = construct_gumbel_selector(review_input, MAX_SENT_LENGTH, EMBEDDING_DIM, MAX_SENTS, word_index)
        if task == 'subsets':
            subset_sampler = SampleSubset(tau, k)
            T = subset_sampler(logits_T)
        else:
            subset_sampler = SampleConcrete(tau, k)
            T = subset_sampler(logits_T)

    # q(X_S)
    with tf.variable_scope('prediction_model'):
        sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')

        embedding_layer = Embedding(MAX_NUM_WORDS + 1,
                                    EMBEDDING_DIM,
                                    input_length=MAX_SENT_LENGTH,
                                    name='embedding',
                                    trainable=True)

        embedded_sequences = embedding_layer(sentence_input)
        net = Dropout(0.2)(embedded_sequences)
        net = Conv1D(250,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1)(net)
        net = GlobalMaxPooling1D()(net)
        sentEncoder2 = Model(sentence_input, net)

        review_encoder2 = TimeDistributed(sentEncoder2)(review_input)
        selected_encoding = Multiply()([review_encoder2, T])
        net = Mean(selected_encoding)
        net = Dense(250)(net)
        net = Activation('relu')(net)
        preds = Dense(2, activation='softmax', name='new_dense')(net)

    model = Model(inputs=review_input, outputs=preds)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    pred_train = np.load('data/pred_train.npy')
    pred_val = np.load('data/pred_val.npy')

    val_acc = np.mean(np.argmax(pred_val, axis=1) == np.argmax(y_val, axis=1))
    print('The validation accuracy of the original model is {}'.format(val_acc))

    if train:
        # split x_train and pred_train into training and validation
        x_train, x_train_val, pred_train, pred_train_val = train_test_split(
            x_train, pred_train, test_size=0.1, random_state=111)

        filepath = f"models/{task}-{tau}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='val_loss',
            verbose=1, save_best_only=True, mode='auto')

        callbacks_list = [checkpoint]

        model.fit(
            x_train, pred_train,
            validation_data=(x_train_val, pred_train_val),
            callbacks=callbacks_list,
            epochs=50, batch_size=BATCHSIZE)

    weights_name = f'{task}-{tau}.hdf5'
    model.load_weights('models/{}'.format(weights_name), by_name=True)

    pred_model = Model(review_input, [T, logits_T, preds])

    pred_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    st = time.time()
    selections, scores, interp_val = pred_model.predict(x_val,
                                                        verbose=1,
                                                        batch_size=BATCHSIZE)

    print('Time spent is {}'.format(time.time() - st))
    return scores, x_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['original', 'l2x', 'subsets'], default='original')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--tau', type=float, default=1.0)
    parser.set_defaults(train=False)

    args = parser.parse_args()
    print(args)

    if args.task == 'original':
        generate_original_preds(args.train)
    else:
        scores, x_val = L2X(args.train, task=args.task, tau=args.tau)
        print('Creating dataset with selected sentences...')
        create_dataset_from_score(scores, x_val, args.task, args.tau)
