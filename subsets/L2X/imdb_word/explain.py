"""
Adapted from https://github.com/Jianbo-Lab/L2X

The code for constructing the original word-CNN is based on
https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py
"""
import time
import pickle
import os

from keras.layers import Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Embedding, Dense, Dropout, Activation
from keras.datasets import imdb
from keras.engine.topology import Layer
from keras import backend as K
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.models import Model, Sequential

import numpy as np
import tensorflow as tf
from subsets.L2X.imdb_word.utils import create_dataset_from_score, calculate_acc
from sklearn.model_selection import train_test_split
from subsets.sample_subsets import sample_subset


# Set parameters:
tf.set_random_seed(10086)
np.random.seed(10086)
max_features = 5000
maxlen = 400
batch_size = 40
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
original_epochs = 5
k = 10  # Number of selected words by L2X.
PART_SIZE = 125

###########################################
###############Load data###################
###########################################


def load_data():
    """
    Load data if data have been created.
    Create data otherwise.

    """
    if 'data' not in os.listdir('.'):
        os.mkdir('data')

    if 'id_to_word.pkl' not in os.listdir('data'):
        print('Loading data...')
        (x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_features, index_from=3)
        word_to_id = imdb.get_word_index()
        word_to_id = {k:(v+3) for k,v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        id_to_word = {value:key for key,value in word_to_id.items()}

        print(len(x_train), 'train sequences')
        print(len(x_val), 'test sequences')

        print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
        y_train = np.eye(2)[y_train]
        y_val = np.eye(2)[y_val]

        np.save('./data/x_train.npy', x_train)
        np.save('./data/y_train.npy', y_train)
        np.save('./data/x_val.npy', x_val)
        np.save('./data/y_val.npy', y_val)
        with open('data/id_to_word.pkl','wb') as f:
            pickle.dump(id_to_word, f)

    else:
        x_train, y_train, x_val, y_val = (
            np.load('data/x_train.npy'),
            np.load('data/y_train.npy'),
            np.load('data/x_val.npy'),
            np.load('data/y_val.npy'))
        with open('data/id_to_word.pkl','rb') as f:
            id_to_word = pickle.load(f)

    return x_train, y_train, x_val, y_val, id_to_word

###########################################
###############Original Model##############
###########################################

def create_original_model():
    """
    Build the original model to be explained.

    """
    model = Sequential()
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def generate_original_preds(train=True):
    """
    Generate the predictions of the original model on training
    and validation datasets.

    The original model is also trained if train = True.

    """
    x_train, y_train, x_val, y_val, id_to_word = load_data()
    model = create_original_model()

    if train:
        filepath="models/original.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
            verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list,
            epochs=original_epochs,
            batch_size=batch_size)

    model.load_weights('./models/original.hdf5', by_name=True)

    pred_train = model.predict(x_train,verbose = 1, batch_size = 1000)
    pred_val = model.predict(x_val,verbose = 1, batch_size = 1000)
    if not train:
        print('The val accuracy is {}'.format(calculate_acc(pred_val,y_val)))
        print('The train accuracy is {}'.format(calculate_acc(pred_train,y_train)))

    np.save('data/pred_train.npy', pred_train)
    np.save('data/pred_val.npy', pred_val)

###########################################
####################L2X####################
###########################################
# Define various Keras layers.
Mean = Lambda(lambda x: K.sum(x, axis = 1) / float(k),
    output_shape=lambda x: [x[0],x[2]])


class Concatenate(Layer):
    """
    Layer for concatenation.

    """
    def __init__(self, **kwargs):
        super(Concatenate, self).__init__(**kwargs)

    def call(self, inputs):
        input1, input2 = inputs
        input1 = tf.expand_dims(input1, axis = -2) # [batchsize, 1, input1_dim]
        dim1 = int(input2.get_shape()[1])
        input1 = tf.tile(input1, [1, dim1, 1])
        return tf.concat([input1, input2], axis = -1)

    def compute_output_shape(self, input_shapes):
        input_shape1, input_shape2 = input_shapes
        input_shape = list(input_shape2)
        input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
        input_shape[-2] = int(input_shape[-2])
        return tuple(input_shape)


class SampleConcrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables.

    """
    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(SampleConcrete, self).__init__(**kwargs)

    def call(self, logits):
        # logits: [batch_size, d, 1]
        logits_ = K.permute_dimensions(logits, (0,2,1))# [batch_size, 1, d]

        d = int(logits_.get_shape()[2])
        uniform = [tf.random_uniform(tf.shape(logits),
            minval=np.finfo(tf.float32.as_numpy_dtype).tiny,
            maxval=1.0) for i in range(self.k)]
        uniform = K.permute_dimensions(tf.concat(uniform, axis=2), (0, 2, 1))

        gumbel = - K.log(-K.log(uniform))
        # tau0 = tf.minimum(tf.exp(self.log_tau0), 1)
        noisy_logits = (gumbel + logits_)/ self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis = 1)
        logits = tf.reshape(logits,[-1, d])
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)

        output = K.in_train_phase(samples, discrete_logits)
        return tf.expand_dims(output,-1)

    def compute_output_shape(self, input_shape):
        return input_shape


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
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
        output = K.in_train_phase(samples, discrete_logits)
        return tf.expand_dims(output,-1)

    def compute_output_shape(self, input_shape):
        return input_shape


def construct_gumbel_selector(X_ph, num_words, embedding_dims, maxlen):
    """
    Build the L2X model for selecting words.

    """
    emb_layer = Embedding(num_words, embedding_dims, input_length = maxlen, name = 'emb_gumbel')
    emb = emb_layer(X_ph) #(400, 50)
    net = Dropout(0.2, name = 'dropout_gumbel')(emb)
    net = emb
    first_layer = Conv1D(100, kernel_size, padding='same', activation='relu', strides=1, name = 'conv1_gumbel')(net)

    # global info
    net_new = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1')(first_layer)
    global_info = Dense(100, name = 'new_dense_1', activation='relu')(net_new)

    # local info
    net = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv2_gumbel')(first_layer)
    local_info = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv3_gumbel')(net)
    combined = Concatenate()([global_info,local_info])
    net = Dropout(0.2, name = 'new_dropout_2')(combined)
    net = Conv1D(100, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(net)

    logits_T = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net)

    return logits_T


def L2X(train = True, task='l2x', tau=0.01):
    """
    Generate scores on features on validation by L2X.

    Train the L2X model with variational approaches
    if train = True.

    """
    print('Loading dataset...')
    x_train, y_train, x_val, y_val, id_to_word = load_data()
    pred_train = np.load('data/pred_train.npy')
    pred_val = np.load('data/pred_val.npy')
    print('Creating model...')

    # P(S|X)
    with tf.variable_scope('selection_model'):
        X_ph = Input(shape=(maxlen,), dtype='int32')

        logits_T = construct_gumbel_selector(X_ph, max_features, embedding_dims, maxlen)
        if task == 'subsets':
            subset_sampler = SampleSubset(tau, k)
            T = subset_sampler(logits_T)
        else:
            subset_sampler = SampleConcrete(tau, k)
            T = subset_sampler(logits_T)

    # q(X_S)
    with tf.variable_scope('prediction_model'):
        emb2 = Embedding(max_features, embedding_dims,
            input_length=maxlen)(X_ph)

        net = Mean(Multiply()([emb2, T]))
        net = Dense(hidden_dims)(net)
        net = Activation('relu')(net)
        preds = Dense(2, activation='softmax',
            name = 'new_dense')(net)

    model = Model(inputs=X_ph, outputs=preds)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',#optimizer,
                  metrics=['acc'])
    train_acc = np.mean(np.argmax(pred_train, axis = 1)==np.argmax(y_train, axis = 1))
    val_acc = np.mean(np.argmax(pred_val, axis = 1)==np.argmax(y_val, axis = 1))
    print('The train and validation accuracy of the original model is {} and {}'.format(train_acc, val_acc))

    if train:
        # split x_train and pred_train into training and validation
        x_train, x_train_val, pred_train, pred_train_val = train_test_split(
            x_train, pred_train, test_size=0.1, random_state=111)

        filepath=f"models/{task}-{tau}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
            verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]
        st = time.time()
        model.fit(x_train, pred_train,
            validation_data=(x_train_val, pred_train_val),
            callbacks = callbacks_list,
            epochs=50, batch_size=batch_size)
        duration = time.time() - st
        print('Training time is {}'.format(duration))

    model.load_weights(f'models/{task}-{tau}.hdf5', by_name=True)

    pred_model = Model(X_ph, logits_T)
    pred_model.compile(loss='categorical_crossentropy',
        optimizer='adam', metrics=['acc'])

    st = time.time()
    scores = pred_model.predict(x_val,
        verbose = 1, batch_size = batch_size)[:,:,0]
    scores = np.reshape(scores, [scores.shape[0], maxlen])

    return scores, x_val

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type = str,
        choices = ['original','l2x', 'subsets'], default = 'original')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--tau', type=float, default=0.01)
    parser.set_defaults(train=False)
    args = parser.parse_args()
    print(args)

    if args.task == 'original':
        generate_original_preds(args.train)
    else:
        scores, x = L2X(args.train, task=args.task, tau=args.tau)
        print('Creating dataset with selected sentences...')
        create_dataset_from_score(x, scores, k, args.task, args.tau)
