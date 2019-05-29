from pathlib import Path

from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Input, Lambda
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential

import numpy as np
import tensorflow as tf
import time
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from scipy.io import loadmat

from subsets.sample_subsets import gumbel_keys, continuous_topk


# Set parameters:
tf.set_random_seed(10086)
np.random.seed(10086)
batch_size = 1000
epochs = 200
pretrain_epochs = 10
currdir = Path(__file__).resolve().parent


def load_data(dataset):
    """
    Load data
    """
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        shuffle_idx = np.arange(x_train.shape[0])
        np.random.shuffle(shuffle_idx)
        x_train = x_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        # flatten
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        # input image dimensions
        img_rows, img_cols = 28, 28
        input_shape = (img_rows * img_cols, )
    elif dataset == '20newsgroups':
        mat = loadmat(str(currdir / 'data/20news_w100.mat'))
        X = np.asarray(mat['documents'].todense()).T
        y = np.squeeze(mat['newsgroups']) - 1

        # move some data over so that there is 15000 in x_train
        shuffle_idx = np.arange(X.shape[0])
        np.random.shuffle(shuffle_idx)
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        x_train = X[:15000]
        y_train = y[:15000]
        x_test = X[15000:]
        y_test = y[15000:]
        input_shape = (100, )
    else:
        raise ValueError('dataset not supported')

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return x_train, y_train, x_test, y_test, input_shape


def distance_matrix_np(X):
    xsq = np.sum(np.square(X), 1)
    dists = xsq[:, np.newaxis] - 2 * X.dot(X.T) + xsq[np.newaxis, :]
    return dists


def distance_matrix(X):
    xsq = tf.reduce_sum(tf.square(X), 1)
    dists = (tf.expand_dims(xsq, 1)
             - 2 * tf.matmul(X, tf.transpose(X)) + tf.expand_dims(xsq, 0))
    return dists


def create_model(X_ph, out_dim, dataset):
    """
    model

    """
    if dataset == 'mnist':
        layers = []
        net = Dense(500, activation='sigmoid')
        layers.append(net)
        net_X = net(X_ph)
        net = Dense(500, activation='sigmoid')
        layers.append(net)
        net_X = net(net_X)
        net = Dense(2000, activation='sigmoid')
        layers.append(net)
        net_X = net(net_X)
        net = Dense(out_dim, activation=None)
        layers.append(net)
        net_X = net(net_X)
    elif dataset == '20newsgroups':
        layers = []
        net = Dense(150, activation='sigmoid')
        layers.append(net)
        net_X = net(X_ph)
        net = Dense(150, activation='sigmoid')
        layers.append(net)
        net_X = net(net_X)
        net = Dense(500, activation='sigmoid')
        layers.append(net)
        net_X = net(net_X)
        net = Dense(out_dim, activation=None)
        layers.append(net)
        net_X = net(net_X)

    return net_X, layers


def trustworthiness_loss(orig_dists, new_dists, k, tau):
    # orig dists is batch_size x batch_size
    # new dists is batch_size x batch_size

    # TODO assuming orig_dists.shape[1] == batch_size
    mask = tf.ones_like(orig_dists, dtype=tf.bool)
    mask = tf.linalg.set_diag(mask, tf.zeros_like(orig_dists, dtype=tf.bool)[0])
    orig_dists = tf.boolean_mask(orig_dists, mask)
    new_dists = tf.boolean_mask(new_dists, mask)
    orig_dists = tf.reshape(orig_dists, [batch_size, batch_size-1])
    new_dists = tf.reshape(new_dists, [batch_size, batch_size-1])

    # sample from subset distribution of neighbors
    # where the subset distribution is proportional to exp(-dists)
    orig_dist_gumbel_keys = gumbel_keys(-orig_dists)
    _, topk_idxs = tf.nn.top_k(orig_dist_gumbel_keys, k=k, sorted=True)
    topk_idxs = tf.stop_gradient(topk_idxs)

    # sample from subset distribution of neighbors
    new_dist_gumbel_keys = gumbel_keys(-new_dists)
    new_topk = continuous_topk(new_dist_gumbel_keys, k, tau, separate=True)

    loss = 0.0
    for i in range(k):
        onehot = tf.one_hot(topk_idxs[:, i], batch_size-1)

        # smoothing the new_topk with a small bias is important
        loss += -tf.reduce_sum(onehot * tf.log(new_topk[i] + 1e-8), 1) / tf.exp(float(i))
    return loss


def pretrain(all_layers, x_train, batch_size, epochs):
    for i in range(len(all_layers)):
        curr_layers = all_layers[0: i+1]
        decoder = Dense(x_train.shape[1], activation='linear')
        curr_layers.append(decoder)
        ae = Sequential(curr_layers)
        ae.compile(loss='mean_squared_error', optimizer='rmsprop')
        ae.fit(x_train, x_train, batch_size=batch_size, epochs=epochs)


def trustworthiness_metric(orig_dists, new_dists, k):
    n = orig_dists.shape[0]
    orig_sorted_idxs = np.argsort(orig_dists, 1)
    new_sorted_idxs = np.argsort(new_dists, 1)

    trustworthiness = 0
    for i in range(n):
        for j in range(k):
            rank_i_j = np.where(orig_sorted_idxs[i] == new_sorted_idxs[i, j+1])[0][0] + 1
            # check if its in the first k+1 since it should have 0 distance
            # to itself
            trustworthiness += max(0, rank_i_j - k - 1)
    trustworthiness = 1 - 2 * trustworthiness / (n*k*(2*n-3*k-1))
    return trustworthiness


def evaluate_embedding(x_train_feats, y_train, x_val, x_val_feats, y_val, model):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train_feats, y_train)
    knn_preds = knn.predict(x_val_feats)
    knn_acc = np.mean(knn_preds == y_val)
    print(f'KNN Test Accuracy: {knn_acc}. KNN Error: {1-knn_acc}')

    # trustworthiness
    x_val_dists = distance_matrix_np(x_val)
    x_val_feat_dists = distance_matrix_np(x_val_feats)
    val_trustworthiness = trustworthiness_metric(x_val_dists, x_val_feat_dists, 12)
    print(f"Val Trustworthiness: {val_trustworthiness}")


def train(train=True, tau=1.0, out_dim=2, dataset='mnist', do_pretrain=True, do_plot=False, k=1):
    """
    Generate scores on features
    """

    print('Loading dataset...')
    x_train, y_train, x_val, y_val, input_shape = load_data(dataset)
    print('Creating model...')
    with tf.variable_scope('model'):
        X_ph = Input(shape=input_shape, dtype='float32')
        feats_X, all_layers = create_model(X_ph, out_dim, dataset)
        # compute distances
        X_dists = Lambda(lambda x: distance_matrix(x))(X_ph)
        feat_dists = Lambda(lambda x: distance_matrix(x))(feats_X)

    def loss(y_true, y_pred):
        return trustworthiness_loss(X_dists, feat_dists, k, tau)

    pretrain_str = 'pretrain' if do_pretrain else 'no_pretrain'
    filepath = f"models/subsets_tsne_{dataset}_{out_dim}_{pretrain_str}_{tau}.hdf5"
    if train:
        if do_pretrain:
            # first do layer wise pretraining
            pretrain(all_layers, x_train, batch_size, epochs=pretrain_epochs)

        model = Model(inputs=[X_ph], outputs=feats_X)
        model.compile(loss=loss, optimizer='adam')

        checkpoint = ModelCheckpoint(
            filepath, monitor='loss',
            verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]
        st = time.time()
        # Note: y_train, y_train_val are not used.
        model.fit(
            x_train, y_train,
            callbacks=callbacks_list,
            epochs=epochs, batch_size=batch_size, shuffle=True)
        duration = time.time() - st
        print('Training time is {}'.format(duration))

    model = Model(inputs=[X_ph], outputs=feats_X)
    model.compile(loss=loss, optimizer='adam')

    model.load_weights(filepath, by_name=True)

    # Test KNN accuracy with k=1
    st = time.time()
    print("Evaluate subset SNE")
    x_train_feats = model.predict(x_train, verbose=1, batch_size=batch_size)
    x_val_feats = model.predict(x_val, batch_size=batch_size)
    evaluate_embedding(x_train_feats, y_train, x_val, x_val_feats, y_val, model)

    if out_dim == 2 and do_plot:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        color_cycle = prop_cycle.by_key()['color']
        colors = [color_cycle[i] for i in y_val]
        plt.scatter(x_val_feats[:, 0], x_val_feats[:, 1], c=colors, s=0.8)
        plt.show()

        colors = [color_cycle[i] for i in y_train]
        plt.scatter(x_train_feats[:, 0], x_train_feats[:, 1], c=colors, s=0.2)
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--out_dim', type=int, default=2)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--do_pretrain', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs

    models_dir = Path('models/')
    models_dir.mkdir(exist_ok=True)
    data_dir = Path('data/')
    data_dir.mkdir(exist_ok=True)

    train(args.train, tau=args.tau, out_dim=args.out_dim, dataset=args.dataset, do_pretrain=args.do_pretrain, do_plot=args.do_plot, k=args.k)
