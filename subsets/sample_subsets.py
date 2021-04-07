import tensorflow as tf
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
import numpy as np
import scipy


EPSILON = np.finfo(tf.float32.as_numpy_dtype).tiny


def gumbel_keys(w):
    # sample some gumbels
    uniform = tf.random_uniform(
        tf.shape(w),
        minval=EPSILON,
        maxval=1.0)
    z = -tf.log(-tf.log(uniform))
    w = w + z
    return w


def continuous_topk(w, k, t, separate=False):
    khot_list = []
    onehot_approx = tf.zeros_like(w, dtype=tf.float32)
    for i in range(k):
        khot_mask = tf.maximum(1.0 - onehot_approx, EPSILON)
        w += tf.log(khot_mask)
        onehot_approx = tf.nn.softmax(w / t, axis=-1)
        khot_list.append(onehot_approx)
    if separate:
        return khot_list
    else:
        return tf.reduce_sum(khot_list, 0)


def sample_subset(w, k, t=0.1):
    '''
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    w = gumbel_keys(w)
    return continuous_topk(w, k, t)

############################################3
# From https://github.com/ermongroup/neuralsort/
##########################################

def bl_matmul(A, B):
    return tf.einsum('mij,jk->mik', A, B)

def br_matmul(A, B):
    return tf.einsum('ij,mjk->mik', A, B)

def batchwise_matmul(A, B):
    return tf.einsum('mij,mj->mi', A, B)

# s: M x n x 1
# sortnet(s): M x n x n
def sortnet(s, tau = 1):
  A_s = s - tf.transpose(s, perm=[0, 2, 1])
  A_s = tf.abs(A_s)
  # As_ij = |s_i - s_j|

  n = tf.shape(s)[1]
  one = tf.ones((n, 1), dtype = tf.float32)

  B = bl_matmul(A_s, one @ tf.transpose(one))
  # B_:k = (A_s)(one)

  K = tf.range(n) + 1
  # K_k = k

  C = bl_matmul(
    s, tf.expand_dims(tf.cast(n + 1 - 2 * K, dtype = tf.float32), 0)
  )
  # C_:k = (n + 1 - 2k)s

  P = tf.transpose(C - B, perm=[0, 2, 1])
  # P_k: = (n + 1 - 2k)s - (A_s)(one)

  P = tf.nn.softmax(P / tau, -1)
  # P_k: = softmax( ((n + 1 - 2k)s - (A_s)(one)) / tau )

  return P

###############################################
