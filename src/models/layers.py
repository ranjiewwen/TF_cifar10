#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-19  in Whu.
"""

import tensorflow as tf
from tensorflow.python.ops import math_ops

def maxpool(name, input):
    out = tf.nn.max_pool(input,
                         [1, 2, 2, 1],
                         [1, 2, 2, 1],
                         padding="SAME",
                         name=name)
    return out


def avg_pool(name,input, k_h, k_w, s_h, s_w,padding="SAME"):
    return tf.nn.avg_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)

# https://www.jianshu.com/p/029895d786f7
def global_avg_pool(name,input):
    # Global average pooling.
    output = math_ops.reduce_mean(input, [1, 2], name=name, keepdims=True)
    return output

def _get_variable(name, initializer, weight_decay, dtype, trainable=True):
    "A little wrapper around tf.get_variable to do weight decay"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    return tf.get_variable(name, initializer=initializer, dtype=dtype, regularizer=regularizer,
                           trainable=trainable)

def conv(input, out_channel,weight_decay = 0.0001, trainable=True):

    in_channel = int(input.get_shape()[-1])
    kernel = _get_variable('weights',
                           initializer=tf.truncated_normal([3, 3, in_channel, out_channel],dtype=tf.float32, stddev=1e-1),
                           weight_decay=weight_decay,
                           dtype=tf.float32,
                           trainable=trainable)
    biases = tf.get_variable("biases",
                             initializer=tf.constant(0.0, shape=[out_channel]),
                             dtype=tf.float32,
                             trainable=trainable)
    conv_res = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding="SAME")
    res = tf.nn.bias_add(conv_res, biases)

    return res



def fc(input, out_channel,weight_decay = 0.01,trainable=True):

    shape = input.get_shape().as_list()
    if len(shape) == 4:
        size = shape[-1] * shape[-2] * shape[-3]
    else:
        size = shape[1]
    input_data_flat = tf.reshape(input, [-1, size])

    weights_initializer = tf.truncated_normal([size, out_channel], dtype=tf.float32, stddev=0.01)
    weights = _get_variable('weights',
                            initializer=weights_initializer,
                            weight_decay=weight_decay,
                            dtype=tf.float32,
                            trainable=trainable)
    biases = tf.get_variable(name="biases", shape=[out_channel], dtype=tf.float32, trainable=trainable)
    res = tf.matmul(input_data_flat, weights)
    out = tf.nn.bias_add(res, biases)

    return out


def softmax(input, name):
    return tf.nn.softmax(input, name)


def dropout(input, keep_prob):
    return tf.nn.dropout(input, keep_prob)


## 全连接层用的：https://www.jianshu.com/p/b2d2f3c7bfc7
epsilon = 1e-3
def batch_norm(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)


from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

def get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    "A little wrapper around tf.get_variable to do weight decay"

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer,
                           trainable=trainable)

def batch_norm_wrapper(x, is_training,bn_decay = 0.9997, UPDATE_OPS_COLLECTION = 'resnet_update_ops'):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = get_variable('gamma', params_shape, initializer=tf.ones_initializer())

    moving_mean = get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, bn_decay)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, bn_decay)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, variance_epsilon = 0.001)