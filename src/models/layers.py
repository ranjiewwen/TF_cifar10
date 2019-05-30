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

def conv(name, input, out_channel, trainable=True):
    in_channel = int(input.get_shape()[-1])
    with tf.variable_scope(name):
        kernel = tf.get_variable("weights", initializer=tf.truncated_normal([3, 3, in_channel, out_channel], dtype=tf.float32, stddev=1e-1), trainable=trainable)
        biases = tf.get_variable("biases", initializer=tf.constant(0.0, shape=[out_channel]), dtype=tf.float32, trainable=trainable)
        conv_res = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding="SAME")
        res = tf.nn.bias_add(conv_res, biases)
        out = tf.nn.relu(res, name=name)
    return out


def fc(name, input, out_channel,relu =True,trainable=True):
    shape = input.get_shape().as_list()
    if len(shape) == 4:
        size = shape[-1] * shape[-2] * shape[-3]
    else:
        size = shape[1]
    input_data_flat = tf.reshape(input, [-1, size])
    with tf.variable_scope(name):
        weights = tf.get_variable(name="weights", shape=[size, out_channel], dtype=tf.float32, trainable=trainable)
        biases = tf.get_variable(name="biases", shape=[out_channel], dtype=tf.float32, trainable=trainable)
        res = tf.matmul(input_data_flat, weights)
        out = tf.nn.bias_add(res, biases)
        if relu == True:
            out = tf.nn.relu(out)
    return out


def softmax(input, name):
    return tf.nn.softmax(input, name)


def dropout(input, keep_prob):
    return tf.nn.dropout(input, keep_prob)

