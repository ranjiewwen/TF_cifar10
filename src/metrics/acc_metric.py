#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-29  in Whu.
"""

import tensorflow as tf

def get_accuracy(prediction,one_hot_label):

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(one_hot_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy