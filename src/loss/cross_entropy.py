#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-29  in Whu.
"""
import tensorflow as tf

def cross_entropy(pred_logits,one_hot_lable):

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_logits,labels=one_hot_lable))

    return loss