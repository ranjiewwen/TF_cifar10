#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-19  in Whu.
"""

from src.models.base_model import BaseModel
from src.models.layers import conv,fc,maxpool,dropout
import tensorflow as tf

class SimpleModel(BaseModel):
    def __init__(self,config,is_training,dropout_keep_prob = 0.5):
        super().__init__(config)
        self.num_class = config.num_classes
        self.keep_prob = dropout_keep_prob
        self.is_training = is_training
        self.feature_map = None

        # self.build_model() # parameter
        # self.init_saver()

    def build_model(self,input_image):

        with tf.variable_scope("conv1_1"):
            conv1 = conv(input_image,64)
            relu1 = tf.nn.relu(conv1)

        with tf.variable_scope("conv1_2"):
            conv1 = conv(relu1,64)
            relu1 = tf.nn.relu(conv1)
        pool1 = maxpool("pool1",relu1)

        with tf.variable_scope("conv2_1"):
            conv2 = conv(pool1,128)
            relu2 = tf.nn.relu(conv2)
        with tf.variable_scope("conv2_2"):
            conv2 = conv(relu2,128)
            relu2 = tf.nn.relu(conv2)
        pool2 = maxpool("pool2",relu2)

        with tf.variable_scope("conv3_1"):
            conv3 = conv(pool2,256)
            relu3 = tf.nn.relu(conv3)
        with tf.variable_scope("conv3_2"):
            conv3 = conv(relu3,256)
            relu3 = tf.nn.relu(conv3)
        pool3 = maxpool("pool3",relu3)

        with tf.variable_scope("conv4_1"):
            conv4 = conv(pool3,256)
            relu4 = tf.nn.relu(conv4)
        with tf.variable_scope("conv4_2"):
            conv4 = conv(relu4,256)
            relu4 = tf.nn.relu(conv4)
        pool4 = maxpool("pool3",relu4)

        with tf.variable_scope('fc1'):
            fc1 = fc(pool4,256)
            fc1 = tf.nn.relu(fc1)

            if self.is_training:
                fc1 = dropout(fc1,self.keep_prob)

        with tf.variable_scope('fc2'):
            out = fc(fc1,self.num_class)

        self.feature_map = {"conv1":conv1,"conv3":conv3,"pool1":pool1,"pool3":pool3}
        self.logits = out
        return out


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)