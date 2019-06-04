#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-19  in Whu.
"""

from src.models.base_model import BaseModel
from src.models.layers import conv,fc,maxpool,dropout,batch_norm_wrapper
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

        with tf.variable_scope("conv1"):
            conv1 = conv(input_image,64)
            relu1 = tf.nn.relu(conv1)
            bn1 = batch_norm_wrapper(relu1,self.is_training,self.config.moving_ave_decay,self.config.UPDATE_OPS_COLLECTION)
        pool1 = maxpool("pool1",bn1)

        with tf.variable_scope("conv2"):
            conv2 = conv(pool1,128)
            relu2 = tf.nn.relu(conv2)
            bn2 = batch_norm_wrapper(relu2, self.is_training,self.config.moving_ave_decay,self.config.UPDATE_OPS_COLLECTION)
        pool2 = maxpool("pool2",bn2)

        with tf.variable_scope("conv3"):
            conv3 = conv(pool2,256)
            relu3 = tf.nn.relu(conv3)
            bn3 = batch_norm_wrapper(relu3, self.is_training,self.config.moving_ave_decay,self.config.UPDATE_OPS_COLLECTION)
        pool3 = maxpool("pool3",bn3)

        with tf.variable_scope('fc1'):
            fc1 = fc(pool3,256)
            fc1 = tf.nn.relu(fc1)
            bn_fc = batch_norm_wrapper(fc1, self.is_training, self.config.moving_ave_decay,
                                       self.config.UPDATE_OPS_COLLECTION)

            # fc1 = tf.cond(self.is_training,
            #               lambda :dropout(fc1,self.keep_prob),
            #               lambda :fc1
            #               )
            # if self.is_training: # tensor
            #     fc1 = dropout(fc1,self.keep_prob)

        with tf.variable_scope('fc2'):
            out = fc(bn_fc,self.num_class)

        self.feature_map = {"conv1":conv1,"conv3":conv3,"pool1":pool1,"pool3":pool3}
        self.logits = out
        return out


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)