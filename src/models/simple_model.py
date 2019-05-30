#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-19  in Whu.
"""

from src.models.base_model import BaseModel
from src.models.layers import conv,fc,maxpool,dropout
import tensorflow as tf

class SimpleModel(BaseModel):
    def __init__(self,config):
        super().__init__(config)
        self.num_class = config.num_classes
        self.keep_prob = config.keep_prob
        self.feature_map = None

        # self.build_model() # parameter
        # self.init_saver()

    def build_model(self,input_image,is_training = False):

        conv1 = conv("conv1",input_image,64)
        pool1 = maxpool("pool1",conv1)

        conv2 = conv("conv2",pool1,128)
        pool2 = maxpool("pool2",conv2)

        conv3 = conv("conv3",pool2,256)
        pool3 = maxpool("pool3",conv3)

        fc1 = fc("fc1",pool3,512)
        if is_training:
            fc1 = dropout(fc1,self.keep_prob)
        out = fc("fc2",fc1,self.num_class,relu=False)

        self.feature_map = {"conv1":conv1,"conv3":conv3,"pool1":pool1,"pool3":pool3}
        self.logits = out
        return out


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)