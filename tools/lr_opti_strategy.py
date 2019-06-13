#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-31  in Whu.
"""

import tensorflow as tf


## lr: https://blog.csdn.net/akadiao/article/details/79560731

## "lr": 0.001,
def get_lr_strategy(config,global_step):

    if config.lr_plan == "exp_decay":
        learning_rate = tf.train.exponential_decay(learning_rate=config.lr,
                                           global_step=global_step,
                                           decay_rate=0.9,
                                           decay_steps=1800,
                                           staircase = True
                                           )
    elif config.lr_plan == "piecewise":
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    boundaries=[int(config.max_iter * 0.6), int(config.max_iter * 0.8)],
                                                    values=[config.lr, config.lr * 0.2, config.lr * 0.01])
    return learning_rate