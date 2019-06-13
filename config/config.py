#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-22  in Whu.
"""

import json
from bunch import Bunch
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join("../experiments", config.exp_name, "summary")
    config.ckpt_dir = os.path.join("../experiments", config.exp_name, "checkpoint")

    return config


## input three method:

#  args = parser.parse_args(): https://github.com/ranjiewwen/TF_RankIQA/blob/master/tools/train_clive.py
#  get_config_from_json(): json or yaml file
#  FLAGS = tf.app.flags.FLAGS: https://github.com/dgurkaynak/tensorflow-cnn-finetune/blob/master/alexnet/finetune.py