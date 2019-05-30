#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-27  in Whu.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from config.config import process_config
from src.models.simple_model import SimpleModel

import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def process_args():
    """
    Parse input arguments
    :return:
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model_path",type=str,default="../experiments/cifar10/checkpoint/iteration_47736.ckpt-47736",help="pre-train best model")
    parser.add_argument("--file_name",type=str,default="F:\\dataset\\cifar10\\train\\11093.png",help="test image to class")
    parser.add_argument(
        '-c', '--config',
        metavar='C',
        default='../config/cifar10_config.json',
        help='The Configuration file')

    args = parser.parse_args()
    config = process_config(args.config)

    return args,config

class_id2name ={0:'automobile', 1:'truck', 2:'bird', 3:'horse', 4:'frog', 5:'cat', 6:'dog', 7:'airplane', 8:'ship', 9:'deer'}
class_name =['automobile', 'truck', 'bird', 'horse', 'frog', 'cat', 'dog', 'airplane', 'ship', 'deer']

def main(args,config):
    image_raw = cv2.imread(args.file_name)
    image = np.asarray(image_raw)
    image = image.astype(np.float32)
    mean = np.array([113.86538318359375, 122.950394140625, 125.306918046875])
    image -= mean
    img = image.reshape([1,32,32,3])
    graph = tf.Graph()
    with graph.as_default():

        ## Input placeholder
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, config.im_shape[0], config.im_shape[1], 3])

        with tf.name_scope("models"):
            model = SimpleModel(config)
            logits = model.build_model(x, False)
            prediction = tf.nn.softmax(logits)

        model.init_saver()

    with tf.Session(graph=graph) as sess:
        model.saver.restore(sess,args.model_path)
        pred = sess.run(prediction,feed_dict={x:img})

        class_id = np.argmax(pred, 1)
        class_name = class_id2name.get(class_id[0])
        print("this file {} class id and name is :{}={}".format(args.file_name,class_id,class_name))

        plt.figure(args.file_name)
        plt.imshow(image_raw,aspect ='auto')
        plt.title("prediction class name is :{}.".format(class_name))
        plt.axis("off")
        plt.show()


if __name__=="__main__":

    args, config = process_args()
    main(args, config)
