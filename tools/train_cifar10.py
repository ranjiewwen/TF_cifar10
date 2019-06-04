#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-29  in Whu.
"""

import tensorflow as tf
from src.datasets.cifar10_dataloader import Cifar10_DataLoader
from src.models.simple_model import SimpleModel as TrainModel
from src.loss.cross_entropy import cross_entropy
from src.metrics.acc_metric import get_accuracy
from tools.utils import setup_logger
from tools.utils import get_args,create_dirs
from tools.lr_opti_strategy import get_lr_strategy
from config.config import process_config
import numpy as np
from datetime import datetime
import time


def main(config):
    graph = tf.Graph()
    with graph.as_default():
        ## import data
        train_dataloader = Cifar10_DataLoader(config.train_txt, True, config)
        val_dataloader = Cifar10_DataLoader(config.val_txt, False, config)


        ## Input placeholder
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32,[None,config.im_shape[0],config.im_shape[1],3])
            y = tf.placeholder(tf.float32,[None,config.num_classes])
            dropout_keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope("models"):
            model = TrainModel(config,True ,dropout_keep_prob = dropout_keep_prob)
            logits = model.build_model(x)
            prediction = tf.nn.softmax(logits)

        with tf.name_scope("total_loss"):
            cross_entropy_loss = cross_entropy(logits,y)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n([cross_entropy_loss] + regularization_losses)
        tf.summary.scalar('cross_entropy',loss)

        with tf.name_scope("trian"):
            global_step = tf.Variable(0, trainable=False)  # global_step变量是不可训练的,参考：https://blog.csdn.net/xiaolifeidaoer/article/details/88218224
            learning_rate = get_lr_strategy(config,global_step)
            if config.optimize == "sgd":
                train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
            else:
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('global_step', global_step)

        with tf.name_scope("accuracy"):
            accuracy = get_accuracy(prediction,y)
        tf.summary.scalar('accuracy', accuracy)

        summary_op = tf.summary.merge_all()
        # Instantiate a SummaryWriter to output summaries and the Graph.
        timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H-%M')
        train_writer = tf.summary.FileWriter(config.summary_dir+'/train/'+'{}_{}'.format(config.exp_name, timestamp),filename_suffix=config.exp_name)
        val_writer = tf.summary.FileWriter(config.summary_dir+'/test/'+'{}_{}'.format(config.exp_name, timestamp),filename_suffix=config.exp_name)

        model.init_saver()

    train_step_per_epoch = np.floor(len(train_dataloader.images_paths)/config.batch_size).astype(np.int16)
    val_step_per_epoch = np.floor(len(val_dataloader.images_paths)/config.batch_size).astype(np.int16)
    with tf.Session(graph=graph) as sess:

        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)
        model.load(sess) # load pre_train model

        logger.info("Strat training...")
        for step in range(config.max_iter):
            train_batch,label_batch = train_dataloader.next_batch()
            sess.run(train_step, feed_dict = {x:train_batch,y:label_batch,dropout_keep_prob: config.keep_prob})

            if step % config.trian_display_step == 0:
                batch_loss,batch_acc,summary_str = sess.run([loss,accuracy,summary_op],
                                                            feed_dict={x:train_batch,y:label_batch,dropout_keep_prob:1.0})
                train_writer.add_summary(summary_str,step)
                logger.info("step {} : Training Batch Loss = {:.4f}, Batch Accuracy = {:.4f}".format(step,batch_loss,batch_acc))

            if step % train_step_per_epoch == 0:
                model.save(sess,step) # every epoch save train model
                total_val_acc = 0.
                val_cnt = 0
                for _ in range(val_step_per_epoch):
                    val_batch,val_label = val_dataloader.next_batch()
                    val_acc = sess.run(accuracy,feed_dict={x:val_batch,y:val_label,dropout_keep_prob: 1.})
                    total_val_acc += val_acc
                    val_cnt += 1
                mean_val_acc = total_val_acc/val_cnt
                s = tf.Summary(value = [tf.Summary.Value(tag="valiation_accuracy",simple_value = mean_val_acc)])
                val_writer.add_summary(s,step)
                logger.info(" Validation step {} : Validation Accuracy = {:.4f}".format(step,mean_val_acc))

        train_writer.close()
        val_writer.close()
        logger.info("Optimization finish!")



if __name__ == "__main__":

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.ckpt_dir])
    # create logger info
    global logger
    logger = setup_logger("TF_cifar10", config.ckpt_dir,"train_cifar10_")
    logger.info(config)

    main(config)

    logger.info("---------Train Finished !----------")