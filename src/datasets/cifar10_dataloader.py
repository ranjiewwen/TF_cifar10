#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-18  in Whu.
"""

import numpy as np
import multiprocessing as mtp
import cv2

# 同级目录引用
from src.datasets.base_dataloader import DataLoaderBase


def parse_data(filename):

    I = np.asarray(cv2.imread(filename))
    I = I.astype(np.float32)
    mean = np.array([113.86538318359375,122.950394140625,125.306918046875])
    I -= mean

    return I

def parse_aug_data(filename):

    I = np.asarray(cv2.imread(filename))
    I = I.astype(np.float32)
    mean = np.array([113.86538318359375,122.950394140625,125.306918046875])
    I -= mean

    return I

class Cifar10_DataLoader(DataLoaderBase):

    def __init__(self,data_file,is_training,config=None):
        # Python3.x 和 Python2.x 的一个区别是: Python 3 可以使用直接使用 super().xxx 代替 super(Class, self).xxx
        super().__init__(config)

        self.works = mtp.Pool(10)
        self.num_classes = self.config.num_classes
        self.batch_size = self.config.batch_size
        self.im_shape = self.config.im_shape

        self.is_training = is_training
        self.images_paths, self.images_lables = self._get_data_list(data_file)
        self.num_images = len(self.images_paths)
        self._perm = None
        self._shuffle_index() # init order

    def _get_data_list(self,file):
        images_paths = []
        images_lables = []
        with open(file) as f:
            lines = f.readlines()
            for l in lines:
                items = l.split()
                images_paths.append(items[0])
                images_lables.append(int(items[1]))
        return images_paths,images_lables

    def _shuffle_index(self):
        '''randomly permute the train order'''
        self._perm = np.random.permutation(np.arange(self.num_images))
        self._cur = 0

    def _get_next_minbatch_index(self):
        """return the indices for the next minibatch"""
        if self._cur + self.batch_size > self.num_images:
            self._shuffle_index()
        next_index = self._perm[self._cur:self._cur + self.batch_size]
        self._cur += self.batch_size
        return next_index


    def get_minibatch(self, minibatch_db):
        """return minibatch datas for train/test"""
        if self.is_training:
            jobs = self.works.map(parse_aug_data, minibatch_db)
        else:
            jobs = self.works.map(parse_data, minibatch_db)
        index = 0
        images_data = np.zeros([self.batch_size, self.im_shape[0], self.im_shape[1], 3])
        for index_job in range(len(jobs)):
            images_data[index, :, :, :] = jobs[index_job]
            index += 1
        return images_data

    def next_batch(self):
        """Get next batch images and labels"""
        db_index = self._get_next_minbatch_index()
        minibatch_db = []
        for i in range(len(db_index)):
            minibatch_db.append(self.images_paths[db_index[i]])

        batch_label = []
        for i in range(len(db_index)):
            batch_label.append(self.images_lables[db_index[i]])

        batch_data = self.get_minibatch(minibatch_db)

        batch_hot_label = np.zeros((self.batch_size,self.num_classes))
        for i,label in enumerate(batch_label):
            batch_hot_label[i][label] = 1

        return batch_data, batch_hot_label


import json
from bunch import Bunch
def get_config_from_json(json_file):
    """
    将配置文件转换为配置类
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)  # 配置字典

    config = Bunch(config_dict)  # 将配置字典转换为类

    return config, config_dict

if __name__ == "__main__":
    config, _ = get_config_from_json("./config/cifar10_config.json")
    cifar10 = Cifar10_DataLoader("F:\\dataset\\cifar10\\train.txt",True,config)

    for i in range(10):
        image_batch,lable_batch = cifar10.next_batch()
        print(image_batch.shape,lable_batch.shape)



