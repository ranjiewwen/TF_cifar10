#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-18  in Whu.

some code borrow from: https://github.com/d2l-ai/d2l-zh/blob/master/chapter_computer-vision/kaggle-gluon-cifar10.md
"""

import os
import pandas as pd
import numpy as np

def read_csv_to_txt(csv_file,data_dir):
    df = pd.read_csv(os.path.join(data_dir,csv_file))
    id = df['id']
    lable = df['label']

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

# read_label_file函数将用来读取训练数据集的标签文件
def read_label_file(data_dir, label_file):
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过文件头行（栏名称）
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((int(idx), label) for idx, label in tokens))
    labels = set(idx_label.values())

    return idx_label,labels

# 定义reorg_train_valid函数来从原始训练集中切分出验证集
def get_train_valid(data_dir, train_dir, labels, idx_label, valid_ratio):
    image_label_pair = []
    labels = list(labels)
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = int(train_file.split('.')[0])
        label = idx_label[idx]
        label_int = labels.index(label) # 获取list元素下标
        image_label_pair.append((os.path.join(data_dir,train_dir,train_file),label_int))

    nums = len(image_label_pair)
    order = np.random.permutation(np.arange(nums))
    with open(os.path.join(data_dir,"train.txt"),'w') as f:
        for i in range(int(nums*(1-valid_ratio))):
            #f.writelines(image_label_pair[order[i]])
            pair = image_label_pair[order[i]]
            f.write("%s %d\n" % (pair[0],pair[1]))

    with open(os.path.join(data_dir,"val.txt"),'w') as f:
        for i in range(int(nums*(1-valid_ratio)),nums):
            pair = image_label_pair[order[i]]
            f.write("%s %d\n" % (pair[0], pair[1]))

    with open(os.path.join(data_dir,"cifar10.name.txt"),'w') as f:
        f.write(str(labels))

    print(" finish get train.txt and val.txt !")



if __name__=="__main__":
    csv_file = "trainLabels.csv"
    train_dir = "train"
    data_dir = "E:\\datasets\\cifar10"

    read_csv_to_txt(csv_file,data_dir)

    idx_label,labels = read_label_file(data_dir,csv_file)
    get_train_valid(data_dir,train_dir,labels,idx_label,0.1)

    pass