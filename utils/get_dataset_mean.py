#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-19  in Whu.

# reference: https://blog.csdn.net/yql_617540298/article/details/83617512
# 另一种实现：将所有图的r,g,b叠加后在进行计算均值
"""

import os
import numpy as np
import cv2

image_dir = "E:\\datasets\\train\\"
image_list = os.listdir(image_dir)
r_mean = []
g_mean = []
b_mean = []
for i,img_name in enumerate(image_list):
    im = cv2.imread(image_dir+img_name)  # plt.imread和PIL.Image.open读入的都是RGB顺序，而cv2.imread读入的是BGR顺序
    im_r = im[:,:,2]
    im_g = im[:,:,1]
    im_b = im[:,:,0]

    im_r_mean = np.mean(im_r)
    im_g_mean = np.mean(im_g)
    im_b_mean = np.mean(im_b)

    r_mean.append(im_r_mean)
    g_mean.append(im_g_mean)
    b_mean.append(im_b_mean)

    if i%50 == 0:
        print("the {}th image {} RGB mean is :{},{},{}\n".format(i,img_name,im_r_mean,im_g_mean,im_b_mean))
mean = [0,0,0]
mean[0] = np.mean(r_mean)
mean[1] = np.mean(g_mean)
mean[2] = np.mean(b_mean)

print("----------the datasets RGB mean is: {},{},{}----------\n".format(mean[0],mean[1],mean[2]))

## ----------the datasets RGB mean is: 125.306918046875,122.950394140625,113.86538318359375----------