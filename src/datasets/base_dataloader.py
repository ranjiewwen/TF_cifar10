#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by admin at  2019-05-18  in Whu.
"""

class DataLoaderBase(object):
    """
    数据加载基类，定义一些公共函数，子类重写一些关键函数
    """
    def __init__(self, config):
        self.config = config
        self.images_paths = []
        self.images_lables = []
        self._cur = 0

    def next_batch(self):
        """
        获取一个batch的数据和标签，训练时IO可能是瓶颈
        :return:
        """
        raise NotImplementedError