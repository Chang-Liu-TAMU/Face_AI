# @Time: 2022/5/27 18:16
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:lru.py

#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
================================================================
Copyright (C) 2019 * Ltd. All rights reserved.
PROJECT      :  Face_AI
FILE_NAME    :  lru_cache
AUTHOR       :  DAHAI LU
TIME         :  2021/6/2 下午4:37
PRODUCT_NAME :  PyCharm
================================================================
"""

import collections


class LRUCache(collections.OrderedDict):

    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def get(self, key: str) -> int:
        if key not in self:
            return -1
        self.move_to_end(key)
        return self[key]

    def put(self, key: str, value: int, call_back_fun=None):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.capacity:
            key, value = self.popitem(last=False)
            if call_back_fun is not None:
                call_back_fun(value, key)


