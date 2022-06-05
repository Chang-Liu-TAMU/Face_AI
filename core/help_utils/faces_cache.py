# @Time: 2022/5/25 14:41
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:faces_cache.py

#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
================================================================
Copyright (C) 2019 * Ltd. All rights reserved.
PROJECT      :  Face_AI
FILE_NAME    :  face_feature_cache
AUTHOR       :  DAHAI LU
TIME         :  2021/6/3 上午10:43
PRODUCT_NAME :  PyCharm
================================================================
"""

import time
import queue
import numpy as np
import collections
from tqdm import tqdm

import sys
import os.path as osp

dir_path = osp.dirname(osp.dirname(osp.realpath(__file__)))
sys.path.append(dir_path)

from core.help_utils import tools
from core.help_utils.lru import LRUCache
from core.help_utils.logger_utils import internal_logger


def get_time(func):
    def wrapper(*args, **kw):
        t = time.time()
        f = func(*args, **kw)
        print('{}: {}'.format(func.__name__, time.time() - t))
        return f

    return wrapper


class FaceFeatureCache(object):
    @staticmethod
    def parse_input(pic_names_list, face_feature_list, label_list):
        update_value_list = [[None]] * len(face_feature_list)
        for i in range(len(face_feature_list)):
            ugroup, uname, uid = label_list[i]
            features = np.asarray(face_feature_list[i])
            pic_name = pic_names_list[i]
            update_value = [ugroup, uname, uid, features, pic_name]
            update_value_list[i] = update_value
        return update_value_list

    def __init__(self, data, per_uid_capacity: int = 7, logger=None):
        if isinstance(data, (list, tuple)):
            self._data = np.vstack(data)
        else:
            self._data = data
        if logger is None:
            self._logger = internal_logger()
        else:
            self._logger = logger
        self._is_initialized = False
        self._need_update_cache = True
        self._cached_indexes = []
        self._per_uid_capacity = per_uid_capacity
        self._id_maps = collections.defaultdict(LRUCache)
        self._free_queue = queue.Queue()
        self._init_id_maps()

    def _get_available_index(self):
        if self._free_queue.empty():
            if self.is_valid():
                return len(self._data)
            else:
                return -1
        else:
            return self._free_queue.get()

    @get_time
    def _init_id_maps(self):
        if self._data is None or len(self._data) == 0:
            self._is_initialized = False
            self._logger.warning("[FaceFeatureCache._init_id_maps] database is Empty, please register first!")
            return
        else:
            self._need_update_cache = True
            self._is_initialized = True

        u_ids = self._data[:, 2]

        unique_ids = set(list(u_ids))

        for uid in tqdm(unique_ids, desc="Create face dataset cache"):
            if uid not in self._id_maps:
                self._id_maps[uid] = LRUCache(self._per_uid_capacity)

            id_indexes = np.where(u_ids == uid)[0]
            u_groups = self._data[id_indexes, 0]
            u_names = self._data[id_indexes, 1]
            file_names = self._data[id_indexes, 4]
            for i in range(len(file_names)):
                pic_names_list = file_names[i].split("_")
                if len(pic_names_list) > 3:
                    file_name = "_".join([u_groups[i], u_names[i], uid, *pic_names_list[3:]])
                else:
                    file_name = "_".join([u_groups[i], u_names[i], uid])
                self._id_maps[uid].put(key=file_name, value=id_indexes[i], call_back_fun=self._empty_data_by_index)

    def _get_cached_indexes(self):
        if self._need_update_cache:
            self._cached_indexes = []
            self._need_update_cache = False
            for lru_cache in self._id_maps.values():
                self._cached_indexes.extend(list(lru_cache.values()))
        return list(set(self._cached_indexes))

    def _get_cached_indexes_by_id(self, uid):
        if not self.contains(uid=uid):
            return []

        lru_cache = self._id_maps.get(uid)
        return list(lru_cache.values())

    def _get_cached_index_by_uid_key(self, uid, key):
        if not self.contains(uid=uid):
            return -1

        lru_cache = self._id_maps.get(uid)
        return lru_cache.get(key=key)

    def is_valid(self):
        return self._is_initialized

    def _empty_data_by_index(self, indexes, key_dummy=None):
        if self.is_valid():
            if isinstance(indexes, (int, np.int64)):
                indexes = [indexes]
            for index in set(indexes):
                self._free_queue.put(index)
            if len(indexes) != 0:
                self._data[indexes, :] = np.zeros((len(indexes), self._data.shape[1]))

    def contains(self, uid):
        return uid in self._id_maps

    def _append_data(self, update_value):
        self._data = np.row_stack((self._data, update_value))

    def _set_data(self, indexes, update_value):
        if isinstance(indexes, int):
            indexes = [indexes]
        self._data[indexes, :] = update_value

    def _update(self, update_value):
        ugroup = update_value[0]
        uname = update_value[1]
        uid = update_value[2]
        pic_name = update_value[-1]
        if not self.contains(uid):
            self._id_maps[uid] = LRUCache(self._per_uid_capacity)

        available_index = self._get_available_index()
        if available_index >= 0:
            self._need_update_cache = True
            self._id_maps[uid].put(key=pic_name, value=available_index, call_back_fun=self._empty_data_by_index)
            if available_index == len(self._data):
                self._append_data(update_value)
            else:
                self._set_data(available_index, update_value)
            self._logger.info(
                "Register staff or leader: group={}, name={}, id={} success!".format(ugroup, uname, uid))
            return True
        else:
            self._logger.warning("database is Empty, please register first!")
            return False

    def put(self, update_value_list):
        if not self.is_valid():
            self._logger.warning("[FaceMachine._update_cache] Database is empty and initialize database cache...")
            if len(update_value_list):
                self._data = np.vstack(update_value_list)
                self._init_id_maps()
            return

        # add or update items
        for update_value in update_value_list:
            ugroup = update_value[0]
            uname = update_value[1]
            uid = update_value[2]
            pic_name = update_value[-1]
            index = self._get_cached_index_by_uid_key(uid=uid, key=pic_name)
            if not self.contains(uid):
                self._id_maps[uid] = LRUCache(self._per_uid_capacity)

            if index >= 0:  # duplicated pic_name
                self._set_data(index, update_value)
                self._id_maps[uid].put(key=pic_name, value=index, call_back_fun=self._empty_data_by_index)
                self._logger.info(
                    "Update staff or leader: group={}, name={}, id={} filename={} success!".format(
                        ugroup, uname, uid, pic_name))
            else:
                if not self._update(update_value):
                    break

    def pop(self, uids):
        if not self.is_valid():
            return
        if isinstance(uids, str):
            uids = [uids]
        # clear cache
        for uid in set(uids):
            indexes = self._get_cached_indexes_by_id(uid=uid)
            if len(indexes) > 0:
                self._id_maps.pop(uid)
                self._need_update_cache = True
                self._empty_data_by_index(indexes)
                self._logger.info("Remove {} staff or leader cache with id {} !".format(len(indexes), uid))
            else:
                self._logger.info("cannot find id {} in staff or leader cache!".format(uid))
        if len(self._id_maps) == 0:
            self._data = None
            self._is_initialized = False
            self._need_update_cache = True
            self._free_queue = queue.Queue()

    def replace(self, update_value_list):
        if not self.is_valid():
            return

        # clear cache
        for update_value in update_value_list:
            uid = update_value[2]
            indexes = self._get_cached_indexes_by_id(uid=uid)
            if len(indexes) > 0:
                self._need_update_cache = True
                self._empty_data_by_index(indexes)
                self._id_maps.pop(uid)

        for i, update_value in enumerate(update_value_list):
            if not self._update(update_value):
                break

    def get_data(self):
        cached_indexes = self._get_cached_indexes()
        if len(cached_indexes) == 0:
            return np.zeros((0, self._data.shape[1]))
        return self._data[cached_indexes, :]

    def get_group_names(self):
        cached_indexes = self._get_cached_indexes()
        if len(cached_indexes) == 0:
            return []
        return self._data[cached_indexes, 0]

    def find(self, u_ids, return_embedding=False):
        result = {}
        if isinstance(u_ids, str):
            u_ids = [u_ids]

        for uid in set(u_ids):
            cached_indexes = self._get_cached_indexes_by_id(uid)
            if len(cached_indexes) > 0:
                res_array = self._data[cached_indexes, :]
                if not return_embedding:
                    res_array = np.take(res_array, indices=[0, 1, 2, 4], axis=1)
                else:
                    res_array[:, 3] = [feature.tolist() for feature in res_array[:, 3]]
                res = res_array.tolist()
            else:
                res = []
            info = dict()
            info["match_number"] = len(res)
            info["match_items"] = res
            result[uid] = info
        return result

    def get_features(self, uid):
        cached_indexes = self._get_cached_indexes_by_id(uid)
        if len(cached_indexes) == 0:
            return []
        return self._data[cached_indexes, 3]

    def get_filenames(self, uid):
        cached_indexes = self._get_cached_indexes_by_id(uid)
        if len(cached_indexes) == 0:
            return []
        return self._data[cached_indexes, 4]


if __name__ == '__main__':

    user_id = "7156"
    pop_ids = ["7156", "6596", "999999"]
    pic_names_list = ["松江班_张永杰_7156_1_20190416105135.jpg", "测试班_佚名_999999_1_234234546.jpg"]
    face_features = [np.asarray([1] * 512) for i in range(2)]
    labels = [("松江班", "张永杰", "7156"), ("测试班", "佚名", "999999")]
    value_list = FaceFeatureCache.parse_input(pic_names_list=pic_names_list,
                                              face_feature_list=face_features,
                                              label_list=labels)

    feature_cache = FaceFeatureCache(data=value_list, per_uid_capacity=7, logger=None)

    # get test
    if feature_cache.contains(user_id):
        print(feature_cache.get_filenames(user_id))
        print(feature_cache.get_features(user_id))

    # put test
    feature_cache.put(update_value_list=value_list)
    feature_cache.get_data()

    # replace test
    feature_cache.replace(update_value_list=value_list)
    feature_cache.get_data()

    # pop test
    feature_cache.pop(pop_ids)
    feature_cache.put(update_value_list=value_list)
    feature_cache.get_data()

    # lru test
    last_id = 2344
    pic_names_list = ["松江班_张永杰_7156_1_{}.jpg".format(tools.generate_unique_id()) for i in range(8)]
    face_features = [np.asarray([1] * 512) for i in range(8)]
    labels = [("松江班", "张永杰", "7156") for i in range(8)]
    value_list = feature_cache.parse_input(pic_names_list=pic_names_list,
                                           face_feature_list=face_features,
                                           label_list=labels)
    feature_cache.put(update_value_list=value_list)
    feature_cache.get_data()
