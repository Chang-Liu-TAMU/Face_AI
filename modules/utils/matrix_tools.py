# @Time: 2022/5/24 13:49
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:matrix_tools.py

# -*- coding:utf-8 -*-

import cv2
import math
import numpy as np
import collections
from copy import copy
from collections import Counter
# from skimage import transform as trans


# deprecated
def compare_embedding_without_matrix(pred_emb, feature_dicts, threshold, group_name=None, distance_metric=0, top_k=7):
    # 为bounding_box 匹配标签
    dataset_emb = []
    names_list = []
    if group_name is None or group_name == "":
        for group_name, value in feature_dicts.items():
            names_list.extend(value[0])
            dataset_emb.extend(list(value[1]))
    else:
        try:
            names_list.extend(feature_dicts[group_name][0])
            dataset_emb.extend(feature_dicts[group_name][1])
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}'.format(e)
            print(errorMessage)
            raise errorMessage
        else:
            pass

    dataset_emb = np.asarray(dataset_emb)
    pred_num = len(pred_emb)
    dataset_num = len(dataset_emb)
    pred_name = []
    ids = []
    distances = []
    top_k_info_list = []
    top_k_same_label_list = []
    for i in range(pred_num):
        dist_list = []
        for j in range(dataset_num):
            dist = distance(pred_emb[i, :], dataset_emb[j, :], distance_metric=distance_metric)
            dist_list.append(dist)
        temp_dist_list = copy.copy(dist_list)
        temp_dist_list.sort()
        top_k_dis = temp_dist_list[:min(top_k, len(temp_dist_list))]
        top_k_info = [(names_list[dist_list.index(dis)], dis) for dis in top_k_dis]
        top_k_same_label = [(names_list[dist_list.index(dis)].split('_')[2], dis) for dis in top_k_dis]
        top_k_same_label_list.append(top_k_same_label)
        top_k_info_list.append(top_k_info)
        min_value = min(dist_list)
        distances.append(min_value)
        if min_value > threshold:
            pred_name.append('unknown')
            ids.append('unknown')
        else:
            pred_name.append(names_list[dist_list.index(min_value)])
            ids.append(names_list[dist_list.index(min_value)].split('_')[2])

    # remove replicated labels
    id_counter = Counter(ids)
    same_id_list = [id for id, num in id_counter.items() if num >= 2]
    same_index_dict = collections.defaultdict(list)
    for i, name in enumerate(ids):
        if name in same_id_list:
            same_index_dict[name].append(i)
    for key, value in same_index_dict.items():
        dis_l = [distances[v] for v in value]
        min_value = min(dis_l)
        index_value = distances.index(min_value)
        same_index_dict[key].remove(index_value)
    new_ids = []
    for key, value in same_index_dict.items():
        for index in value:
            find_flag = False
            for i, (label, dis) in enumerate(top_k_same_label_list[index]):
                if label != key and label not in new_ids:
                    find_flag = True
                    if dis > threshold:
                        pred_name[index] = 'unknown'
                        distances[index] = dis
                    else:
                        pred_name[index] = top_k_info_list[index][i][0]
                        distances[index] = top_k_info_list[index][i][1]
                        new_ids.append(pred_name[index].split('_')[2])
                    break
            if not find_flag:
                pred_name[index] = 'unknown'
                distances[index] = 0

    return pred_name, distances, top_k_info_list


def cosine_distance(embeddings1, embeddings2, axis=0):
    embeddings1 = embeddings1.flatten()
    embeddings2 = embeddings2.flatten()
    dot = np.dot(embeddings1, embeddings2)
    norm = np.linalg.norm(embeddings1, axis=axis) * np.linalg.norm(embeddings2, axis=axis)
    similarity = dot / norm
    dist = np.arccos(similarity) / math.pi
    return dist


def distance(embeddings1, embeddings2, distance_metric=0, axis=0):
    # distance_metric
    # == 0 --> Euclidian distance
    # == 1 --> Distance based on cosine similarity
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sqrt(np.sum(np.square(diff), axis))
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=axis)
        norm = np.linalg.norm(embeddings1, axis=axis) * np.linalg.norm(embeddings2, axis=axis)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric
    return dist


# Euclidean distance between two matrices
def euclidean_distances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED.transpose()


def compare_embedding(pred_emb, feature_dicts, threshold, group_name=None, top_k=7):
    """
    comparing predicted face embeddings with mysql face datasets
    :param pred_emb: face features to be compare
    :param feature_dicts: face feature database
    :param threshold: distance threshold between two face features
    :param group_name: the name of group
    :param top_k: represents the ORDER distances BY dis ASC
    :return: pred_name, distances, top_k_info_list
    """
    if group_name is None or group_name == "":
        dataset_array = feature_dicts
    else:
        dataset_array = fetch_by_group_name(feature_dicts, group_name)

    # 获取数据库中的入库时的图片名称  pic_names在数据库中存的是数组列索引4这个位置
    pic_names = dataset_array[:, 4]
    # 获取入库时图片对象的uid  pic_uid在数据库中存的是数组列索引2这个位置
    pic_uid = dataset_array[:, 2]

    face_features = np.asarray(np.vstack(dataset_array[:, 3]))
    distance_all = np.zeros((len(dataset_array), len(pred_emb)))
    distance_all[:] = euclidean_distances(pred_emb, face_features)

    distance_all = distance_all.transpose()
    # 获取距离最近的值
    # np.argsort() 返回排序后的索引
    pic_min_dis = np.amin(distance_all, axis=1)
    pic_min_uid = []
    user_min_names = []
    top_k_info_list = []
    for i in range(0, len(pic_min_dis)):
        temp_dist_list = copy(distance_all[i])
        temp_dist_list.sort()
        top_k_dis = temp_dist_list[:min(top_k, len(temp_dist_list))]
        top_k_info = [(pic_names[np.where(distance_all[i] == dis)[0][0]],
                       dis, np.where(distance_all[i] == dis)[0][0]) for dis in top_k_dis]
        top_k_info_list.append(top_k_info)

        # 获取最小值的index
        index = np.where(distance_all[i] == pic_min_dis[i])[0][0]
        # 有多个符合条件的只取第一个
        if pic_min_dis[i] > threshold:
            user_min_names.append('unknown')
            pic_min_uid.append("unknown")
        else:
            user_min_names.append(pic_names[index])
            pic_min_uid.append(pic_uid[index])

    # remove replicated labels
    id_counter = Counter(pic_min_uid)
    same_id_list = [id for id, num in id_counter.items() if num >= 2 and id != 'unknown']
    same_index_dict = collections.defaultdict(list)
    if len(same_id_list) > 0:
        for i, name in enumerate(pic_min_uid):
            if name in same_id_list:
                same_index_dict[name].append(i)
    for key, value in same_index_dict.items():
        dis_l = [pic_min_dis[v] for v in value]
        min_value = min(dis_l)
        index_value = np.where(pic_min_dis == min_value)[0][0]
        same_index_dict[key].remove(index_value)
    new_ids = []
    for key, value in same_index_dict.items():
        for index in value:
            find_flag = False
            for label, dis, ind in top_k_info_list[index]:
                if pic_uid[ind] != key and pic_uid[ind] not in new_ids:
                    find_flag = True
                    if dis > threshold:
                        user_min_names[index] = 'unknown'
                        pic_min_dis[index] = dis
                    else:
                        user_min_names[index] = label
                        pic_min_dis[index] = dis
                        new_ids.append(pic_uid[ind])
                    break
            if not find_flag:
                user_min_names[index] = 'unknown'
                pic_min_dis[index] = 0

    return user_min_names, pic_min_dis, top_k_info_list


def fetch_by_group_name(feature_dicts, group_name):
    try:
        target_indices = feature_dicts[:, 0] == group_name
        dataset_array = feature_dicts[target_indices]
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}'.format(e)
        print(errorMessage)
        raise errorMessage
    return dataset_array


def read_image(img_path, **kwargs):
    mode = kwargs.get('mode', 'rgb')
    layout = kwargs.get('layout', 'HWC')
    if mode == 'gray':
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
        if mode == 'rgb':
            # print('to rgb')
            img = img[..., ::-1]
        if layout == 'CHW':
            img = np.transpose(img, (2, 0, 1))
    return img


# def preprocess(img, bbox=None, landmark=None, **kwargs):
#     if isinstance(img, str):
#         img = read_image(img, **kwargs)
#     M = None
#     image_size = []
#     str_image_size = kwargs.get('image_size', '')
#     if len(str_image_size) > 0:
#         image_size = [int(x) for x in str_image_size.split(',')]
#         if len(image_size) == 1:
#             image_size = [image_size[0], image_size[0]]
#         assert len(image_size) == 2
#         assert image_size[0] == 112
#         assert image_size[0] == 112 or image_size[1] == 96
#     if landmark is not None:
#         assert len(image_size) == 2
#         src = np.array([
#             [30.2946, 51.6963],
#             [65.5318, 51.5014],
#             [48.0252, 71.7366],
#             [33.5493, 92.3655],
#             [62.7299, 92.2041]], dtype=np.float32)
#         if image_size[1] == 112:
#             src[:, 0] += 8.0
#         dst = landmark.astype(np.float32)
#
#         tform = trans.SimilarityTransform()
#         tform.estimate(dst, src)
#         M = tform.params[0:2, :]
#         # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)
#
#     if M is None:
#         if bbox is None:  # use center crop
#             det = np.zeros(4, dtype=np.int32)
#             det[0] = int(img.shape[1] * 0.0625)
#             det[1] = int(img.shape[0] * 0.0625)
#             det[2] = img.shape[1] - det[0]
#             det[3] = img.shape[0] - det[1]
#         else:
#             det = bbox
#         margin = kwargs.get('margin', 44)
#         bb = np.zeros(5, dtype=np.float32)
#         bb[0] = np.maximum(det[0] - margin / 2, 0)
#         bb[1] = np.maximum(det[1] - margin / 2, 0)
#         bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
#         bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
#         bb[4] = 0
#         ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
#         if len(image_size) > 0:
#             ret = cv2.resize(ret, (image_size[1], image_size[0]))
#         return ret, bb
#     else:  # do align using landmark
#         assert len(image_size) == 2
#
#         # src = src[0:3,:]
#         # dst = dst[0:3,:]
#
#         # print(src.shape, dst.shape)
#         # print(src)
#         # print(dst)
#         # print(M)
#         if bbox is None:  # use center crop
#             det = np.zeros(4, dtype=np.int32)
#             det[0] = int(img.shape[1] * 0.0625)
#             det[1] = int(img.shape[0] * 0.0625)
#             det[2] = img.shape[1] - det[0]
#             det[3] = img.shape[0] - det[1]
#         else:
#             det = bbox
#         margin = kwargs.get('margin', 44)
#         bb = np.zeros(5, dtype=np.float32)
#         bb[0] = np.maximum(det[0] - margin / 2, 0)
#         bb[1] = np.maximum(det[1] - margin / 2, 0)
#         bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
#         bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
#         bb[4] = 0
#         warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
#
#         # tform3 = trans.ProjectiveTransform()
#         # tform3.estimate(src, dst)
#         # warped = trans.warp(img, tform3, output_shape=_shape)
#         return warped, bb
