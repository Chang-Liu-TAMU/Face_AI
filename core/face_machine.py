# @Time: 2022/5/25 14:42
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:face_machine.py

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import io
import os
import traceback
import numpy as np
from typing import Dict, List
import logging

# import sys
# import os.path as osp
#
# dir_path = osp.dirname(osp.dirname(osp.realpath(__file__)))
# sys.path.append(dir_path)

from core.database import FaceMysql
from core.help_utils import tools, file_process, image_process
from core.help_utils.faces_cache import FaceFeatureCache
from core.help_utils.logger_utils import internal_logger


from modules.processing import Processing
from modules.utils.image_provider import get_images, get_images_async
from modules.utils.serialization import *


class FaceMachine:
    def __init__(self, configs, logger=None):
        self.configs = configs
        if logger is None:
            self.logger = internal_logger()
        else:
            self.logger = logger

        # self.logger = logging.getLogger("internal_logger")
        self._device = self.configs.models.device
        self._margin = self.configs.defaults.margin
        self._log_path = self.configs.log_path
        self._max_size = self.configs.defaults.max_size
        self._dis_metric = self.configs.defaults.dis_metric
        self._detect_angle = self.configs.defaults.detect_angle
        self._detect_threshold = self.configs.defaults.det_threshold
        self._staff_face_path = self.configs.defaults.staff_face_path
        # 60 / 180 * np.pi == 1.0472 --> cos(60°) = 0.5
        self._face_dis_threshold = self.configs.defaults.face_dis_threshold
        self._face_sim_threshold = self.configs.defaults.face_sim_threshold
        self._min_face_ratio = self.configs.defaults.min_face_ratio
        self._max_faces_per_uid = self.configs.defaults.max_faces_per_uid
        self._log_distance_top_k = self.configs.log_distance_top_k

        # build Face Processor
        self.face_model = Processing(det_name=self.configs.models.det_name,
                                        rec_name=self.configs.models.rec_name,
                                        ga_name=self.configs.models.ga_name,
                                        device=self.configs.models.device,
                                        max_size=self.configs.defaults.max_size,
                                        max_rec_batch_size=self.configs.models.rec_batch_size,
                                        max_det_batch_size=self.configs.models.det_batch_size,
                                        backend_name=self.configs.models.backend_name,
                                        force_fp16=self.configs.models.fp16,
                                        triton_uri=self.configs.models.triton_uri,
                                        root_dir=self.configs.defaults.models_path,
                                        logger=self.logger,
                                        download_model=self.configs.models.download_model)

        for key, val in self.face_model.models_info.items():
            if not val:
                logger.info("Environment Information:\n" + self.configs.collect_env_info())
                text = f"the {key} is None"
                logger.warning(text)
                raise ValueError(text)


        # database
        self._db = FaceMysql(host=self.configs.database.db_host,
                             port=self.configs.database.db_port,
                             username=self.configs.database.db_user,
                             password=self.configs.database.db_password,
                             database_name=self.configs.database.face_db_name,
                             table_name=self.configs.database.db_table)
        if self._db is None:
            self.logger.error("loading face feature database failed...")
            return

        # loading face data from mysql database
        self.logger.info("loading face feature database...")
        dataset = tools.parse_features(self._db.findall_facejson())
        self._faces_cache = FaceFeatureCache(data=dataset,
                                             per_uid_capacity=self._max_faces_per_uid,
                                             logger=self.logger)
        if self._faces_cache.is_valid():
            self.logger.info("loading face feature database finished...")

        self._dis_writer = open(os.path.join(self._log_path, "recognization_dis_log.txt"), mode='a')


    @staticmethod
    def _convert_to_dict(probability, result, status='ok', message=[]):
        return {'probability': probability, 'result': result, 'status': status, "message": message}

    def _update_cache(self, update_value_list, operator):
        # replace existed items
        if operator == 'replace':
            self._faces_cache.replace(update_value_list=update_value_list)
        else:  # add or update items
            self._faces_cache.put(update_value_list=update_value_list)

    def _remove_cache(self, uids):
        if not self._faces_cache.is_valid():
            self.logger.warning("[FaceMachine._remove_cache] database is Empty")
            return
        self._faces_cache.pop(uids=uids)

    #methods with problems
    def _align_faces(self, image_data, function_name, det_threshold=None,
                     max_size=None, margin=None, min_face_ratio=None, limit_faces=0):
        # use internal params if not given
        if det_threshold is None:
            det_threshold = self._detect_threshold
        if max_size is None:
            max_size = self._max_size
        if margin is None:
            margin = self._margin
        if min_face_ratio is None:
            min_face_ratio = self._min_face_ratio

        if image_data.get('traceback') is None:
            img = image_data.get('data')
            image_id = image_data.get('image_id', "None")
            bounding_boxes, points = self.face_model.detect_face(img=img,
                                                                 det_thresh=det_threshold,
                                                                 max_size=max_size,
                                                                 limit_faces=limit_faces)
            #???
            cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #???
            img_size = np.asarray(cv_img.shape)[0:2]

            nrof_faces = len(bounding_boxes)
            if nrof_faces < 1:
                text = "{} can't detect face , ignore ".format(function_name) + image_id
                self.logger.warning(text)
                return None, text
            elif nrof_faces > 1:
                text = '{} detect more than one face in '.format(function_name) + image_id
                self.logger.warning(text)
                bbs_area_list, img_area = tools.get_bbox_areas(bounding_boxes, img_size, margin=margin)
                ratio = max(bbs_area_list) / img_area
                if ratio < min_face_ratio:
                    text = '{} the area ratio of face in {} is too small --> {}, lower than {}'.format(
                        function_name, image_id, ratio, min_face_ratio)
                    self.logger.warning(text)
                    return None, text
                bounding_boxes = bounding_boxes[np.argmax(bbs_area_list)]
                points = points[np.argmax(bbs_area_list)]
                #??? have det_threshold at detection stage
                # also, bounding_boxes is a two-D array, index error???
                if bounding_boxes[-1] < det_threshold:
                    text = '{} the face score of {} is {}, lower than {}'.format(
                        function_name, image_id, bounding_boxes[-1], det_threshold)
                    self.logger.warning(text)
                    return None, text
                #???
                align_image, _ = self.face_model.prepocess(img=cv_img, det_arr=[bounding_boxes],
                                                           points=[points], margin=margin)
            else:
                #???error
                if bounding_boxes[0][-1] < det_threshold:
                    text = '{} the face score of {} is {}, lower than {}'.format(
                        function_name, image_id, bounding_boxes[0][-1], det_threshold)
                    self.logger.warning(text)
                    return None, text
                #???
                align_image, bbox = self.face_model.prepocess(img=cv_img, det_arr=bounding_boxes,
                                                              points=points, margin=margin)
                bbox = bbox[0]
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                ratio = bbox_area / (img_size[0] * img_size[1])
                if ratio < min_face_ratio:
                    text = '{} the area of face in {} is too small --> {}, lower than {}'.format(
                        function_name, image_id, ratio, min_face_ratio)
                    self.logger.warning(text)
                    return None, text
            return align_image[0], 'ok'
        else:
            return None, "[_align_faces] Parsing images failed!"


    def get_face_rois(self, images, detect_angle=False, det_threshold=None,
                      max_size=None, margin=None, min_face_ratio=None,
                      limit_faces=0):
        # use internal params if not given
        if det_threshold is None:
            det_threshold = self._detect_threshold
        if max_size is None:
            max_size = self._max_size
        if margin is None:
            margin = self._margin
        if min_face_ratio is None:
            min_face_ratio = self._min_face_ratio

        function_name = "[{}]-".format("get_face_rois")
        face_rois = {}
        info_list = []
        try:
            # if detect_angle:
            #     angles = self.face_model.detect_angle(images=images)
            # else:
            #     angles = [0 for i in range(len(images))]

            for index, image in enumerate(images):
                image_id = image.get('image_id', "None")
                if image_id == "None":
                    image_id += str(index)
                align_image, info = self._align_faces(image, function_name=function_name,
                                                      det_threshold=det_threshold,
                                                      max_size=max_size, margin=margin,
                                                      min_face_ratio=min_face_ratio,
                                                      limit_faces=limit_faces)
                if align_image is None:
                    info_list.append(info)
                    continue
                face_rois[image_id] = align_image
        except Exception as e:
            self.logger.exception(e)
            return face_rois, [str(e)]
        return face_rois, info_list




    def get_faces_info(self, data, max_size=None, det_threshold=None, limit_faces=0, embed_only=False,
                       detect_angle=False, extract_embedding=True, extract_ga=False,
                       return_face_data=False, return_landmarks=False, api_ver="1"):

        # use internal params if not given
        if det_threshold is None:
            det_threshold = self._detect_threshold
        if max_size is None:
            max_size = self._max_size

        self.logger.info("*" * 50 + " start getting face information " + "*" * 50)
        output = dict(took={}, data=[], status="ok")
        try:
            output = self.extract(images=data, max_size=max_size, det_threshold=det_threshold,
                                  limit_faces=limit_faces, embed_only=embed_only,
                                  detect_angle=detect_angle,
                                  return_face_data=return_face_data,
                                  return_landmarks=return_landmarks,
                                  extract_embedding=extract_embedding,
                                  extract_ga=extract_ga,
                                  verbose_timings=True, api_ver=api_ver)

        except Exception as e:
            self.logger.info("Environment Information:\n" + self.configs.collect_env_info())
            output["status"] = str(e)
        return Serializer().serialize(output, api_ver=api_ver)

    async def get_faces_info_async(self, data, max_size=None, det_threshold=None, limit_faces=0, embed_only=False,
                                   detect_angle=False, extract_embedding=True, extract_ga=False,
                                   return_face_data=False, return_landmarks=False, api_ver="1"):
        # use internal params if not given
        if det_threshold is None:
            det_threshold = self._detect_threshold
        if max_size is None:
            max_size = self._max_size

        self.logger.info("*" * 50 + " start getting face information " + "*" * 50)
        output = dict(took={}, data=[], status="ok")
        try:
            output = await self.extract_async(images=data, max_size=max_size, det_threshold=det_threshold,
                                              limit_faces=limit_faces, embed_only=embed_only,
                                              detect_angle=detect_angle,
                                              return_face_data=return_face_data,
                                              return_landmarks=return_landmarks,
                                              extract_embedding=extract_embedding,
                                              extract_ga=extract_ga,
                                              verbose_timings=True, api_ver=api_ver)
        except Exception as e:
            self.logger.info("Environment Information:\n" + self.configs.collect_env_info())
            output["status"] = str(e)
        return Serializer().serialize(output, api_ver=api_ver)


    def recognize_faces(self, data, detect_angle=False, det_threshold=None, sim_threshold=None,
                        max_size=None, margin=None, group_name=None, extract_ga=False, limit_faces=0):
        """
        this is a kind of face verifier
        :param data: represents the single image to be verified
        :param detect_angle:
        :param det_threshold:
        :param sim_threshold:
        :param max_size:
        :param margin:
        :param extract_ga:
        :param limit_faces:
        :param group_name: represents the name of the current group
        :return:
        """
        # use internal params if not given
        if det_threshold is None:
            det_threshold = self._detect_threshold

        if sim_threshold is None:
            dis_threshold = self._face_dis_threshold
        else:
            dis_threshold = tools.probability_to_dis_threshold(probability=sim_threshold, precision=4)

        if max_size is None:
            max_size = self._max_size
        if margin is None:
            margin = self._margin

        dict_list = []
        self.logger.info("*" * 50 + " start face recognition " + "*" * 50)

        try:
            if not self._faces_cache.is_valid():
                warning_message = "[FaceMachine.recognize_faces] database is Empty, please register first!"
                self.logger.warning(warning_message)
                return [warning_message]

            if group_name is not None and group_name != "":
                if group_name not in self._faces_cache.get_group_names():
                    self.logger.warning("group_name is invalid and will be ignored...")
                    return dict_list

            person_number_list = []
            images = get_images(data, logger=self.logger)
            if detect_angle:
                angles = self.face_model.detect_angle(images=images)
            else:
                angles = [0 for _ in range(len(images))]

            for image_data, angle in zip(images, angles):
                image_id = image_data.get("image_id", "None")
                json_dict = {}
                try:
                    if image_data.get('traceback') is None:
                        image = image_data.get('data')
                        # 获取 判断标识 bounding_box crop_image
                        bounding_boxes, points = self.face_model.detect_face(
                            img=image, det_thresh=det_threshold,
                            max_size=max_size, limit_faces=limit_faces)
                        box_probs, det, landmark_list = tools.face_filter(image, bounding_boxes, points,
                                                                          det_threshold, margin)
                        if bounding_boxes is None or landmark_list is None:
                            continue
                        nrof_faces = len(det)
                        assert nrof_faces == len(landmark_list) == len(box_probs)
                        if nrof_faces < 1:
                            self.logger.warning(
                                '{}{}'.format("[recognize_faces] system error or cannot find any face in ", image_id))
                            continue

                        person_number_list.append(nrof_faces)
                        if nrof_faces > 0:
                            det_arr = []
                            if nrof_faces > 1:
                                for i in range(nrof_faces):
                                    det_arr.append(np.squeeze(det[i]))
                            else:
                                det_arr.append(np.squeeze(det))
                            face_images, bbs = self.face_model.prepocess(image, det_arr, landmark_list, margin)
                        else:
                            self.logger.warning("[recognize_faces] cannot find any face in this {}".format(image_id))
                            continue
                        if len(face_images) > 0:
                            # detect faces
                            pred_name, distances, top_k_info_list = \
                                self.face_model.recognize_faces(face_images,
                                                                self._faces_cache.get_data(),
                                                                dis_threshold=dis_threshold,
                                                                group_name=group_name)

                            if extract_ga:
                                ages, genders = self.face_model.get_age_gender(face_images)
                            else:
                                ages, genders = None, None

                            pred_label_for_show = ['_'.join(name.split('_')[:3])
                                                   if name != 'unknown' else 'unknown' for name in pred_name]
                            json_dict = file_process.convert_to_json_with_ga(image_id, bbs, pred_label_for_show,
                                                                                distances, angle, ages, genders)
                            if self._log_distance_top_k:
                                file_process.log_distances_top_k(self._dis_writer, image_id, pred_name,
                                                                    top_k_info_list, box_probs)
                except Exception as e:
                    tb = traceback.format_exc()
                    self.logger.warning(tb)
                    json_dict = {}

                dict_list.append(json_dict)

            self.logger.info('match person numbers : {} \t rotate angle: {}'.format(person_number_list, angles))
        except Exception as e:
            self.logger.info("Environment Information:\n" + self.configs.collect_env_info())
            self.logger.exception(e)
        return dict_list


    def verify_faces(self, data=None, user_ids=None, detect_angle=False, det_threshold=None, sim_threshold=None,
                     max_size=None, margin=None, extract_ga=False, min_face_ratio=None, limit_faces=0):
        """
        this is a kind of face verifier
        :param data:
        :param detect_angle:
        :param det_threshold:
        :param sim_threshold:
        :param max_size:
        :param margin:
        :param extract_ga:
        :param min_face_ratio:
        :param limit_faces:
        :param user_ids: represents the id of the current person
        :return: result_dict: {'probability': List[probability], 'result': List[result["1", "-1"]]}
        """

        self.logger.info("*" * 50 + " start face verification " + "*" * 50)

        user_ids = tools.parse_data(user_ids)
        if len(user_ids) == 0:
            result_list = [0]
            probability_list = [-1]
        else:
            result_list = [0] * len(user_ids)
            probability_list = [-1] * len(user_ids)

        # use internal params if not given
        if det_threshold is None:
            det_threshold = self._detect_threshold
        if max_size is None:
            max_size = self._max_size
        if margin is None:
            margin = self._margin
        if min_face_ratio is None:
            min_face_ratio = self._min_face_ratio

        try:
            if data is None:
                text = '{}...{}:{}'.format('data is None', 'probability', probability_list)
                self.logger.error(text)
                return self._convert_to_dict(probability_list, result_list, text)
            images = get_images(data, logger=self.logger)
            if detect_angle:
                angles = self.face_model.detect_angle(images=images)
            else:
                angles = [0 for i in range(len(images))]
            face_images = []
            for image in images:
                face, info = self._align_faces(image, function_name="verify_faces",
                                               det_threshold=det_threshold,
                                               max_size=max_size, margin=margin,
                                               min_face_ratio=min_face_ratio,
                                               limit_faces=limit_faces)
                if face is None:
                    return self._convert_to_dict(probability_list, result_list, info)
                else:
                    face_images.append(face)
            target_embeddings = self.face_model.get_embedding(face_images)
            if extract_ga:
                ages, genders = self.face_model.get_age_gender(face_images)
            else:
                ages, genders = None, None

            min_dis_list = []
            best_match_label_list = []
            just_compare_two_faces = False
            if user_ids is not None and len(user_ids) > 0:
                if not self._faces_cache.is_valid():
                    warning_message = "[FaceMachine.verify_faces] database is Empty, please register first!"
                    self.logger.warning(warning_message)
                    return self._convert_to_dict(probability_list, result_list, warning_message)

                if len(user_ids) != target_embeddings.shape[0]:
                    warning_message = "[FaceMachine.verify_faces] user_ids do not match input images!"
                    self.logger.warning(warning_message)
                    return self._convert_to_dict(probability_list, result_list, warning_message)

                for index, user_id in enumerate(user_ids):
                    if self._faces_cache.contains(user_id):
                        emb_arrays = self._faces_cache.get_features(uid=user_id)
                        label_list = self._faces_cache.get_filenames(uid=user_id)
                    else:
                        text = 'KeyError:{} is not in {}'.format(user_id, 'dataset')
                        min_dis_list.append(-1)
                        best_match_label_list.append(text)
                        continue

                    if len(emb_arrays) == 0:
                        text = 'cannot match the id: {} in the {}'.format(user_id, 'face feature database')
                        min_dis_list.append(-1)
                        best_match_label_list.append(text)
                        continue

                    dis_list = [self.face_model.embedding_distance(
                        emb, target_embeddings[index], distance_metric=self._dis_metric) for emb in emb_arrays]
                    min_dis = min(dis_list)
                    min_dis_list.append(min_dis)
                    best_match_label_list.append(label_list[dis_list.index(min_dis)])
            else:
                assert len(target_embeddings) % 2 == 0
                result_list = [0] * (len(target_embeddings) // 2)
                for i in range(0, len(target_embeddings) - 1, 2):
                    just_compare_two_faces = True
                    dis_list = [self.face_model.embedding_distance(target_embeddings[i], target_embeddings[i + 1],
                                                                   distance_metric=self._dis_metric)]
                    min_dis = min(dis_list)
                    min_dis_list.append(min_dis)
                    best_match_label_list.append("None")

            if sim_threshold is None:
                sim_threshold = self._face_sim_threshold
            probability_list = [file_process.convert_to_similarity(min_dis)
                                if min_dis >= 0 else -1 for min_dis in min_dis_list]
            for index, probability in enumerate(probability_list):
                self.logger.info('{}\t{}: {}'.format(best_match_label_list[index],
                                                     'probability', probability))

                if sim_threshold < probability:
                    result = -1
                    # if just_compare_two_faces and len(genders) == 2:
                    #     # judge genders finally
                    #     if genders is not None and genders[0] != genders[1]:
                    #         result = 0
                    #     else:
                    #         result = -1
                    # else:
                    #     result = -1
                else:
                    result = 0

                result_list[index] = result
                probability_list[index] = probability
        except Exception as e:
            self.logger.info("Environment Information:\n" + self.configs.collect_env_info())
            self.logger.exception(e)
            return self._convert_to_dict(probability_list, result_list, str(e))
        return self._convert_to_dict(probability_list, result_list, message=best_match_label_list)



    def register_faces(self, operator, data, u_groups, u_names, u_ids,
                       detect_angle=False, det_threshold=None, max_size=None, margin=None,
                       min_face_ratio=None, limit_faces=0):
        request_result = {}
        self.logger.info("start registering faces...")

        # use internal params if not given
        if det_threshold is None:
            det_threshold = self._detect_threshold
        if max_size is None:
            max_size = self._max_size
        if margin is None:
            margin = self._margin
        if min_face_ratio is None:
            min_face_ratio = self._min_face_ratio

        try:
            images = get_images(data, logger=self.logger)
            # update image_id for each image if possible

            if len(u_groups) > 0 and len(u_names) > 0 and len(u_ids) > 0:
                assert len(u_groups) == len(u_names) == len(u_ids)
                u_groups = [u_group.strip() for u_group in u_groups]
                u_names = [u_name.strip() for u_name in u_names]
                u_ids = [u_id.strip() for u_id in u_ids]
                if len(images) != len(u_ids):
                    u_groups = [u_groups[0]] * len(images)
                    u_names = [u_names[0]] * len(images)
                    u_ids = [u_ids[0]] * len(images)

                for index, image in enumerate(images):
                    image_id = image.get("image_id", "None")
                    if image_id == "None":
                        image_id = "{}_{}_{}_{}".format(u_groups[index], u_names[index], u_ids[index],
                                                        tools.generate_unique_id())
                    else:
                        filename = image_id.split("_")
                        if len(filename) > 1:
                            filename = filename[-2:]
                            filename = '_'.join(filename)
                            image_id = "{}_{}_{}_{}".format(u_groups[index], u_names[index], u_ids[index], filename)
                        else:
                            image_id = "{}_{}_{}_{}".format(u_groups[index], u_names[index], u_ids[index], image_id)
                    image.update(image_id=image_id)
            else:
                u_groups = ["None"] * len(images)
                u_names = ["None"] * len(images)
                u_ids = ["None"] * len(images)
            img_num = len(images)
            # obtain face rois

            #debug
            print("calling get_face_rois")
            face_rois_dict, info_list = self.get_face_rois(images=images,
                                                           detect_angle=detect_angle,
                                                           det_threshold=det_threshold,
                                                           max_size=max_size, margin=margin,
                                                           min_face_ratio=min_face_ratio,
                                                           limit_faces=limit_faces)
            if len(face_rois_dict) < 1:
                request_result['status'] = 'unknown error' if len(info_list) == 0 else info_list[-1]
                request_result['ignore'] = '0'
                request_result['id'] = '-1'
                return request_result



            face_features = self.face_model.get_embedding(image_crops=list(face_rois_dict.values()),
                                                          use_normalization=True)
            feature_num = len(face_features)
            pic_names_list = list(face_rois_dict.keys())
            assert len(pic_names_list) == feature_num
            if img_num != feature_num:
                self.logger.warning("ignore number: {}".format(img_num - feature_num))

            # update mysql database
            try:
                labels = list(map(lambda x: x.split("_")[:3], pic_names_list))
                ugroups, unames, uids = list(zip(*labels))
            except Exception as e:
                labels = list(zip(u_groups, u_names, u_ids))
                ugroups, unames, uids = u_groups, u_names, u_ids

            update_value_list = FaceFeatureCache.parse_input(pic_names_list=pic_names_list,
                                                             face_feature_list=face_features,
                                                             label_list=labels)
            self._update_cache(update_value_list, operator)
            updated_faces_info = self._faces_cache.find(u_ids=u_ids, return_embedding=True)
            ignore_ids = []
            if len(updated_faces_info) > 0:
                ugroups = []
                unames = []
                uids = []
                pic_names_list = []
                feature_vectors = []
                for uid, info in updated_faces_info.items():
                    face_info = info.get("match_items", [])
                    if len(face_info) == 0:
                        ignore_ids.append(uid)
                        continue
                    for face in face_info:
                        ugroups.append(face[0])
                        unames.append(face[1])
                        uids.append(face[2])
                        feature_vectors.append(",".join(list(map(str, face[3]))))
                        pic_names_list.append(face[4])

                last_id = self._db.update_facejson(pic_names=pic_names_list,
                                                   features=feature_vectors,
                                                   unames=unames, uids=uids,
                                                   ugroups=ugroups)
            else:
                last_id = "Cannot update faces!"

            if isinstance(last_id, str):
                request_result['status'] = last_id
                request_result['ignore'] = '0'
                request_result['id'] = '-1'
                if len(ugroups) > 0:
                    self.logger.warning(
                        "registering staff or leader: group={}, name={}, id={} failed! the reason is : {}".
                            format(ugroups[0], unames[0], uids[0], request_result['status']))
                else:
                    self.logger.warning(request_result['status'])
            else:
                if len(ignore_ids) == 0:
                    request_result['status'] = 'ok'
                else:
                    request_result['status'] = "Ignore ids: ".format(ignore_ids)
                request_result['ignore'] = str(img_num - feature_num)
                request_result['id'] = str(last_id)

            return request_result

        except Exception as e:
            self.logger.info("Environment Information:\n" + self.configs.collect_env_info())
            self.logger.exception(e)
            request_result['status'] = str(e)
            request_result['ignore'] = '0'
            request_result['id'] = '-1'
            return request_result


    def delete_faces(self, u_ids):
        try:
            if u_ids is None:
                return {'status': 'no uids input', 'ignore': '0', 'success': '0', 'detail': '0'}

            if isinstance(u_ids, str):
                if "," in u_ids:
                    u_ids = u_ids.split(",")
                else:
                    u_ids = [u_ids]
            if isinstance(u_ids, (tuple, list)):
                if len(u_ids) == 0:
                    return {'status': 'no uids input', 'ignore': '0', 'success': '0', 'detail': '0'}
                u_ids = list(u_ids)
            u_ids = [uid.strip() for uid in u_ids]
            res = self._db.remove_faces_by_ids(u_ids)

            if isinstance(res, str) or len(res) == 0:
                self.logger.info("errors: {}  --> deleting face items with id{} failed!".format(res, u_ids))
                return {'status': res, 'ignore': '0', 'success': '0', 'detail': '0'}
            else:
                self.logger.info("[delete_faces] detail: {}!".format(res))
                self._remove_cache(u_ids)
                ignore_items = [status for status in res.values() if status == 0]
                detail = file_process.json_wrapper(res)
                return {'status': 'ok', 'ignore': str(len(ignore_items)),
                        'success': str(len(res) - len(ignore_items)), 'detail': detail}

        except Exception as e:
            request_result = {}
            self.logger.info("Environment Information:\n" + self.configs.collect_env_info())
            self.logger.exception(e)
            request_result['status'] = str(e)
            request_result['ignore'] = '0'
            request_result['success_number'] = '-1'
            return request_result



    def find_faces(self, u_ids, return_embedding=False):
        try:
            u_ids = tools.parse_data(data=u_ids)
            if len(u_ids) == 0:
                return {'status': 'no uids input', 'ignore': '0', 'success': '0', 'detail': '0'}
            res = self._faces_cache.find(u_ids=u_ids, return_embedding=return_embedding)
            # res = self._db.find_faces_by_ids(u_ids=u_ids, return_embedding=return_embedding)
            for id, items in res.items():
                self.logger.info("[find_faces] uid: {} --> match_items_number: {}".format(id, items["match_number"]))
            return {'status': 'ok', 'find_result': res}

        except Exception as e:
            request_result = {}
            self.logger.info("Environment Information:\n" + self.configs.collect_env_info())
            self.logger.exception(e)
            request_result['status'] = str(e)
            request_result['find_result'] = '-1'
            return request_result


    def extract(self, images: Dict[str, list], max_size: List[int] = None, det_threshold: float = 0.6,
                limit_faces: int = 0, embed_only: bool = False, detect_angle: bool = False,
                return_face_data: bool = False, return_landmarks: bool = False, extract_embedding: bool = True,
                extract_ga: bool = True, verbose_timings=True, api_ver: str = "1"):
        return self.face_model.extract(images=images, max_size=max_size, threshold=det_threshold,
                                       limit_faces=limit_faces, embed_only=embed_only,
                                       detect_angle=detect_angle,
                                       return_face_data=return_face_data,
                                       return_landmarks=return_landmarks,
                                       extract_embedding=extract_embedding,
                                       extract_ga=extract_ga,
                                       verbose_timings=verbose_timings, api_ver=api_ver)


    async def extract_async(self, images: Dict[str, list], max_size: List[int] = None, det_threshold: float = 0.6,
                            limit_faces: int = 0, embed_only: bool = False, detect_angle: bool = False,
                            return_face_data: bool = False, return_landmarks: bool = False,
                            extract_embedding: bool = True, extract_ga: bool = True,
                            verbose_timings=True, api_ver: str = "1"):
        return await self.face_model.extract_async(images=images, max_size=max_size, threshold=det_threshold,
                                                   limit_faces=limit_faces, embed_only=embed_only,
                                                   detect_angle=detect_angle,
                                                   return_face_data=return_face_data,
                                                   return_landmarks=return_landmarks,
                                                   extract_embedding=extract_embedding,
                                                   extract_ga=extract_ga,
                                                   verbose_timings=verbose_timings, api_ver=api_ver)


    def draw_images(self, images, det_threshold=0.6, detect_angle=False, draw_landmarks=True,
                    draw_scores=True, draw_sizes=True, limit_faces=0, multipart=False):
        return self.face_model.draw(images, threshold=det_threshold,
                                    detect_angle=detect_angle,
                                    draw_landmarks=draw_landmarks,
                                    draw_scores=draw_scores, draw_sizes=draw_sizes,
                                    limit_faces=limit_faces, multipart=multipart)

    async def draw_images_async(self, images, det_threshold=0.6, detect_angle=False, draw_landmarks=True,
                                draw_scores=True, draw_sizes=True, limit_faces=0, multipart=False):
        return await self.face_model.draw_async(images, threshold=det_threshold,
                                                detect_angle=detect_angle,
                                                draw_landmarks=draw_landmarks,
                                                draw_scores=draw_scores, draw_sizes=draw_sizes,
                                                limit_faces=limit_faces, multipart=multipart)

    async def draw_recognitions_async(self, images, det_threshold: float = 0.6,
                                      sim_threshold: float = 0.5, detect_angle: bool = True,
                                      draw_landmarks: bool = True, draw_scores: bool = True,
                                      draw_age_gender: bool = True, margin: int = 44, limit_faces=0,
                                      text_color=(0, 0, 255), bbox_color=(0, 255, 0), multipart=False, group_name=None):
        if multipart:
            images = await image_process.files_wrapper_async([images])

        dict_list = self.recognize_faces(data=images, detect_angle=detect_angle, det_threshold=det_threshold,
                                         sim_threshold=sim_threshold, margin=margin, group_name=group_name,
                                         extract_ga=draw_age_gender,
                                         limit_faces=limit_faces)

        if len(dict_list) > 0 and not isinstance(dict_list[0], str):
            data = await get_images_async(images, logger=self.logger)
            data = data[0].get('data')
            image = image_process.visualize_img(data=data, detections_dict=dict_list[0],
                                                   draw_landmarks=draw_landmarks,
                                                   draw_scores=draw_scores,
                                                   draw_age_gender=draw_age_gender,
                                                   only_bbox=False, result_file_path=None,
                                                   text_color=text_color, bbox_color=bbox_color)
            is_success, buffer = cv2.imencode(".jpg", image)
            io_buf = io.BytesIO(buffer)
        else:
            io_buf = io.BytesIO()
        return io_buf








