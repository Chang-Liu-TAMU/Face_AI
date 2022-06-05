# @Time: 2022/5/30 10:53
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:face_main.py

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from typing import Dict, List
import sys
import os.path as osp

dir_path = osp.dirname(osp.dirname(osp.realpath(__file__)))
sys.path.append(dir_path)

from core.face_machine import FaceMachine
from core.help_utils.manager_utils import Logger


class FaceAI(object):
    def __init__(self, configs):
        self.configs = configs
        self.logger = Logger(log_file=configs.log_file, log_name='FaceAI').logger
        self.logger.setLevel(self.configs.log_level)
        self._face_machine = FaceMachine(configs=configs, logger=self.logger)
        if self._face_machine is None:
            self.logger.info("Environment Information:\n" + self.configs.collect_env_info())
            text = "the face machine is None"
            self.logger.warning(text)
            raise ValueError(text)

    def recognize_faces(self, data, detect_angle=False, det_threshold=None, sim_threshold=None,
                        max_size=None, margin=None, group_name=None, extract_ga=False, limit_faces=0):
        return self._face_machine.recognize_faces(data=data,
                                                  detect_angle=detect_angle,
                                                  det_threshold=det_threshold,
                                                  sim_threshold=sim_threshold,
                                                  max_size=max_size, margin=margin,
                                                  group_name=group_name, extract_ga=extract_ga,
                                                  limit_faces=limit_faces)

    async def recognize_faces_async(self, data, detect_angle=False, max_size=None, det_threshold=None,
                                    sim_threshold=None, margin=None, group_name=None, extract_ga=False, limit_faces=0):
        return self.recognize_faces(data=data,
                                    detect_angle=detect_angle,
                                    det_threshold=det_threshold,
                                    sim_threshold=sim_threshold,
                                    max_size=max_size, margin=margin,
                                    group_name=group_name, extract_ga=extract_ga,
                                    limit_faces=limit_faces)

    def verify_faces(self, data=None, user_ids=None, detect_angle=False, det_threshold=None, sim_threshold=None,
                     max_size=None, margin=None, extract_ga=False, min_face_ratio=None, limit_faces=0):
        return self._face_machine.verify_faces(data=data, user_ids=user_ids, detect_angle=detect_angle,
                                               det_threshold=det_threshold, sim_threshold=sim_threshold,
                                               max_size=max_size, margin=margin, extract_ga=extract_ga,
                                               min_face_ratio=min_face_ratio, limit_faces=limit_faces)

    async def verify_faces_async(self, data=None, user_ids=None, detect_angle=False, det_threshold=None,
                                 sim_threshold=None, max_size=None, margin=None, extract_ga=False,
                                 min_face_ratio=None, limit_faces=0):
        return self.verify_faces(data=data, user_ids=user_ids, detect_angle=detect_angle,
                                 det_threshold=det_threshold, sim_threshold=sim_threshold,
                                 max_size=max_size, margin=margin, extract_ga=extract_ga,
                                 min_face_ratio=min_face_ratio, limit_faces=limit_faces)

    def register_faces(self, operator, data, u_groups, u_names, u_ids,
                       detect_angle=False, det_threshold=None, max_size=None, margin=None,
                       min_face_ratio=None, limit_faces=0):
        return self._face_machine.register_faces(operator=operator, data=data, u_groups=u_groups, u_names=u_names,
                                                 u_ids=u_ids, detect_angle=detect_angle, det_threshold=det_threshold,
                                                 max_size=max_size, margin=margin, min_face_ratio=min_face_ratio,
                                                 limit_faces=limit_faces)

    async def register_faces_async(self, operator, data, u_groups, u_names, u_ids,
                                   detect_angle=False, det_threshold=None, max_size=None, margin=None,
                                   min_face_ratio=None, limit_faces=0):
        return self.register_faces(operator=operator, data=data, u_groups=u_groups, u_names=u_names,
                                   u_ids=u_ids, detect_angle=detect_angle, det_threshold=det_threshold,
                                   max_size=max_size, margin=margin, min_face_ratio=min_face_ratio,
                                   limit_faces=limit_faces)

    def delete_faces(self, u_ids):
        return self._face_machine.delete_faces(u_ids)

    async def delete_faces_async(self, u_ids):
        return self.delete_faces(u_ids)

    def find_faces(self, u_ids, return_embedding=False):
        return self._face_machine.find_faces(u_ids=u_ids, return_embedding=return_embedding)

    async def find_faces_async(self, u_ids, return_embedding=False):
        return self.find_faces(u_ids=u_ids, return_embedding=return_embedding)

    def get_faces_info(self, data, max_size=None, det_threshold=None, limit_faces=0, embed_only=False,
                       detect_angle=False, extract_embedding=True, extract_ga=False,
                       return_face_data=False, return_landmarks=False, api_ver="1"):
        return self._face_machine.get_faces_info(data, max_size=max_size,
                                                 det_threshold=det_threshold,
                                                 limit_faces=limit_faces,
                                                 embed_only=embed_only,
                                                 detect_angle=detect_angle,
                                                 extract_embedding=extract_embedding,
                                                 extract_ga=extract_ga,
                                                 return_face_data=return_face_data,
                                                 return_landmarks=return_landmarks,
                                                 api_ver=api_ver)

    async def get_faces_info_async(self, data, max_size=None, det_threshold=None, limit_faces=0, embed_only=False,
                                   detect_angle=False, extract_embedding=True, extract_ga=False,
                                   return_face_data=False, return_landmarks=False, api_ver="1"):
        return await self._face_machine.get_faces_info_async(data, max_size=max_size,
                                                             det_threshold=det_threshold,
                                                             limit_faces=limit_faces,
                                                             embed_only=embed_only,
                                                             detect_angle=detect_angle,
                                                             extract_embedding=extract_embedding,
                                                             extract_ga=extract_ga,
                                                             return_face_data=return_face_data,
                                                             return_landmarks=return_landmarks,
                                                             api_ver=api_ver)

    def extract(self, images: Dict[str, list], max_size: List[int] = None, det_threshold: float = 0.6,
                limit_faces: int = 0, embed_only: bool = False, detect_angle: bool = False,
                return_face_data: bool = False, return_landmarks: bool = False, extract_embedding: bool = True,
                extract_ga: bool = True, verbose_timings=True, api_ver: str = "1"):
        return self._face_machine.extract(images=images, max_size=max_size, det_threshold=det_threshold,
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
                            extract_embedding: bool = True, extract_ga: bool = True, verbose_timings=True,
                            api_ver: str = "1"):
        return await self._face_machine.extract_async(images=images, max_size=max_size, det_threshold=det_threshold,
                                                      limit_faces=limit_faces, embed_only=embed_only,
                                                      detect_angle=detect_angle,
                                                      return_face_data=return_face_data,
                                                      return_landmarks=return_landmarks,
                                                      extract_embedding=extract_embedding,
                                                      extract_ga=extract_ga,
                                                      verbose_timings=verbose_timings, api_ver=api_ver)

    def draw(self, images, det_threshold=0.6, detect_angle: bool = False, draw_landmarks=True,
             draw_scores=True, draw_sizes=True, limit_faces=0, multipart=False):
        return self._face_machine.draw_images(images,
                                              det_threshold=det_threshold,
                                              detect_angle=detect_angle,
                                              draw_landmarks=draw_landmarks,
                                              draw_scores=draw_scores,
                                              draw_sizes=draw_sizes,
                                              limit_faces=limit_faces,
                                              multipart=multipart)

    async def draw_async(self, images, det_threshold=0.6, detect_angle: bool = False, draw_landmarks=True,
                         draw_scores=True, draw_sizes=True, limit_faces=0, multipart=False):
        return await self._face_machine.draw_images_async(images,
                                                          det_threshold=det_threshold,
                                                          detect_angle=detect_angle,
                                                          draw_landmarks=draw_landmarks,
                                                          draw_scores=draw_scores,
                                                          draw_sizes=draw_sizes,
                                                          limit_faces=limit_faces,
                                                          multipart=multipart)

    async def draw_recognitions_async(self, images, det_threshold: float = 0.6,
                                      sim_threshold: float = 0.5, detect_angle: bool = True,
                                      draw_landmarks: bool = True, draw_scores: bool = True,
                                      draw_age_gender: bool = True, margin: int = 44, limit_faces=0,
                                      text_color=(0, 0, 255), bbox_color=(0, 255, 0), multipart=False, group_name=None):
        return await self._face_machine.draw_recognitions_async(images,
                                                                det_threshold=det_threshold,
                                                                sim_threshold=sim_threshold,
                                                                detect_angle=detect_angle,
                                                                draw_landmarks=draw_landmarks,
                                                                draw_scores=draw_scores,
                                                                draw_age_gender=draw_age_gender,
                                                                margin=margin,
                                                                limit_faces=limit_faces,
                                                                text_color=text_color,
                                                                bbox_color=bbox_color,
                                                                multipart=multipart,
                                                                group_name=group_name)
