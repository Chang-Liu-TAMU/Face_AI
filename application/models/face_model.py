# @Time: 2022/5/30 11:10
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:face_model.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
token model
"""

import pydantic
from pydantic import BaseModel
from typing import Optional, List, Union
from core.env_parser import EnvConfigs

configs = EnvConfigs()


class Images(BaseModel):
    data: Optional[List[str]] = pydantic.Field(default=None,
                                               example=None,
                                               description='List of base64 encoded images')
    urls: Optional[List[str]] = pydantic.Field(default=None,
                                               example=[f"{configs.defaults.root_path}/test/face_images/test2.jpg"],
                                               description='List of images urls')


class BodyExtract(BaseModel):
    images: Images
    max_size: Optional[Union[List[int], str]] = pydantic.Field(default=configs.defaults.max_size,
                                                               example=configs.defaults.max_size,
                                                               description='Resize all images to this proportions')

    det_threshold: Optional[float] = pydantic.Field(default=configs.defaults.det_threshold,
                                                    example=configs.defaults.det_threshold,
                                                    description='Detector det_threshold [0, 1]')

    detect_angle: Optional[bool] = pydantic.Field(default=configs.defaults.detect_angle,
                                                  example=configs.defaults.detect_angle,
                                                  description='Apply angle detection for input images')

    embed_only: Optional[bool] = pydantic.Field(default=False,
                                                example=False,
                                                description='Treat input images as face crops and omit detection step')

    return_face_data: Optional[bool] = pydantic.Field(default=configs.defaults.return_face_data,
                                                      example=configs.defaults.return_face_data,
                                                      description='Return face crops encoded in base64')

    return_landmarks: Optional[bool] = pydantic.Field(default=configs.defaults.return_landmarks,
                                                      example=configs.defaults.return_landmarks,
                                                      description='Return face landmarks')

    extract_embedding: Optional[bool] = pydantic.Field(default=configs.defaults.extract_embedding,
                                                       example=configs.defaults.extract_embedding,
                                                       description='Extract face embeddings (otherwise only detect \
                                                       faces)')

    extract_ga: Optional[bool] = pydantic.Field(default=configs.defaults.extract_ga,
                                                example=configs.defaults.extract_ga,
                                                description='Extract gender/age')

    limit_faces: Optional[int] = pydantic.Field(default=0,
                                                example=0,
                                                description='Maximum number of faces to be processed')

    min_face_size: Optional[int] = pydantic.Field(default=0,
                                                  example=0,
                                                  description='Ignore faces smaller than this size')

    verbose_timings: Optional[bool] = pydantic.Field(default=False,
                                                     example=False,
                                                     description='Return all timings.')

    api_ver: Optional[str] = pydantic.Field(default=configs.defaults.api_ver,
                                            example='2',
                                            description='Output data serialization format.')


class BodyDraw(BaseModel):
    images: Images

    det_threshold: Optional[float] = pydantic.Field(default=configs.defaults.det_threshold,
                                                    example=configs.defaults.det_threshold,
                                                    description='Detector det_threshold [0, 1]')

    sim_threshold: Optional[float] = pydantic.Field(default=configs.defaults.face_sim_threshold,
                                                    example=configs.defaults.face_sim_threshold,
                                                    description='Recognizer face distance threshold '
                                                                '[0.5, 1.5] recommended')

    detect_angle: Optional[bool] = pydantic.Field(default=configs.defaults.detect_angle,
                                                  example=configs.defaults.detect_angle,
                                                  description='Apply angle detection for input images')

    draw_landmarks: Optional[bool] = pydantic.Field(default=True,
                                                    example=True,
                                                    description='Return face landmarks')

    draw_scores: Optional[bool] = pydantic.Field(default=True,
                                                 example=True,
                                                 description='Draw detection scores')

    draw_sizes: Optional[bool] = pydantic.Field(default=True,
                                                example=True,
                                                description='Draw face sizes')

    draw_age_gender: Optional[bool] = pydantic.Field(default=True,
                                                     example=True,
                                                     description='Draw ages and genders')

    text_color: Optional[Union[List[int], str]] = pydantic.Field(default=(255, 0, 0),
                                                                 example=(255, 0, 0),
                                                                 description='Draw text color')

    bbox_color: Optional[Union[List[int], str]] = pydantic.Field(default=(0, 255, 0),
                                                                 example=(0, 255, 0),
                                                                 description='Draw bounding box color')

    margin: Optional[int] = pydantic.Field(default=44,
                                           example=44,
                                           description='bounding box margin')

    limit_faces: Optional[int] = pydantic.Field(default=0,
                                                example=0,
                                                description='Maximum number of faces to be processed')

    min_face_size: Optional[int] = pydantic.Field(default=0,
                                                  example=0,
                                                  description='Ignore faces smaller than this size')


class FaceExtract(BaseModel):
    images: Images

    det_threshold: Optional[float] = pydantic.Field(default=configs.defaults.det_threshold,
                                                    example=configs.defaults.det_threshold,
                                                    description='Detector det_threshold [0, 1] recommended')

    sim_threshold: Optional[float] = pydantic.Field(default=configs.defaults.face_sim_threshold,
                                                    example=configs.defaults.face_sim_threshold,
                                                    description='Recognizer face distance threshold '
                                                                '[0.4, 0.8] recommended')

    min_face_ratio: Optional[float] = pydantic.Field(default=configs.defaults.min_face_ratio,
                                                     example=configs.defaults.min_face_ratio,
                                                     description='The minimum face ratio in a whole image')

    detect_angle: Optional[bool] = pydantic.Field(default=configs.defaults.detect_angle,
                                                  example=configs.defaults.detect_angle,
                                                  description='Apply angle detection for input images')

    extract_ga: Optional[bool] = pydantic.Field(default=configs.defaults.extract_ga,
                                                example=configs.defaults.extract_ga,
                                                description='Extract gender/age')

    max_size: Optional[Union[List[int], str]] = pydantic.Field(default=configs.defaults.max_size,
                                                               example=configs.defaults.max_size,
                                                               description='Resize all images to this proportions')

    margin: Optional[int] = pydantic.Field(default=configs.defaults.margin,
                                           example=configs.defaults.margin,
                                           description='Return bounding box with margin')

    limit_faces: Optional[int] = pydantic.Field(default=0,
                                                example=0,
                                                description='Maximum number of faces to be processed')

    user_groups: Optional[Union[List[str], str]] = pydantic.Field(default="",
                                                                  example="",
                                                                  description='The user group names.')

    user_ids: Optional[Union[List[str], str]] = pydantic.Field(default="",
                                                               example="",
                                                               description='The user ids.')

    compare: Optional[bool] = pydantic.Field(default=False,
                                             example=False,
                                             description='Compare two different faces.')

    api_ver: Optional[str] = pydantic.Field(default=configs.defaults.api_ver,
                                            example='2',
                                            description='Output data serialization format.')


class FaceRegister(BaseModel):
    operator: str
    user_ids: Union[List[str], str]
    images: Optional[Images] = None

    det_threshold: Optional[float] = pydantic.Field(default=configs.defaults.det_threshold,
                                                    example=configs.defaults.det_threshold,
                                                    description='Detector det_threshold [0, 1] recommended')

    min_face_ratio: Optional[float] = pydantic.Field(default=configs.defaults.min_face_ratio,
                                                     example=configs.defaults.min_face_ratio,
                                                     description='The minimum face ratio in a whole image')

    detect_angle: Optional[bool] = pydantic.Field(default=configs.defaults.detect_angle,
                                                  example=configs.defaults.detect_angle,
                                                  description='Apply angle detection for input images')

    return_embedding: Optional[bool] = pydantic.Field(default=False,
                                                      example=False,
                                                      description='Return face embedding')

    max_size: Optional[Union[List[int], str]] = pydantic.Field(default=configs.defaults.max_size,
                                                               example=configs.defaults.max_size,
                                                               description='Resize all images to this proportions')

    margin: Optional[int] = pydantic.Field(default=configs.defaults.margin,
                                           example=configs.defaults.margin,
                                           description='Return bounding box with margin')

    limit_faces: Optional[int] = pydantic.Field(default=0,
                                                example=0,
                                                description='Maximum number of faces to be processed')

    user_groups: Optional[Union[List[str], str]] = pydantic.Field(default="",
                                                                  example="",
                                                                  description='The user group names.')

    user_names: Optional[Union[List[str], str]] = pydantic.Field(default="",
                                                                 example="",
                                                                 description='The user names.')

    api_ver: Optional[str] = pydantic.Field(default=configs.defaults.api_ver,
                                            example='2',
                                            description='Output data serialization format.')