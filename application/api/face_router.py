# @Time: 2022/5/30 11:08
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:face_router.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/29 15:43
# @Author  : CoderCharm
# @File    : sys_api.py
# @Software: PyCharm
# @Github  : github/CoderCharm
# @Email   : wg_python@163.com
# @Desc    :
"""
Face router
"""

import os
from typing import Optional, List
from fastapi import APIRouter
from fastapi import File, Form, UploadFile
from fastapi.responses import UJSONResponse
from fastapi.encoders import jsonable_encoder
from starlette.responses import StreamingResponse


from core.help_utils import tools
from application.models.face_model import BodyExtract, BodyDraw, FaceRegister, FaceExtract
from core import FaceAI
from core.env_parser import EnvConfigs
from core.help_utils import image_process

router = APIRouter()

# Read runtime settings from environment variables
configs = EnvConfigs()

# init face ai object
face_ai = FaceAI(configs=configs)


def response_wrapper(output):
    return UJSONResponse(output)


@router.post('/face_registration', tags=['FaceAI'])
async def face_registration(data: FaceRegister):
    """
    Face registration endpoint accept json with
    parameters in following format:
       - **images**: dict containing either links or data lists. (*optional*)
       - **user_ids**: The user ids. (*required*)
       - **operator**: Supported operator: ["add", "delete", "replace", "find"]. (*required*)
       - **user_groups**: The user groups name. Default: [""] (*optional*)
       - **user_names**: The user names. Default: [""] (*optional*)
       - **return_embedding**: Return face embedding. Default: True (*optional*)
       - **detect_angle**: Apply angle detection for input images. Default: False (*optional*)
       - **det_threshold**: Detection det_threshold. Default: 0.6  [0, 1] recommended (*optional*)
       - **min_face_ratio**: The minimum face ratio in a whole image. Default: 0.15 (*optional*)
       - **extract_ga**: Extract gender/age. Default: False (*optional*)
       - **max_size**: Resize all images to this proportions. Default: '960,960' (*optional*)
       - **margin**: Return bounding box with margin. Default: 44 (*optional*)
       - **limit_faces**: Maximum number of faces to be processed. 0 for unlimited number. Default: 0 (*optional*)
       \f

       :return:
       List[dict]
    """
    if data.operator not in configs.support_operators:
        output = {'status': 'error', "message": "operator '{0}' has not been supported!".format(data.operator)}
        return response_wrapper(output)

    user_groups = tools.parse_data(data.user_groups)
    user_names = tools.parse_data(data.user_names)
    user_ids = tools.parse_data(data.user_ids)

    if data.operator == 'delete':
        if len(user_ids) == 0:
            output = {'status': 'error', "message": "user_ids must be set in delete mode"}
        else:
            output = await face_ai.delete_faces_async(user_ids)
    elif data.operator == 'find':
        if len(user_ids) == 0:
            output = {'status': 'error', "message": "user_ids must be set in find mode"}
        else:
            output = await face_ai.find_faces_async(u_ids=user_ids, return_embedding=data.return_embedding)
    else:  # add or replace
        if len(user_groups) == 0 or len(user_names) == 0 or len(user_ids) == 0:
            output = {'status': 'error',
                      "message": "user_groups, user_names and user_ids must be set in {} mode".format(data.operator)}
        else:
            if not (len(user_groups) == len(user_names) == len(user_ids)):
                output = {'status': 'error',
                          "message": "user_groups, user_names and user_ids must be set in same dimension"}
            else:
                if data.images is None:
                    output = {'status': 'error',
                              "message": "files must be set in {} mode".format(data.operator)}
                else:
                    images = jsonable_encoder(data.images)
                    if isinstance(data.max_size, str):
                        data.max_size = tools.parse_size(size=data.max_size, def_size=configs.defaults.max_size_str)
                    output = await face_ai.register_faces_async(operator=data.operator,
                                                                data=images,
                                                                u_groups=user_groups,
                                                                u_names=user_names,
                                                                u_ids=user_ids,
                                                                detect_angle=data.detect_angle,
                                                                det_threshold=data.det_threshold,
                                                                max_size=data.max_size,
                                                                margin=data.margin,
                                                                min_face_ratio=data.min_face_ratio,
                                                                limit_faces=data.limit_faces)

    return response_wrapper(output=output)



@router.post('/multipart/face_registration', tags=['FaceAI'])
async def face_registration_upl(files: List[UploadFile] = File(None),
                                operator: str = Form(""),
                                user_ids: List[str] = Form(None),
                                user_groups: List[str] = Form(None),
                                user_names: List[str] = Form(None),
                                return_embedding: bool = Form(False),
                                det_threshold: float = Form(configs.defaults.det_threshold),
                                min_face_ratio: float = Form(configs.defaults.min_face_ratio),
                                detect_angle: bool = Form(configs.defaults.detect_angle),
                                max_size: str = Form(configs.defaults.max_size_str),
                                margin: int = Form(configs.defaults.margin),
                                limit_faces: int = Form(0)):
    """
    Face registration endpoint accept json with
    parameters in following format:
       - **images**: dict containing either links or data lists. (*optional*)
       - **user_ids**: The user ids. (*required*)
       - **operator**: Supported operator: ["add", "delete", "replace", "find"]. (*required*)
       - **user_groups**: The user groups name. Default: [""] (*optional*)
       - **user_names**: The user names. Default: [""] (*optional*)
       - **return_embedding**: Return face embedding. Default: True (*optional*)
       - **detect_angle**: Apply angle detection for input images. Default: False (*optional*)
       - **det_threshold**: Detection det_threshold. Default: 0.6  [0, 1] recommended (*optional*)
       - **min_face_ratio**: The minimum face ratio in a whole image. Default: 0.15 (*optional*)
       - **extract_ga**: Extract gender/age. Default: False (*optional*)
       - **max_size**: Resize all images to this proportions. Default: '960,960' (*optional*)
       - **margin**: Return bounding box with margin. Default: 44 (*optional*)
       - **limit_faces**: Maximum number of faces to be processed. 0 for unlimited number. Default: 0 (*optional*)
       \f

       :return:
       List[dict]
    """
    if operator == "":
        output = {'status': 'error', "message": "operator required!".format(operator)}
        return response_wrapper(output)

    if operator not in configs.support_operators:
        output = {'status': 'error', "message": "operator '{0}' has not been supported!".format(operator)}
        return response_wrapper(output)

    #debut
    print(type(user_names))
    print(type(user_names))
    print(type(user_ids))
    user_groups = tools.parse_data(user_groups)
    user_names = tools.parse_data(user_names)
    user_ids = tools.parse_data(user_ids)

    #debug
    print(user_groups, user_names, user_ids)

    if operator == 'delete':
        if len(user_ids) == 0:
            output = {'status': 'error', "message": "user_ids must be set in delete mode"}
        else:
            output = await face_ai.delete_faces_async(user_ids)
    elif operator == 'find':
        if len(user_ids) == 0:
            output = {'status': 'error', "message": "user_ids must be set in find mode"}
        else:
            output = await face_ai.find_faces_async(u_ids=user_ids, return_embedding=return_embedding)
    else:  # add or replace
        if len(user_groups) == 0 or len(user_names) == 0 or len(user_ids) == 0:
            output = {'status': 'error',
                      "message": "user_groups, user_names and user_ids must be set in {} mode".format(operator)}
        else:
            if not (len(user_groups) == len(user_names) == len(user_ids)):
                output = {'status': 'error',
                          "message": "user_groups, user_names and user_ids must be set in same dimension"}
            else:
                if files is None:
                    output = {'status': 'error',
                              "message": "files must be set in {} mode".format(operator)}
                else:
                    images = await image_process.files_wrapper_async(files)

                    max_size = tools.parse_size(size=max_size, def_size=configs.defaults.max_size_str)
                    output = await face_ai.register_faces_async(operator=operator,
                                                                data=images,
                                                                u_groups=user_groups,
                                                                u_names=user_names,
                                                                u_ids=user_ids,
                                                                detect_angle=detect_angle,
                                                                det_threshold=det_threshold,
                                                                max_size=max_size,
                                                                margin=margin,
                                                                min_face_ratio=min_face_ratio,
                                                                limit_faces=limit_faces)

    return response_wrapper(output=output)


@router.post('/face_recognition', tags=['FaceAI'])
async def face_recognition(data: FaceExtract):
    """
    Face recognition endpoint accept json with
    parameters in following format:
       - **images**: dict containing either links or data lists. (*required*)
       - **detect_angle**: Apply angle detection for input images. Default: False (*optional*)
       - **det_threshold**: Detection det_threshold. Default: 0.6  [0, 1] recommended (*optional*)
       - **sim_threshold**: Recognizer face similarity threshold. Default: 0.5  [0.4, 0.8] recommended (*optional*)
       - **min_face_ratio**: The minimum face ratio in a whole image. Default: 0.15 (*optional*)
       - **extract_ga**: Extract gender/age. Default: False (*optional*)
       - **max_size**: Resize all images to this proportions. Default: '960,960' (*optional*)
       - **margin**: Return bounding box with margin. Default: 44 (*optional*)
       - **limit_faces**: Maximum number of faces to be processed. 0 for unlimited number. Default: 0 (*optional*)
       - **user_groups**: The user groups name. Default: "" (*optional*)
       - **user_ids**: The user ids. Default: "" (*optional*)
       - **compare**: Compare two different faces. Default: False (*optional*)
       - **api_ver**: Output data serialization format. Currently only version "1" is supported (*optional*)
       \f

       :return:
       List[dict]
    """
    images = jsonable_encoder(data.images)
    if isinstance(data.max_size, str):
        data.max_size = tools.parse_size(size=data.max_size, def_size=configs.defaults.max_size_str)
    if data.compare:
        if len(images) != 2:
            output = [{"message": "the number of upload files must be 2 in compare mode !!!"}]
        else:
            output = await face_ai.verify_faces_async(data=images, user_ids=None,
                                                      detect_angle=data.detect_angle,
                                                      det_threshold=data.det_threshold,
                                                      sim_threshold=data.sim_threshold,
                                                      max_size=data.max_size, margin=data.margin,
                                                      extract_ga=data.extract_ga,
                                                      min_face_ratio=data.min_face_ratio,
                                                      limit_faces=data.limit_faces)
    else:
        data.user_ids = tools.parse_data(data.user_ids)
        if len(data.user_ids) != 0:
            output = await face_ai.verify_faces_async(data=images, user_ids=data.user_ids,
                                                      detect_angle=data.detect_angle,
                                                      det_threshold=data.det_threshold,
                                                      sim_threshold=data.sim_threshold,
                                                      max_size=data.max_size, margin=data.margin,
                                                      extract_ga=data.extract_ga,
                                                      min_face_ratio=data.min_face_ratio,
                                                      limit_faces=data.limit_faces)
        else:
            user_groups = tools.parse_data(data.user_groups)
            if len(user_groups) == 0:
                user_groups = [None]
            output = await face_ai.recognize_faces_async(data=images,
                                                         detect_angle=data.detect_angle,
                                                         det_threshold=data.det_threshold,
                                                         sim_threshold=data.sim_threshold,
                                                         max_size=data.max_size,
                                                         margin=data.margin,
                                                         group_name=user_groups[0],
                                                         extract_ga=data.extract_ga,
                                                         limit_faces=data.limit_faces)

    return response_wrapper(output=output)


@router.post('/multipart/face_recognition', tags=['FaceAI'])
async def face_recognition_upl(files: List[UploadFile] = File(...),
                               det_threshold: float = Form(configs.defaults.det_threshold),
                               sim_threshold: float = Form(configs.defaults.face_sim_threshold),
                               min_face_ratio: float = Form(configs.defaults.min_face_ratio),
                               detect_angle: bool = Form(configs.defaults.detect_angle),
                               extract_ga: bool = Form(configs.defaults.extract_ga),
                               max_size: str = Form(configs.defaults.max_size_str),
                               margin: int = Form(configs.defaults.margin),
                               limit_faces: int = Form(0),
                               user_groups: List[str] = Form(None),
                               user_ids: List[str] = Form(None),
                               compare: bool = Form(False)):
    """
    Face recognition endpoint accept json with
    parameters in following format:

       - **files**: dict containing either links or data lists. (*required*)
       - **detect_angle**: Apply angle detection for input images. Default: False (*optional*)
       - **det_threshold**: Detection det_threshold. Default: 0.6  [0, 1] recommended (*optional*)
       - **sim_threshold**: Recognizer face similarity threshold. Default: 0.5  [0.4, 0.8] recommended (*optional*)
       - **min_face_ratio**: The minimum face ratio in a whole image. Default: 0.15 (*optional*)
       - **extract_ga**: Extract gender/age. Default: False (*optional*)
       - **max_size**: Resize all images to this proportions. Default: '960,960' (*optional*)
       - **margin**: Return bounding box with margin. Default: 44 (*optional*)
       - **limit_faces**: Maximum number of faces to be processed. 0 for unlimited number. Default: 0 (*optional*)
       - **user_groups**: The user groups name. Default: "" (*optional*)
       - **user_ids**: The user ids. Default: [""] (*optional*)
       - **compare**: Compare two different faces. Default: False (*optional*)
       \f

       :return:
       List[dict]
    """
    images = await image_process.files_wrapper_async(files)
    max_size = tools.parse_size(size=max_size, def_size=configs.defaults.max_size_str)
    output = []
    if compare:
        if len(images) % 2 != 0:
            return [{"message": "the number of upload files must be 2 in compare mode !!!"}]
        face_dict = await face_ai.verify_faces_async(data=images, user_ids=None,
                                                     detect_angle=detect_angle,
                                                     det_threshold=det_threshold,
                                                     sim_threshold=sim_threshold,
                                                     max_size=max_size, margin=margin,
                                                     extract_ga=extract_ga,
                                                     min_face_ratio=min_face_ratio,
                                                     limit_faces=limit_faces)
        output.append(face_dict)
    else:
        user_ids = tools.parse_data(user_ids)
        if len(user_ids) != 0:
            face_dict = await face_ai.verify_faces_async(data=images, user_ids=user_ids,
                                                         detect_angle=detect_angle,
                                                         det_threshold=det_threshold,
                                                         sim_threshold=sim_threshold,
                                                         max_size=max_size, margin=margin,
                                                         extract_ga=extract_ga,
                                                         min_face_ratio=min_face_ratio,
                                                         limit_faces=limit_faces)
            output.append(face_dict)
        else:
            user_groups = tools.parse_data(user_groups)
            if len(user_groups) == 0:
                user_groups = [None]
            output = await face_ai.recognize_faces_async(data=images,
                                                         detect_angle=detect_angle,
                                                         det_threshold=det_threshold,
                                                         sim_threshold=sim_threshold,
                                                         max_size=max_size,
                                                         margin=margin,
                                                         group_name=user_groups[0],
                                                         extract_ga=extract_ga,
                                                         limit_faces=limit_faces)

    return response_wrapper(output=output)


@router.post('/draw_recognitions', tags=['FaceAI'])
async def draw_recognitions(data: BodyDraw):
    """
    Return image with drawn faces recognition information for testing purposes.
       - **images**: dict containing either links or data lists. (*required*)
       - **det_threshold**: Detection det_threshold. Default: 0.6  [0, 1] recommended (*optional*)
       - **sim_threshold**: Recognizer face similarity threshold. Default: 0.5  [0.4, 0.8] recommended (*optional*)
       - **detect_angle**: Apply angle detection for input images. Default: False (*optional*)
       - **draw_landmarks**: Draw faces landmarks Default: True (*optional*)
       - **draw_scores**: Draw detection scores Default: True (*optional*)
       - **text_color**: Draw text color. Default: (255, 0, 0) (*optional*)
       - **bbox_color**: Draw text color. Default: (0, 255, 0) (*optional*)
       - **draw_age_gender**: Draw ages and genders: True (*optional*)
       - **margin**: bounding box margin: 44 (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       \f
    """

    images = jsonable_encoder(data.images)
    try:
        text_color = tools.parse_color(color=data.text_color, def_color="255,0,0")
        bbox_color = tools.parse_color(color=data.bbox_color, def_color="0,255,0")
        if len(text_color) != 3:
            text_color = (255, 0, 0)
        if len(bbox_color) != 3:
            bbox_color = (0, 255, 0)
    except Exception as e:
        text_color = (255, 0, 0)
        bbox_color = (0, 255, 0)
    output = await face_ai.draw_recognitions_async(images=images,
                                                   det_threshold=data.det_threshold,
                                                   sim_threshold=data.sim_threshold,
                                                   detect_angle=data.detect_angle,
                                                   draw_landmarks=data.draw_landmarks,
                                                   draw_scores=data.draw_scores,
                                                   draw_age_gender=data.draw_age_gender,
                                                   margin=data.margin,
                                                   limit_faces=data.limit_faces,
                                                   text_color=text_color,
                                                   bbox_color=bbox_color,
                                                   multipart=False)
    output.seek(0)
    return StreamingResponse(output, media_type="image/png")


@router.post('/multipart/draw_recognitions', tags=['FaceAI'])
async def draw_recognitions_upl(file: UploadFile = File(...),
                                det_threshold: float = Form(configs.defaults.det_threshold),
                                sim_threshold: float = Form(configs.defaults.face_sim_threshold),
                                margin: int = Form(configs.defaults.margin),
                                detect_angle: bool = Form(True),
                                draw_landmarks: bool = Form(True),
                                draw_scores: bool = Form(True),
                                text_color: str = Form("255,0,0"),
                                bbox_color: str = Form("0,255,0"),
                                draw_age_gender: bool = Form(True),
                                limit_faces: int = Form(0),
                                group_name: str = Form("")):
    """
    Return image with drawn faces recognition information for testing purposes.

       - **file**: dict containing either links or data lists. (*required*)
       - **det_threshold**: Detection det_threshold. Default: 0.6  [0, 1] recommended (*optional*)
       - **sim_threshold**: Recognizer face similarity threshold. Default: 0.5  [0.4, 0.8] recommended (*optional*)
       - **detect_angle**: Apply angle detection for input images. Default: False (*optional*)
       - **draw_landmarks**: Draw faces landmarks Default: True (*optional*)
       - **draw_scores**: Draw detection scores Default: True (*optional*)
       - **text_color**: Draw text color. Default: (255, 0, 0) (*optional*)
       - **bbox_color**: Draw text color. Default: (0, 255, 0) (*optional*)
       - **draw_age_gender**: Draw ages and genders: True (*optional*)
       - **margin**: bounding box margin: 44 (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       \f
    """

    try:
        text_color = tools.parse_color(color=text_color, def_color="255,0,0")
        bbox_color = tools.parse_color(color=bbox_color, def_color="0,255,0")
        if len(text_color) != 3:
            text_color = (255, 0, 0)
        if len(bbox_color) != 3:
            bbox_color = (0, 255, 0)
    except Exception as e:
        text_color = (255, 0, 0)
        bbox_color = (0, 255, 0)
    output = await face_ai.draw_recognitions_async(images=file,
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
                                                   multipart=True,
                                                   group_name=group_name)
    output.seek(0)
    return StreamingResponse(output, media_type='image/jpg')


@router.post('/get_faces_info', tags=['Detection & recognition'])
async def get_faces_info(data: BodyExtract):
    """
    Get Face information such as extract endpoint accept json with
    parameters in following format:

       - **images**: dict containing either links or data lists. (*required*)
       - **max_size**: Resize all images to this proportions. Default: '960,960' (*optional*)
       - **det_threshold**: Detection det_threshold. Default: 0.6  [0, 1] recommended (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       - **embed_only**: Treat input images as face crops, omit detection step. Default: False (*optional*)
       - **detect_angle**: Apply angle detection for input images. Default: False (*optional*)
       - **extract_embedding**: Extract face embeddings (otherwise only detect faces). Default: True (*optional*)
       - **extract_ga**: Extract gender/age. Default: False (*optional*)
       - **return_face_data**: Return face crops encoded in base64. Default: False (*optional*)
       - **return_landmarks**: Return face landmarks. Default: False (*optional*)
       - **api_ver**: Output data serialization format. Currently only version "1" is supported (*optional*)
       \f

       :return:
       List[List[dict]]
    """
    images = jsonable_encoder(data.images)
    if isinstance(data.max_size, str):
        data.max_size = tools.parse_size(size=data.max_size, def_size=configs.defaults.max_size_str)
    output = await face_ai.get_faces_info_async(images, max_size=data.max_size,
                                                det_threshold=data.det_threshold,
                                                limit_faces=data.limit_faces,
                                                embed_only=data.embed_only,
                                                detect_angle=data.detect_angle,
                                                extract_embedding=data.extract_embedding,
                                                extract_ga=data.extract_ga,
                                                return_face_data=data.return_face_data,
                                                return_landmarks=data.return_landmarks,
                                                api_ver=data.api_ver)
    return response_wrapper(output=output)


@router.post('/multipart/get_faces_info', tags=['Detection & recognition'])
async def get_faces_info_upl(files: List[UploadFile] = File(...),
                             det_threshold: float = Form(configs.defaults.det_threshold),
                             detect_angle: bool = Form(configs.defaults.detect_angle),
                             extract_embedding: bool = Form(configs.defaults.extract_embedding),
                             extract_ga: bool = Form(configs.defaults.extract_ga),
                             return_face_data: bool = Form(configs.defaults.return_face_data),
                             return_landmarks: bool = Form(configs.defaults.return_landmarks),
                             limit_faces: int = Form(0),
                             embed_only: bool = Form(False),
                             max_size: str = Form(configs.defaults.max_size_str),
                             api_ver: str = Form(configs.defaults.api_ver)):
    """
    Get Face information such as extract endpoint accept json with
    parameters in following format:

       - **images**: dict containing either links or data lists. (*required*)
       - **max_size**: Resize all images to this proportions. Default: '960,960' (*optional*)
       - **det_threshold**: Detection det_threshold. Default: 0.6  [0, 1] recommended (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       - **embed_only**: Treat input images as face crops, omit detection step. Default: False (*optional*)
       - **detect_angle**: Apply angle detection for input images. Default: False (*optional*)
       - **extract_embedding**: Extract face embeddings (otherwise only detect faces). Default: True (*optional*)
       - **extract_ga**: Extract gender/age. Default: False (*optional*)
       - **return_face_data**: Return face crops encoded in base64. Default: False (*optional*)
       - **return_landmarks**: Return face landmarks. Default: False (*optional*)
       - **api_ver**: Output data serialization format. Currently only version "1" is supported (*optional*)
       \f

       :return:
       List[List[dict]]
    """

    images = await image_process.files_wrapper_async(files)
    max_size = tools.parse_size(size=max_size, def_size=configs.defaults.max_size_str)
    output = await face_ai.get_faces_info_async(images, max_size=max_size,
                                                det_threshold=det_threshold,
                                                limit_faces=limit_faces,
                                                embed_only=embed_only,
                                                detect_angle=detect_angle,
                                                extract_embedding=extract_embedding,
                                                extract_ga=extract_ga,
                                                return_face_data=return_face_data,
                                                return_landmarks=return_landmarks,
                                                api_ver=api_ver)
    return response_wrapper(output=output)


@router.post('/extract', tags=['Detection & recognition'])
async def extract(data: BodyExtract):
    """
    Face extraction/embeddings endpoint accept json with
    parameters in following format:

       - **images**: dict containing either links or data lists. (*required*)
       - **max_size**: Resize all images to this proportions. Default: '960,960' (*optional*)
       - **det_threshold**: Detection det_threshold. Default: 0.6  [0, 1] recommended (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       - **embed_only**: Treat input images as face crops, omit detection step. Default: False (*optional*)
       - **detect_angle**: Apply angle detection for input images. Default: False (*optional*)
       - **return_face_data**: Return face crops encoded in base64. Default: False (*optional*)
       - **return_landmarks**: Return face landmarks. Default: False (*optional*)
       - **extract_embedding**: Extract face embeddings (otherwise only detect faces). Default: True (*optional*)
       - **extract_ga**: Extract gender/age. Default: False (*optional*)
       - **verbose_timings**: Return all timings. Default: False (*optional*)
       - **api_ver**: Output data serialization format. Currently only version "1" is supported (*optional*)
       \f

       :return:
       List[List[dict]]
    """
    images = jsonable_encoder(data.images)
    output = await face_ai.extract_async(images, max_size=data.max_size, det_threshold=data.det_threshold,
                                         limit_faces=data.limit_faces, embed_only=data.embed_only,
                                         detect_angle=data.detect_angle, return_face_data=data.return_face_data,
                                         return_landmarks=data.return_landmarks,
                                         extract_embedding=data.extract_embedding, extract_ga=data.extract_ga,
                                         verbose_timings=data.verbose_timings, api_ver=data.api_ver)
    return response_wrapper(output=output)


@router.post('/draw_detections', tags=['Detection & recognition'])
async def draw(data: BodyDraw):
    """
    Return image with drawn faces for testing purposes.

       - **images**: dict containing either links or data lists. (*required*)
       - **det_threshold**: Detection det_threshold. Default: 0.6  [0, 1] recommended (*optional*)
       - **detect_angle**: Apply angle detection for input images. Default: False (*optional*)
       - **draw_landmarks**: Draw faces landmarks Default: True (*optional*)
       - **draw_scores**: Draw detection scores Default: True (*optional*)
       - **draw_sizes**: Draw face sizes Default: True (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       \f
    """

    images = jsonable_encoder(data.images)
    output = await face_ai.draw_async(images,
                                      det_threshold=data.det_threshold,
                                      detect_angle=data.detect_angle,
                                      draw_landmarks=data.draw_landmarks,
                                      draw_scores=data.draw_scores,
                                      draw_sizes=data.draw_sizes,
                                      limit_faces=data.limit_faces,
                                      multipart=False)
    output.seek(0)
    return StreamingResponse(output, media_type="image/png")


@router.post('/multipart/draw_detections', tags=['Detection & recognition'])
async def draw_upl(file: bytes = File(...), det_threshold: float = Form(configs.defaults.det_threshold),
                   detect_angle: bool = Form(True), draw_landmarks: bool = Form(True), draw_scores: bool = Form(True),
                   draw_sizes: bool = Form(True), limit_faces: int = Form(0)):
    """
    Return image with drawn faces for testing purposes.

       - **file**: dict containing either links or data lists. (*required*)
       - **det_threshold**: Detection det_threshold. Default: 0.6  [0, 1] recommended (*optional*)
       - **detect_angle**: Apply angle detection for input images. Default: False (*optional*)
       - **draw_landmarks**: Draw faces landmarks Default: True (*optional*)
       - **draw_scores**: Draw detection scores Default: True (*optional*)
       - **draw_sizes**: Draw face sizes Default: True (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       \f
    """

    output = await face_ai.draw_async(file, det_threshold=det_threshold,
                                      detect_angle=detect_angle,
                                      draw_landmarks=draw_landmarks,
                                      draw_scores=draw_scores,
                                      draw_sizes=draw_sizes,
                                      limit_faces=limit_faces,
                                      multipart=True)
    output.seek(0)
    return StreamingResponse(output, media_type='image/jpg')