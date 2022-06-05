# @Time: 2022/5/17 14:37
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:processing.py

#standard
from typing import List, Union
import io
import time
import logging

from numpy.linalg import norm

#own
#core class
from .face_model import FaceAnalysis, Face
from .utils import matrix_tools, image_processing, fast_face_align
from .utils.helpers import to_chunks, reproject_points
from .imagedata import ImageData

#decode images func
from modules.utils.image_provider import *
# from modules.utils.image_processing import *
from modules.utils.serialization import *
from core.help_utils.logger_utils import internal_logger


class Processing:
    def __init__(self, det_name: str = 'retinaface_r50_v1', rec_name: str = 'glintr100',
                 ga_name: str = 'genderage_v1', mask_detector: str = "mask_detector",device: str = 'cuda', max_size: List[int] = None,
                 backend_name: str = 'trt', max_rec_batch_size: int = 1, max_det_batch_size: int = 1,force_fp16: bool = False,
                 triton_uri=None, root_dir: str = '/models', download_model: bool=False, logger=None):
        if logger is None:
            self.logger = internal_logger()
        else:
            self.logger = logging
        # self.logger = logging

        self.max_rec_batch_size = max_rec_batch_size
        self.max_det_batch_size = max_det_batch_size
        self.det_name = det_name
        self.rec_name = rec_name
        self.ga_name = ga_name
        self.max_size = max_size
        self.model = FaceAnalysis(det_name=det_name, rec_name=rec_name, ga_name=ga_name,
                                  mask_detector=mask_detector, device=device,
                                  max_size=self.max_size, max_rec_batch_size=self.max_rec_batch_size,
                                  max_det_batch_size=max_det_batch_size,
                                  backend_name=backend_name, force_fp16=force_fp16, triton_uri=triton_uri,
                                  root_dir=root_dir, download_model=download_model)

        self.models_info = self.check_initialization()


    def check_initialization(self):
        res = {}
        res["analysis_model"] = self.model is not None
        res["det_model"] = self.model.det_model is not None
        res["rec_model"] = self.model.rec_name is not None
        res["ga_model"] = self.model.ga_model is not None
        res["mask_model"] = self.model.mask_model is not None
        print(res)
        return res

    def __iterate_faces(self, crops):
        for face in crops:
            if face.get('traceback') is None:
                face = Face(facedata=face.get('data'))
                yield face

    def prepocess(self, img, det_arr, points=None, margin=44):
        """
        :param img:
        :param det_arr:
        :param points:
        :param margin:
        :return: add margins to bbx and produce face crops
        """
        bbs = []
        face_images = []
        for i, det in enumerate(det_arr):
            landmarks = points[i]
            det = np.squeeze(det)
            if det.shape[0] == 0:
                return None, None
            bb = np.zeros(5, dtype=np.float32)
            bb[0] = np.maximum(det[0] - margin // 2, 0)
            bb[1] = np.maximum(det[1] - margin // 2, 0)
            bb[2] = np.minimum(det[2] + margin // 2, img.shape[1])
            bb[3] = np.minimum(det[3] + margin // 2, img.shape[0])
            bb[4] = det[4]

            if len(landmarks.shape) == 1 and landmarks.shape[0] == 10:
                landmarks = landmarks.reshape((2, 5)).T
            nimg = fast_face_align.norm_crop(img, landmark=landmarks)
            face_images.append(nimg)
            bbs.append(bb)
        return face_images, bbs




    #angle detection
    def detect_single_angle(self, image):
        angle = 0
        if image is None:
            return angle, image

        try:
            from modules import pcn_wrapper
        except Exception as e:
            self.logger.warning("Do not support image angle detection and ignore it!")
            return angle, image

        return pcn_wrapper.detect(image)

    def detect_angle(self, images):
        """
        Note: this function may modify input images
        :param images: input image list
        :return: angle list
        """
        angles = [0 for i in range(len(images))]

        try:
            from modules import pcn_wrapper
        except Exception as e:
            self.logger.warning("Do not support image angle detection and ignore it!")
            return angles

        for index, image_data in enumerate(images):
            if image_data.get('traceback') is None:
                image = image_data.get('data')
                angles[index], img = pcn_wrapper.detect(image)
                image_data["data"] = img
            else:
                self.logger.warning("ignore invalid image data!")
        return angles

    async def detect_angle_async(self, images):
        """
        Note: this function may modify input images
        :param images: input image list
        :return: angle list
        """
        return self.detect_angle(images=images)
    #angle detection


    def detect_face(self, img, det_thresh=0.7, max_size=[640, 640], limit_faces=0):
        '''
        :param img:
        :param det_thresh:
        :param max_size:
        :param limit_faces:
        :return: detection faces (locations, lmks) from a single image
        '''
        bounding_boxes, points = self._detect_face(img, det_thresh=det_thresh,
                                     max_size=max_size, limit_faces=limit_faces)
        return bounding_boxes, points

    def _detect_face(self, img, det_thresh=0.7, max_size=[640, 640], limit_faces=0):
        # If detector has input_shape attribute, use it instead of provided value
        try:
            max_size = self.model.det_model.retina.input_shape[2:][::-1]
        except:
            pass

        print("image shape:", img.shape)

        img = ImageData(img, max_size=max_size)
        img.resize_image(mode='pad')
        print("transformed image shape:", img.transformed_image.shape)
        #debug
        print("calling self.model.det_model.detect")
        boxes, probs, landmarks = self.model.det_model.detect(img.transformed_image, threshold=det_thresh)

        #tem solution
        boxes, probs, landmarks = boxes[0], probs[0], landmarks[0]
        #
        if boxes.shape[0] == 0:
            return [], []

        bboxes_array = np.zeros((len(boxes), 5), dtype=np.float32)

        if 0 < limit_faces < boxes.shape[0]:
            boxes, probs, landmarks = self.model.sort_boxes(boxes, probs, landmarks,
                                                                                  shape=img.transformed_image.shape,
                                                                                  max_num=limit_faces)
        for i in range(len(boxes)):
            # Translate points to original image size
            bboxes_array[i, 4] = probs[i]
            bbox = reproject_points(boxes[i], img.scale_factor)
            bboxes_array[i, 0:4] = bbox
            landmarks[i] = reproject_points(landmarks[i], img.scale_factor)

        landmarks = np.hstack(landmarks).T.reshape(landmarks.shape[0], 10)
        return bboxes_array, landmarks

    def get_age_gender(self, aligned_list):
        """
        :param aligned_list: List[face crops]
        :return: List[age], List[gender]
        """
        ages = []
        genders = []
        if self.models_info["ga_model"]:
            chunked_faces = to_chunks(aligned_list, self.max_rec_batch_size)
            for chunk in chunked_faces:
                chunk = list(chunk)
                crops = [e for e in chunk]
                ga = self.model.ga_model.get(crops)
                for gender, age in ga:
                    if gender == 0:
                        gender = 'female'
                    elif gender == 1:
                        gender = 'male'
                    else:
                        gender = 'unknown'
                    ages.append(age)
                    genders.append(gender)
        else:
            ages = None
            genders = None
        return ages, genders

    def get_embedding(self, image_crops, use_normalization=True):
        """
        :param image_crops:
        :param use_normalization:
        :return: embeddings from a batch of face crops
        """
        chunked_faces = to_chunks(image_crops, self.max_rec_batch_size)
        embeddings_list = []
        for chunk in chunked_faces:
            chunk = list(chunk)
            crops = [e for e in chunk]
            embeddings = self.model.rec_model.get_embedding(crops)
            if use_normalization:
                for i, crop in enumerate(crops):
                    embedding = embeddings[i]
                    embeddings[i] = embedding / norm(embedding)
            embeddings_list.append(np.squeeze(np.array(embeddings)))

        return np.vstack(embeddings_list)


    def recognize_faces(self, face_images, face_dataset, dis_threshold, group_name=None):
        return matrix_tools.compare_embedding(self.get_embedding(face_images), feature_dicts=face_dataset,
                                              threshold=dis_threshold, group_name=group_name)


    @staticmethod
    def embedding_distance(embeddings1, embeddings2, distance_metric=0, axis=0):
        return matrix_tools.distance(embeddings1, embeddings2, distance_metric=distance_metric, axis=axis)


    def embed_crops(self, images, extract_embedding: bool = True, extract_ga: bool = True):
        '''
        :param images: List(Dict(face crops data))
        :param extract_embedding:
        :param extract_ga:
        :return: complete face data
        '''
        t0 = time.time()
        output = dict(took=None, data=[], status="ok")

        iterator = self.__iterate_faces(images)
        faces = self.model.process_faces(iterator, extract_embedding=extract_embedding,
                                         extract_ga=extract_ga, return_face_data=False,
                                         detect_masks=True, mask_thresh=0.8)

        try:
            for image in images:
                if image.get('traceback') is not None:
                    _face_dict = dict(status='failed',
                                      traceback=image.get('traceback'))
                else:
                    _face_dict = serialize_face(face=next(faces), return_face_data=False,
                                                return_landmarks=False)
                output['data'].append(_face_dict)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            output['status'] = 'failed'
            output['traceback'] = tb

        took = time.time() - t0
        output['took'] = took
        return output

    def embed(self, images: List[Dict], angles: List[int] = None, max_size: List[int] = None,
              threshold: float = 0.6, limit_faces: int = 0, extract_embedding: bool = True,
              extract_ga: bool = True, return_face_data: bool = False, return_landmarks: bool = False):
        """
        :param images:
        :param angles:
        :param max_size:
        :param threshold:
        :param limit_faces:
        :param extract_embedding:
        :param extract_ga:
        :param return_face_data:
        :param return_landmarks:
        :return: get embeddings from list of original images
        """

        assert len(images) == len(angles)

        output = dict(took={}, data=[])

        for image_data, angle in zip(images, angles):
            _faces_dict = dict(status=None, took=None, faces=[])
            try:
                t1 = time.time()
                if image_data.get('traceback') is not None:
                    _faces_dict['status'] = 'failed'
                    _faces_dict['traceback'] = image_data.get('traceback')
                else:
                    image = image_data.get('data')
                    faces = self.model.get(image, max_size=max_size, threshold=threshold,
                                           extract_embedding=extract_embedding, detect_masks=True,
                                           extract_ga=extract_ga, limit_faces=limit_faces,
                                           return_face_data=return_face_data, mask_thresh=0.89)

                    print(f"totally generate {len(faces)} face")
                    for idx, face in enumerate(faces):
                        _face_dict = serialize_face(face=face, return_face_data=return_face_data,
                                                    return_landmarks=return_landmarks)
                        _faces_dict['faces'].append(_face_dict)
                    took_image = time.time() - t1
                    _faces_dict['took'] = took_image
                    _faces_dict['angle'] = angle
                    _faces_dict['status'] = 'ok'

            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                _faces_dict['status'] = 'failed'
                _faces_dict['traceback'] = tb

            output['data'].append(_faces_dict)
        return output

    async def embed_async(self, images: List[Dict], angles: List[int] = None, max_size: List[int] = None,
                          threshold: float = 0.6, limit_faces: int = 0, extract_embedding: bool = True,
                          extract_ga: bool = True, return_face_data: bool = False, return_landmarks: bool = False):

        assert len(images) == len(angles)
        output = dict(took={}, data=[])
        for image_data, angle in zip(images, angles):
            _faces_dict = dict(status=None, took=None, faces=[])
            try:
                t1 = time.time()
                if image_data.get('traceback') is not None:
                    _faces_dict['status'] = 'failed'
                    _faces_dict['traceback'] = image_data.get('traceback')
                else:
                    image = image_data.get('data')
                    faces = await self.model.get_async(image, max_size=max_size, threshold=threshold,
                                           extract_embedding=extract_embedding, detect_masks=True,
                                           extract_ga=extract_ga, limit_faces=limit_faces,
                                           return_face_data=return_face_data, mask_thresh=0.89)

                    for idx, face in enumerate(faces):
                        _face_dict = serialize_face(face=face, return_face_data=return_face_data,
                                                    return_landmarks=return_landmarks)
                        _faces_dict['faces'].append(_face_dict)
                    took_image = time.time() - t1
                    _faces_dict['took'] = took_image
                    _faces_dict['angle'] = angle
                    _faces_dict['status'] = 'ok'

            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                _faces_dict['status'] = 'failed'
                _faces_dict['traceback'] = tb

            output['data'].append(_faces_dict)
        return output


    def extract(self, images: Dict[str, list], max_size: List[int] = None, threshold: float = 0.8,
                limit_faces: int = 0, embed_only: bool = False, detect_angle: bool = False,
                return_face_data: bool = False, return_landmarks: bool = False,
                extract_embedding: bool = True, extract_ga: bool = True,
                verbose_timings=True, api_ver: str = "1"):

        if not max_size:
            max_size = self.max_size

        t0 = time.time()

        tl0 = time.time()
        images = get_images(images, logger=self.logger)
        tl1 = time.time()
        took_loading = tl1 - tl0
        self.logger.debug(f'Reading images took: {took_loading} s.')

        if detect_angle:
            angles = self.detect_angle(images=images)
        else:
            angles = [0 for i in range(len(images))]

        if embed_only:
            _faces_dict = self.embed_crops(images, extract_embedding=extract_embedding, extract_ga=extract_ga)
            return _faces_dict
        else:
            te0 = time.time()
            output = self.embed(images, angles=angles, max_size=max_size, threshold=threshold,
                                limit_faces=limit_faces, extract_embedding=extract_embedding, extract_ga=extract_ga,
                                return_face_data=return_face_data, return_landmarks=return_landmarks)
            took_embed = time.time() - te0
            took = time.time() - t0
            output['took']['total'] = took
            if verbose_timings:
                output['took']['read_imgs'] = took_loading
                output['took']['embed_all'] = took_embed

            serializer = Serializer()
            return serializer.serialize(output, api_ver=api_ver)

    async def extract_async(self, images: Dict[str, list], max_size: List[int] = None, threshold: float = 0.6,
                            limit_faces: int = 0, embed_only: bool = False, detect_angle: bool = False,
                            return_face_data: bool = False, return_landmarks: bool = False,
                            extract_embedding: bool = True, extract_ga: bool = True,
                            verbose_timings=True, api_ver: str = "1"):

        if not max_size:
            max_size = self.max_size

        t0 = time.time()

        tl0 = time.time()
        images = await get_images_async(images, logger=self.logger)
        tl1 = time.time()
        took_loading = tl1 - tl0
        self.logger.debug(f'Reading images took: {took_loading} s.')

        if detect_angle:
            angles = await self.detect_angle_async(images=images)
        else:
            angles = [0 for i in range(len(images))]

        if embed_only:
            _faces_dict = self.embed_crops(images, extract_embedding=extract_embedding, extract_ga=extract_ga)
            return _faces_dict
        else:
            te0 = time.time()
            output = await self.embed_async(images, angles=angles, max_size=max_size,
                                            threshold=threshold, limit_faces=limit_faces,
                                            extract_embedding=extract_embedding, extract_ga=extract_ga,
                                            return_face_data=return_face_data, return_landmarks=return_landmarks)
            took_embed = time.time() - te0
            took = time.time() - t0
            output['took']['total'] = took
            if verbose_timings:
                output['took']['read_imgs'] = took_loading
                output['took']['embed_all'] = took_embed

            serializer = Serializer()
            return serializer.serialize(output, api_ver=api_ver)


    def draw(self, images: Union[Dict[str, list], bytes], threshold: float = 0.9,
             detect_angle: bool = False, draw_landmarks: bool = True, draw_scores: bool = True,
             draw_sizes: bool = True, limit_faces=0, multipart=False, test=False):
        if isinstance(images, np.ndarray):
            image = images
        elif not multipart:
            images = get_images(images, logger=self.logger)
            if detect_angle:
                angles = self.detect_angle(images=images)
                angle = angles[0]
            else:
                angle = 0
            image = images[0].get('data')
        else:
            __bin = np.fromstring(images, np.uint8)
            image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)
            if detect_angle:
                angle, image = self.detect_single_angle(image=image)
            else:
                angle = 0

        faces = self.model.get(image, threshold=threshold,
                                extract_embedding=False, detect_masks=False,
                                extract_ga=False, limit_faces=limit_faces,
                                return_face_data=False)

        for face in faces:
            bbox = face.bbox.astype(int)
            pt1 = tuple(bbox[0:2])
            pt2 = tuple(bbox[2:4])
            color = (0, 255, 0)
            x, y = pt1
            r, b = pt2
            w = r - x
            if face.mask_probs:
                if face.mask_probs >= 0.2:
                    color = (0, 255, 255)
            cv2.rectangle(image, pt1, pt2, color, 1)

            if draw_landmarks:
                lms = face.landmark.astype(int)
                pt_size = int(w * 0.05)
                cv2.circle(image, (lms[0][0], lms[0][1]), 1, (0, 0, 255), pt_size)
                cv2.circle(image, (lms[1][0], lms[1][1]), 1, (0, 255, 255), pt_size)
                cv2.circle(image, (lms[2][0], lms[2][1]), 1, (255, 0, 255), pt_size)
                cv2.circle(image, (lms[3][0], lms[3][1]), 1, (0, 255, 0), pt_size)
                cv2.circle(image, (lms[4][0], lms[4][1]), 1, (255, 0, 0), pt_size)

            if draw_scores:
                text = f"{face.det_score:.3f}"
                pos = (x + 3, y - 5)
                textcolor = (0, 0, 0)
                thickness = 1
                border = int(thickness / 2)
                cv2.rectangle(image, (x - border, y - 21, w + thickness, 21), color, -1, 16)
                cv2.putText(image, text, pos, 0, 0.5, (0, 255, 0), 3, 16)
                cv2.putText(image, text, pos, 0, 0.5, textcolor, 1, 16)
            if draw_sizes:
                text = f"w:{w}"
                pos = (x + 3, b - 5)
                cv2.putText(image, text, pos, 0, 0.5, (0, 0, 0), 3, 16)
                cv2.putText(image, text, pos, 0, 0.5, (0, 255, 0), 1, 16)

        if detect_angle:
            total = f'faces: {len(faces)} angle: {str(angle)} ({self.det_name})'
        else:
            total = f'faces: {len(faces)} ({self.det_name})'
        bottom = image.shape[0]
        cv2.putText(image, total, (5, bottom - 5), 0, 1, (0, 0, 0), 3, 16)
        cv2.putText(image, total, (5, bottom - 5), 0, 1, (0, 255, 0), 1, 16)
        if test:
            return image
        is_success, buffer = cv2.imencode(".jpg", image)
        io_buf = io.BytesIO(buffer)
        return io_buf

    async def draw_async(self, images: Union[Dict[str, list], bytes], threshold: float = 0.6,
                         detect_angle: bool = False, draw_landmarks: bool = True, draw_scores: bool = True,
                         draw_sizes: bool = True, limit_faces=0, multipart=False):

        if not multipart:
            images = await get_images_async(images, logger=self.logger)
            if detect_angle:
                angles = await self.detect_angle_async(images=images)
                angle = angles[0]
            else:
                angle = 0
            image = images[0].get('data')
        else:
            __bin = np.fromstring(images, np.uint8)
            image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)
            if detect_angle:
                angle, image = self.detect_single_angle(image=image)
            else:
                angle = 0

        faces = await self.model.get_async(image, threshold=threshold,
                                extract_embedding=False, detect_masks=False,
                                extract_ga=False, limit_faces=limit_faces,
                                return_face_data=False)

        for face in faces:
            bbox = face.bbox.astype(int)
            pt1 = tuple(bbox[0:2])
            pt2 = tuple(bbox[2:4])
            color = (0, 255, 0)
            x, y = pt1
            r, b = pt2
            w = r - x
            if face.mask_probs:
                if face.mask_probs >= 0.2:
                    color = (0, 255, 255)
            cv2.rectangle(image, pt1, pt2, color, 1)

            if draw_landmarks:
                lms = face.landmark.astype(int)
                pt_size = int(w * 0.05)
                cv2.circle(image, (lms[0][0], lms[0][1]), 1, (0, 0, 255), pt_size)
                cv2.circle(image, (lms[1][0], lms[1][1]), 1, (0, 255, 255), pt_size)
                cv2.circle(image, (lms[2][0], lms[2][1]), 1, (255, 0, 255), pt_size)
                cv2.circle(image, (lms[3][0], lms[3][1]), 1, (0, 255, 0), pt_size)
                cv2.circle(image, (lms[4][0], lms[4][1]), 1, (255, 0, 0), pt_size)

            if draw_scores:
                text = f"{face.det_score:.3f}"
                pos = (x + 3, y - 5)
                textcolor = (0, 0, 0)
                thickness = 1
                border = int(thickness / 2)
                cv2.rectangle(image, (x - border, y - 21, w + thickness, 21), color, -1, 16)
                cv2.putText(image, text, pos, 0, 0.5, (0, 255, 0), 3, 16)
                cv2.putText(image, text, pos, 0, 0.5, textcolor, 1, 16)
            if draw_sizes:
                text = f"w:{w}"
                pos = (x + 3, b - 5)
                cv2.putText(image, text, pos, 0, 0.5, (0, 0, 0), 3, 16)
                cv2.putText(image, text, pos, 0, 0.5, (0, 255, 0), 1, 16)

        if detect_angle:
            total = f'faces: {len(faces)} angle: {str(angle)} ({self.det_name})'
        else:
            total = f'faces: {len(faces)} ({self.det_name})'
        bottom = image.shape[0]
        cv2.putText(image, total, (5, bottom - 5), 0, 1, (0, 0, 0), 3, 16)
        cv2.putText(image, total, (5, bottom - 5), 0, 1, (0, 255, 0), 1, 16)

        is_success, buffer = cv2.imencode(".jpg", image)
        io_buf = io.BytesIO(buffer)
        return io_buf
