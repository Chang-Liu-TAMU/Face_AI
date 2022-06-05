# @Time: 2022/5/24 14:08
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:serialization.py
import cv2
import base64

class Serializer:

    def serialize(self, data, api_ver: str = '1'):
        serializer = self.get_serializer(api_ver)
        return serializer(data)

    def get_serializer(self, api_ver):
        if api_ver == '1':
            return self._serializer_v1
        else:
            return self._serializer_v2

    def _serializer_v1(self, data):
        data = data.get('data', [])
        resp = [img.get('faces') for img in data]
        return resp

    def _serializer_v2(self, data):

        # Response data is by default in v2 format
        return data


def serialize_face(face, return_face_data: bool = False,
                   return_landmarks: bool = False, use_serialization: bool = True):
    _face_dict = dict(
        det=face.num_det,
        prob=None,
        bbox=None,
        size=None,
        landmarks=None,
        gender=face.gender,
        age=face.age,
        mask_prob=None,
        norm=None,
        vec=None,
    )

    if face.embedding_norm:
        if use_serialization:
            _face_dict.update(vec=face.normed_embedding.tolist(),
                              norm=float(face.embedding_norm))
        else:
            _face_dict.update(vec=face.normed_embedding,
                              norm=float(face.embedding_norm))

    # Warkaround for embed_only flag
    if face.det_score:
        if use_serialization:
            _face_dict.update(prob=float(face.det_score),
                              bbox=face.bbox.astype(int).tolist(),
                              size=int(face.bbox[2] - face.bbox[0]))
        else:
            _face_dict.update(prob=float(face.det_score),
                              bbox=face.bbox.astype(int),
                              size=int(face.bbox[2] - face.bbox[0]))

        if return_landmarks:
            if use_serialization:
                _face_dict.update({
                    'landmarks': face.landmark.astype(int).tolist()
                })
            else:
                _face_dict.update({
                    'landmarks': face.landmark.astype(int)
                })

    if face.mask_probs:
        _face_dict.update(mask_prob=float(face.mask_probs["mask"]))

    if return_face_data:
        if use_serialization:
            _face_dict.update({
                'facedata': base64.b64encode(
                    cv2.imencode('.jpg', face.facedata)[1].tostring()).decode('utf-8')
            })
        else:
            _face_dict.update({
                'facedata': face.facedata
            })
    else:
        _face_dict.update({'facedata': None})

    return _face_dict
