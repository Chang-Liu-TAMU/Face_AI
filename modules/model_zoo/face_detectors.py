# @Time: 2022/5/18 8:50
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:face_detectors.py

from .detectors.retinaface import RetinaFace


def get_retinaface(model_path, backend, outputs, rac, masks=False, **kwargs):
    inference_backend = backend.DetectorInfer(model=model_path, output_order=outputs, **kwargs)
    model = RetinaFace(inference_backend=inference_backend, rac=rac, masks=masks)
    return model


def retinaface_r50_v1(model_path, backend, outputs, **kwargs):
    model = get_retinaface(model_path, backend, outputs, rac="net3", **kwargs)
    return model



