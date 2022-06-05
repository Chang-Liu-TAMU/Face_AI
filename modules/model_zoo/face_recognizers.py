# @Time: 2022/5/18 8:50
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:face_recognizers.py

def arcface_torch(model_path, backend, **kwargs):
    model = backend.Arcface(rec_name=model_path, input_mean=127.5, input_std=127.5, **kwargs)
    return model


# Backend wrapper for Gender/Age estimation model.
def genderage_v1(model_path, backend, **kwargs):
    model = backend.FaceGenderage(rec_name=model_path, **kwargs)
    return model

# Backend wrapper for mask detection model.
def mask_detector(model_path, backend, **kwargs):
    model = backend.MaskDetection(rec_name=model_path, **kwargs)
    return model
