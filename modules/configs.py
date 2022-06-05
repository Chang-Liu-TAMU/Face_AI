# @Time: 2022/5/18 14:03
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:configs.py


import os
from collections import namedtuple

# Net outputs in correct order expected by postprocessing code.
# TensorRT might change output order for some reasons.
# Also Triton Inference Server may change output order for both
# ONNX and TensorRT backends if automatic configuration is used.

retina_outputs = ['face_rpn_cls_prob_reshape_stride32',
                  'face_rpn_bbox_pred_stride32',
                  'face_rpn_landmark_pred_stride32',
                  'face_rpn_cls_prob_reshape_stride16',
                  'face_rpn_bbox_pred_stride16',
                  'face_rpn_landmark_pred_stride16',
                  'face_rpn_cls_prob_reshape_stride8',
                  'face_rpn_bbox_pred_stride8',
                  'face_rpn_landmark_pred_stride8']

anticov_outputs = [
    'face_rpn_cls_prob_reshape_stride32',
    'face_rpn_bbox_pred_stride32',
    'face_rpn_landmark_pred_stride32',
    'face_rpn_type_prob_reshape_stride32',
    'face_rpn_cls_prob_reshape_stride16',
    'face_rpn_bbox_pred_stride16',
    'face_rpn_landmark_pred_stride16',
    'face_rpn_type_prob_reshape_stride16',
    'face_rpn_cls_prob_reshape_stride8',
    'face_rpn_bbox_pred_stride8',
    'face_rpn_landmark_pred_stride8',
    'face_rpn_type_prob_reshape_stride8'
]

centerface_outputs = ['537', '538', '539', '540']
dbface_outputs = ["hm", "tlrb", "landmark"]
scrfd_outputs = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32', 'kps_8', 'kps_16', 'kps_32']
yolo_outputs = ['output']

models = {
    'retinaface_r50_v1': {
        'shape': (1, 3, 480, 640),
        'outputs': retina_outputs,
        'reshape': True,
        'in_package': False,
        'function': 'retinaface_r50_v1',
        'link': '1peUaq0TtNBhoXUbMqsCyQdL7t5JuhHMH',
        'dl_type': 'google'
    },

    'genderage_v1': {
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'reshape': False,
        'in_package': False,
        'function': 'genderage_v1',
        'link': '1MnkqBzQHLlIaI7gEoa9dd6CeknXMCyZH',
        'dl_type': 'google'
    },
    'centerface': {
        'in_package': False,
        'shape': (1, 3, 480, 640),
        'reshape': True,
        'function': 'centerface',
        'outputs': centerface_outputs,
        'link': 'https://raw.githubusercontent.com/Star-Clouds/CenterFace/master/models/onnx/centerface_bnmerged.onnx'
    },

    'glintr100': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'reshape': False,
        'function': 'arcface_torch',
        'link': '1TR_ImGvuY7Dt22a9BOAUAlHasFfkrJp-',
        'dl_type': 'google'
    },
    'w600k_r50': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'function': 'arcface_torch',
        'reshape': False,
        'link': '1_3WcTE64Mlt_12PZHNWdhVCRpoPiblwq',
        'dl_type': 'google'
    },

    'w600k_mbf': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'function': 'arcface_torch',
        'reshape': False,
        'link': '1GtBKfGucgJDRLHvGWR3jOQovHYXY-Lpe',
        'dl_type': 'google'
    },

    'mask_detector': {
        'shape': (1, 224, 224, 3),
        'allow_batching': True,
        'reshape': False,
        'in_package': False,
        'function': 'mask_detector',
        'link': '1RsQonthhpJDwwdcB0sYsVGMTqPgGdMGV',
        'dl_type': 'google'
    },

    'mask_detector112': {
        'shape': (1, 112, 112, 3),
        'allow_batching': True,
        'reshape': False,
        'in_package': False,
        'function': 'mask_detector',
        'link': '1ghS0LEGV70Jdb5un5fVdDO-vmonVIe6Z',
        'dl_type': 'google'
    },

    # You can put your own pretrained ArcFace model to /models/onnx/custom_rec_model
    'custom_rec_model': {
        'in_package': False,
        'shape': (1, 3, 112, 112),
        'allow_batching': True,
        'function': 'arcface_torch',
        'reshape': False
    }
}


class Configs(object):
    def __init__(self, models_dir: str = '/models'):
        self.models_dir = self.__get_param('MODELS_DIR', models_dir)
        self.onnx_models_dir = os.path.join(self.models_dir, 'onnx')
        self.trt_engines_dir = os.path.join(self.models_dir, 'trt-engines')
        self.models = models
        self.type2path = dict(
            onnx=self.onnx_models_dir,
            engine=self.trt_engines_dir,
            plan=self.trt_engines_dir
        )

    def __get_param(self, ENV, default=None):
        return os.environ.get(ENV, default)

    def build_model_paths(self, model_name: str, ext: str):
        base = self.type2path[ext]
        parent = os.path.join(base, model_name)
        file = os.path.join(parent, f"{model_name}.{ext}")
        return parent, file

    def get_outputs_order(self, model_name):
        return self.models.get(model_name, {}).get('outputs')

    def get_shape(self, model_name):
        return self.models.get(model_name, {}).get('shape')

    def get_dl_link(self, model_name):
        return self.models.get(model_name, {}).get('link')

    def get_dl_type(self, model_name):
        return self.models.get(model_name, {}).get('dl_type')
