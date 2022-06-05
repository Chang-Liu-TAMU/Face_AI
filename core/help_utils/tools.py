# @Time: 2022/5/25 14:43
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:tools.py
import numpy as np
import datetime
import random
from distutils import util

def to_bool(input):
    try:
        if isinstance(input, str):
            return bool(util.strtobool(input))
        elif isinstance(input, bool):
            return input
        else:
            return False
    except:
        return False

def parse_size(size=None, def_size='640,480'):
    if size is None:
        size = def_size
    size_lst = list(map(int, size.split(',')))
    return size_lst

def parse_features(database):
    if database is None or len(database) == 0:
        return None
    database_array = np.asarray(database)[:, 1:-2]
    feature_list = database_array[:, 3]
    database_array[:, 3] = [np.fromstring(feature, dtype=np.float, sep=',') for feature in feature_list]
    # database_array[:, 3] = [np.asarray(list(map(float, feature.split(','))))
    #                         for feature in feature_list]
    return database_array

def get_bbox_areas(bounding_boxes, img_size, margin=44):
    bbs_area_list = []
    img_height = img_size[0]
    img_width = img_size[1]
    for det in bounding_boxes:
        det = np.squeeze(det)
        bb = [int(np.maximum(det[0] - margin // 2, 0)),
              int(np.maximum(det[1] - margin // 2, 0)),
              int(np.minimum(det[2] + margin // 2, img_width)),
              int(np.minimum(det[3] + margin // 2, img_height))]
        area = (bb[2] - bb[0]) * (bb[3] - bb[1])
        bbs_area_list.append(area)
    return bbs_area_list, img_height * img_width

def probability_to_dis_threshold(probability, precision=4):
    dis_threshold = np.arccos(probability)
    return float(round(dis_threshold, precision))


def filter_unqalified(bounding_boxes, landmarks, threshold):
    new_bboxes = []
    new_points = []
    for box, landmark in zip(bounding_boxes, landmarks):
        if box[4] > threshold:
            new_bboxes.append(box)
            new_points.append(landmark)
    new_bboxes = np.asarray(sorted(new_bboxes, key=lambda x: x[0]))
    new_points = np.asarray(sorted(new_points, key=lambda x: x[0]))
    return new_bboxes, new_points


def face_filter(image, bounding_boxes, points, threshold, margin=44, mini_face_size=8):
    bbs = []
    new_bboxes = []
    box_probs = []
    if bounding_boxes is None or points is None:
        return [], bounding_boxes, points
    if len(bounding_boxes) == 0 or len(points) == 0:
        return [], bounding_boxes, points
    det_arr, landmark_list = filter_unqalified(bounding_boxes, points, threshold=threshold)
    img_size = image.shape[0:2]
    for det in det_arr:
        det = np.squeeze(det)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin // 2, 0)  # x_min
        bb[1] = np.maximum(det[1] - margin // 2, 0)  # y_min
        bb[2] = np.minimum(det[2] + margin // 2, img_size[1])  # x_max
        bb[3] = np.minimum(det[3] + margin // 2, img_size[0])  # y_max
        if bb[2] - bb[0] >= mini_face_size and bb[3] - bb[1] >= mini_face_size:
            bbs.append(bb)
        else:
            continue
    ava_width = np.mean([bb[2] - bb[0] for bb in bbs])
    ava_height = np.mean([bb[3] - bb[1] for bb in bbs])
    new_landmark_list = []
    for tem_det, bb, landmark in zip(det_arr, bbs, landmark_list):
        det = np.squeeze(tem_det)
        face_width = bb[2] - bb[0]
        face_height = bb[3] - bb[1]
        if face_width > ava_width * 0.5 and face_height > ava_height * 0.5:
            new_bboxes.append(tem_det)
            box_probs.append(det[4])
            new_landmark_list.append(landmark)
    return box_probs, new_bboxes, new_landmark_list



# generate a unique id by datetime now
def generate_unique_id():
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    random_number = random.randint(0, 100)
    random_number_str = str(random_number)
    if random_number < 10:
        random_number_str = str(0) + str(random_number)
    now_random_str = now_str + "-" + random_number_str
    return now_random_str


def parse_color(color=None, def_color='255,0,0'):
    if isinstance(color, (tuple, list)):
        return tuple(color)

    if color is None:
        color = def_color
    color_tuple = tuple(map(int, color.split(',')))
    return color_tuple

def parse_data(data):
    if isinstance(data, str):
        if "," in data:
            return [d.strip() for d in data.split(',')]
        elif data == "":
            return []
        else:
            return [data.strip()]
    elif isinstance(data, (tuple, list)):
        new_data = []
        if len(data) == 1 and "," in data[0]:
            new_data = [d.strip() for d in data[0].split(',')]
        else:
            for d in data:
                if d is None or d == "":
                    return []
                new_data.append(str(d).strip())
        return new_data
    else:
        return [] if data is None else [data]

