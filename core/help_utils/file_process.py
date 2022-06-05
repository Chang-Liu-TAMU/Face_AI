# @Time: 2022/5/25 14:40
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:file_process.py
import numpy as np
from collections import OrderedDict
import json


def convert_to_similarity(dis, min_limit=0, max_limit=0.99, precision=3):
    similarity = np.cos(dis)
    return float(min(round(max(similarity, min_limit), precision), max_limit))

def convert_to_json_with_ga(image_id, *args):
    boxes, boxes_name, distances, rotate_angle, ages, genders = args
    total_number = len(boxes)
    assert total_number == len(boxes_name) == len(distances)
    data = OrderedDict()
    index = 1
    missing_math_num = 0
    for labels, box, dis in zip(boxes_name, boxes, distances):
        age = None if ages is None else ages[index - 1]
        gender = None if genders is None else genders[index - 1]
        if labels == 'unknown':
            group = 'unknown'
            name = 'unknown'
            user_id = 'unknown'
            result = 0
            missing_math_num += 1
        else:
            label_list = labels.split('_')
            group = label_list[0]
            name = label_list[1]
            user_id = label_list[2]
            result = -1
        box_with = int(box[2] - box[0])
        box_height = int(box[3] - box[1])
        box_left = int(box[0])
        box_top = int(box[1])
        face_score = box[4]
        probability = convert_to_similarity(dis, min_limit=0, max_limit=0.99, precision=4)

        face = {"user_id": user_id,
                "name": name,
                "group": group,
                "age": str(age),
                "gender": str(gender),
                "probability": str(probability),
                "face_score": str(round(face_score, 5)),
                "result": str(result),
                "face_rectangle": {
                    "width": str(box_with),
                    "top": str(box_top),
                    "left": str(box_left),
                    "height": str(box_height)
                }}
        face_key = "{}{}".format("face", str(index))
        index += 1
        data[face_key] = face
    data["image_id"] = image_id
    data["rotate_angle"] = str(rotate_angle)
    data["total_number"] = str(total_number)
    data["match_number"] = str(total_number - missing_math_num)
    data["match_rate"] = str(round((total_number - missing_math_num) * 1.0 / total_number, 4))
    return data


def log_distances_top_k(dis_writer, pic_name, pred_label, top_k_info_list, box_probs):
    """
    log distances
    :param dis_writer:
    :param pic_name:
    :param pred_label:
    :param distances:
    :return:
    """
    detect_number = len(pred_label)
    assert detect_number == len(box_probs)
    dis_writer.write(pic_name)
    dis_writer.write('--> detect number: {}\n'.format(detect_number))
    for i in range(detect_number):
        top_k_info = top_k_info_list[i]
        top_k_info_str = ''
        for info in top_k_info:
            top_k_info_str += '_'.join([info[0], str(round(info[1], 2))])
        text = '{}({}:{}-{}:{})\n'.format(pred_label[i], 'box_prob',
                                          round(box_probs[i], 4), 'dis_info', top_k_info_str)
        dis_writer.write(text)
    dis_writer.write('\n')
    dis_writer.flush()


def json_wrapper(text_dict, ensure_ascii=False, sort_keys=False, indent=4):
    return json.dumps(text_dict, ensure_ascii=ensure_ascii, sort_keys=sort_keys, indent=indent)

