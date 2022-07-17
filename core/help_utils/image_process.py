# @Time: 2022/5/25 14:40
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:image_process.py

import base64
import os
import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image

def drawPoints(bgr_image, landmarks):
    for landmark in landmarks:
        assert (len(landmark) % 2 == 0)
        landmark_cord = []
        num_cords = len(landmark) // 2
        for i in range(num_cords):
            landmark_cord.append((landmark[i], landmark[num_cords + i]))

        for mk in landmark_cord:
            mk = (int(mk[0]), int(mk[1]))
            cv2.circle(bgr_image, mk, color=(0, 255, 0), radius=1, thickness=2)


def get_save_image_text(bgr_image, boxes, boxes_name, landmarks=None,
                        id_scores=None, ages=None, genders=None, filename=None,
                        text_color=(0, 0, 255), bbox_color=(0, 255, 0)):
    '''
    :param boxes_name:
    :param filename:
    :param bgr_image: bgr image
    :param boxes: [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :param landmarks: []
    :param id_scores: []
    :param ages: []
    :param genders: []
    :param text_color: (0, 0, 255)
    :param bbox_color: (0, 255, 0)
    :return:
    '''
    box_widths = []
    box_heights = []

    # fix opencv channel brg
    bbox_color = tuple(bbox_color[::-1])
    text_color = tuple(text_color)

    for label, box in zip(boxes_name, boxes):
        box_widths.append(box[2] - box[0])
        box_heights.append(box[3] - box[1])
        if label == 'unknown':
            cv2.rectangle(bgr_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2, 8, 0)
        else:
            cv2.rectangle(bgr_image, (box[0], box[1]), (box[2], box[3]), bbox_color, 2, 8, 0)
    if landmarks:
        drawPoints(bgr_image, landmarks)
    box_max = max(np.mean(box_widths), np.mean(box_heights))
    img_PIL = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_PIL)

    # 字体颜色
    # img_fraction = 1.6
    img_fraction = 1.0
    cur_path = os.path.dirname(os.path.realpath(__file__))
    font_path = os.path.join(cur_path, "ukai.ttc")
    for i, (label, box) in enumerate(list(zip(boxes_name, boxes))):
        fontsize = 10  # starting font size
        # portion of image width you want text width to be
        font = ImageFont.truetype(font_path, fontsize)
        while font.getsize(label)[0] < img_fraction * box_max:
            # iterate until the text size is just larger than the criteria
            fontsize += 1
            font = ImageFont.truetype(font_path, fontsize)

        # 文字输出位置
        position = (box[0], box[1] - 1.05 * font.size)
        draw.text(position, text=label, fill=text_color, font=font)

        if id_scores:
            extra_label = "id_score:{}".format(id_scores[i])
            position = (position[0], position[1] - font.size)
            draw.text(position, text=extra_label, fill=text_color, font=font)
        if genders:
            extra_label = "gender:{}".format(genders[i])
            position = (position[0], position[1] - font.size)
            draw.text(position, text=extra_label, fill=text_color, font=font)
        if ages:
            extra_label = "age:{}".format(ages[i])
            position = (position[0], position[1] - font.size)
            draw.text(position, text=extra_label, fill=text_color, font=font)

    if filename is not None:
        img_PIL.save(filename)

    out_image = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return out_image


def cv_rotate_image(image, angle, clockwise=False):
    """Rotate image by a certain angle around its center.

    Parameters
    ----------
    image : ndarray
        Input image.
    angle : int
        Rotation angle in degrees in default direction (counter-clockwise).
    clockwise : bool, optional
        Determine rotation direction. Default is False, namely counter-clockwise.

    Returns
    -------
    rotated : ndarray
        Rotated version of the input.
    """

    supported_angle = [0, 90, 180, 270, -90]
    # if given angle is unsupported or equal to zero and return unchanged image
    if angle not in supported_angle and angle == 0:
        return image

    """ flipCode a flag to specify how to flip the array; 0 means
    .   flipping around the x-axis and positive value (for example, 1) means
    .   flipping around y-axis. Negative value (for example, -1) means flipping
    .   around both axes."""
    if angle == 90:
        out = cv2.transpose(image)
        rotated_image = cv2.flip(out, flipCode=1 if clockwise else 0)
    elif angle == 180:
        rotated_image = cv2.flip(image, flipCode=-1)
    elif angle == 270 or angle == -90:
        out = cv2.transpose(image)
        rotated_image = cv2.flip(out, flipCode=0 if clockwise else 1)
    else:
        return image
    return rotated_image


def visualize_img(data, detections_dict, draw_landmarks=False,
                  draw_scores=False, draw_age_gender=False,
                  only_bbox=False, result_file_path=None,
                  text_color=(0, 0, 255), bbox_color=(0, 255, 0)):
    if isinstance(data, str):
        cv_img = cv2.imread(data)
    else:
        cv_img = data

    angle = int(detections_dict["rotate_angle"])
    rotate_img = cv_rotate_image(image=cv_img, angle=angle, clockwise=False)
    box_names = []
    bboxes = []
    landmark_list = []
    ages = []
    genders = []
    id_scores = []
    for key, value in detections_dict.items():
        if key.startswith("face"):
            age = value["age"]
            ages.append(age)
            gender = value["gender"]
            genders.append(gender)
            id_score = value["probability"]
            id_scores.append(id_score)
            if only_bbox:
                box_names.append(str(id_score))
                landmarks = value["landmark"]
                landmark_list.append(list(map(float, landmarks.split(','))))
            else:
                group = value["group"]
                id = value["user_id"]
                name = value["name"]
                if "face_score" in value.keys():
                    face_score = float(value["face_score"])
                    if id == "unknown":
                        label = "unknown_" + str(round(face_score, 3))
                    else:
                        label = '_'.join([group, id, name, str(round(face_score, 3))])
                else:
                    if id == "unknown":
                        label = "unknown"
                    else:
                        label = '_'.join([group, id, name])
                box_names.append(label)

            rect = value["face_rectangle"]

            box = [int(rect["left"]), int(rect["top"]),
                   int(rect["width"]) + int(rect["left"]),
                   int(rect["height"]) + int(rect["top"])]
            bboxes.append(box)

    if not draw_scores:
        id_scores = None
    if not draw_landmarks:
        landmark_list = None
    if not draw_age_gender:
        ages = None
        genders = None

    if result_file_path is not None:
        name, ext = os.path.splitext(os.path.basename(data))
        out_put_name = os.path.join(result_file_path, "{}_result{}".format(name, ext))
        out_image = get_save_image_text(rotate_img, bboxes, box_names, landmarks=landmark_list,
                                        id_scores=id_scores, ages=ages, genders=genders,
                                        filename=out_put_name, text_color=text_color, bbox_color=bbox_color)
    else:
        out_image = get_save_image_text(rotate_img, bboxes, box_names, landmarks=landmark_list,
                                        id_scores=id_scores, ages=ages, genders=genders, filename=None,
                                        text_color=text_color, bbox_color=bbox_color)
    return out_image


def buffer2base64(buffer_bytes):
    encoded = base64.b64encode(buffer_bytes).decode('ascii')
    return encoded

async def files_wrapper_async(files_buffer):
    images = dict()
    images["data"] = []
    images["image_ids"] = []
    for buffer in files_buffer:
        buffer_bytes = await buffer.read()
        images["data"].append(buffer2base64(buffer_bytes=buffer_bytes))
        images["image_ids"].append(buffer.filename)
    return images



