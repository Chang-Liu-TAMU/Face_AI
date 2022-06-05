# -*- coding: utf-8 -*-
"""
## image utils
@author: asher
"""
import six
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO


def bytes_to_array(string_binary):
    img = cv2.imdecode(np.fromstring(string_binary, np.uint8), cv2.IMREAD_COLOR)
    # buf = six.BytesIO()
    # buf.write(string_binary)
    # buf.seek(0)
    # img = Image.open(buf).convert('RGB')
    return np.asarray(img)


def base64_to_PIL(img_string):
    """
    base64 string to PIL
    """
    try:
        base64_data = base64.b64decode(img_string)
        buf = six.BytesIO()
        buf.write(base64_data)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img
    except:
        return None


def PIL_to_base64(image, format="png"):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    output = BytesIO()
    image.save(output, format=format)
    bytes_str = output.getvalue()
    base64_str = base64.b64encode(bytes_str)
    output.close()
    return base64_str


def base64_to_cv(img_string):
    """
    base64 string to cv
    """
    try:
        base64_data = base64.b64decode(img_string)
        img = cv2.imdecode(np.fromstring(base64_data, np.uint8), cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return img
    except:
        return None


def cv_to_base64(image, format="jpg"):
    """
    image to base64 string
    """
    # cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    base64_str = cv2.imencode(".{}".format(format), image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str


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

    supported_angle = [0, 90, 180, 270]
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
    elif angle == 270:
        out = cv2.transpose(image)
        rotated_image = cv2.flip(out, flipCode=0 if clockwise else 1)
    else:
        return image
    return rotated_image


def cv_resize_image(image, min_side=640, dsize=None, padding=False):
    """Resize image by a certain length.

        Parameters
        ----------
        image : ndarray
            Input image.
        min_side : int
            Minimum side to keep.
        dsize : tuple, optional
            the new size.
        padding : bool, optional
            Determine whether apply padding on given image. Default is False.

        Returns
        -------
        rotated : ndarray
            Resized version of the input.
    """

    if dsize is not None:
        return cv2.resize(src=image, dsize=dsize)

    h, w = image.shape[0:2]
    # max side will be scaled to min sides
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(image, (new_w, new_h))

    if padding:
        #  padding to min_side * min_side
        if new_w % 2 != 0 and new_h % 2 == 0:
            top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, \
                                       (min_side - new_w) / 2 + 1, (min_side - new_w) / 2
        elif new_h % 2 != 0 and new_w % 2 == 0:
            top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, \
                                       (min_side - new_w) / 2, (min_side - new_w) / 2
        elif new_h % 2 == 0 and new_w % 2 == 0:
            top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, \
                                       (min_side - new_w) / 2, (min_side - new_w) / 2
        else:
            top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, \
                                       (min_side - new_w) / 2 + 1, (min_side - new_w) / 2

        # padding pixels from image margin including top, bottom, left, right
        resize_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right),
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return resize_img
