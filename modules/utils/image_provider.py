# @Time: 2022/5/17 14:56
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:image_provider.py

import urllib
import urllib.request
import traceback
import os

import cv2
import numpy as np

# from turbojpeg import TurboJPEG
from typing import Dict

# jpeg = TurboJPEG()

from .image_processing import jpeg, decode_image, decode_image_async, to_rgb

def dl_image(path, headers=None, logger=None):
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
        }

    im_data = dict(data=None,
                   traceback=None)

    try:
        if path.startswith('http'):
            req = urllib.request.Request(
                path,
                headers=headers
            )

            resp = urllib.request.urlopen(req)
            __bin = bytearray(resp.read())
            try:
                _image = jpeg.decode(__bin)
            except:
                message = 'TurboJPEG failed, fallback to cv2.imdecode'
                if logger is not None:
                    logger.debug(message)
                else:
                    print(message)
                _image = np.asarray(__bin, dtype="uint8")
                _image = cv2.imdecode(_image, cv2.IMREAD_COLOR)
        else:
            if not os.path.exists(path):
                im_data.update(traceback=f"File: '{path}' not found")
                return im_data
            try:
                in_file = open(path, 'rb')
                _image = jpeg.decode(in_file.read())
                in_file.close()
            except:
                message = 'TurboJPEG failed, fallback to cv2.imdecode'
                if logger is not None:
                    logger.debug(message)
                else:
                    print(message)
                _image = cv2.imread(path, cv2.IMREAD_COLOR)
    except Exception:
        tb = traceback.format_exc()
        if logger is not None:
            logger.warning(tb)
        else:
            print(tb)
        im_data.update(traceback=tb)
        return im_data

    if _image.ndim == 2:
        _image = to_rgb(_image)
    _image = _image[:, :, 0:3]
    im_data.update(data=_image)
    if isinstance(path, str):
        im_data.update(image_id=os.path.basename(path).split(".")[0])
    else:
        im_data.update(image_id="None")
    return im_data


async def dl_image_async(path, headers=None, logger=None):
    return dl_image(path=path, headers=headers, logger=logger)



def get_images(data: Dict[str, list], logger=None):
    images = []
    if isinstance(data, np.ndarray):
        images.append({"traceback": None,
                       "data": data})
    elif data.get('urls') is not None:
        urls = data['urls']
        for url in urls:
            _image = dl_image(url, logger=logger)
            images.append(_image)
    elif data.get('data') is not None:
        b64_images = data['data']
        image_ids = data.get("image_ids", [])
        images = []
        for i, b64_img in enumerate(b64_images):
            _image = decode_image(b64_img, logger=logger)
            if i < len(image_ids):
                _image.update(image_id=image_ids[i])
            else:
                _image.update(image_id="None")
            images.append(_image)

    return images


async def get_images_async(data: Dict[str, list], logger=None):
    images = []
    if data.get('urls') is not None:
        urls = data['urls']
        for url in urls:
            _image = await dl_image_async(url, logger=logger)
            images.append(_image)
    elif data.get('data') is not None:
        b64_images = data['data']
        image_ids = data.get("image_ids", [])
        images = []
        for i, b64_img in enumerate(b64_images):
            _image = await decode_image_async(b64_img, logger=logger)
            if i < len(image_ids):
                _image.update(image_id=image_ids[i])
            else:
                _image.update(image_id="None")
            images.append(_image)

    return images


