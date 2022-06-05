# @Time: 2022/5/24 13:49
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:image_process.py
import base64
import time
import traceback
import cv2
import numpy as np
from turbojpeg import TurboJPEG

try:
    jpeg = TurboJPEG()
except:
    jpeg = None
    print("turbojpeg problem occurs...")


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def decode_image(b64encoded, logger=None):
    im_data = dict(data=None,
                   traceback=None)
    try:
        __bin = b64encoded.split(",")[-1]
        __bin = base64.b64decode(__bin)
        t0 = time.time()
        try:
            _image = jpeg.decode(__bin)
        except:
            message = 'TurboJPEG failed, fallback to cv2.imdecode'
            if logger is not None:
                logger.debug(message)
            else:
                print(message)
            __bin = np.fromstring(__bin, np.uint8)
            _image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)
        t1 = time.time()
        message = f'Decoding took: {t1 - t0}'
        if logger is not None:
            logger.debug(message)
        else:
            print(message)
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
    return im_data


async def decode_image_async(b64encoded, logger=None):
    return decode_image(b64encoded=b64encoded, logger=logger)

