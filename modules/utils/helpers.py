# @Time: 2022/5/17 14:48
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:helpers.py

import os
from itertools import chain, islice
from distutils import util
import logging

def prepare_folders(paths):
    logging.info("preparing model folders ...")
    for path in paths:
        os.makedirs(path, exist_ok=True)

def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

def colorize_log(string, color):
    colors = dict(
        grey="\x1b[38;21m",
        yellow="\x1b[33;21m",
        red="\x1b[31;21m",
        bold_red="\x1b[31;1m",
        green="\x1b[32;1m",
    )
    reset = "\x1b[0m"
    col = colors.get(color)
    if col is None:
        return string
    string = f"{col}{string}{reset}"
    return string

def validate_max_size(max_size):
    if max_size[0] % 32 != 0 or max_size[1] % 32 != 0:
        max_size[0] = max_size[0] // 32 * 32
        max_size[1] = max_size[1] // 32 * 32
        logging.warning(f'Input image dimensions should be multiples of 32. Max size changed to: {max_size}')
    return max_size

# Translate bboxes and landmarks from resized to original image size
def reproject_points(dets, scale: float):
    if scale != 1.0:
        dets = dets / scale
    return dets




