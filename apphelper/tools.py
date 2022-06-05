#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chineseocr
FILE_NAME    :  tools
AUTHOR       :  DAHAI LU
TIME         :  2019/7/29 下午1:22
PRODUCT_NAME :  PyCharm
"""
import os
import io
import cv2
import json
import errno
import shutil
import base64
import random
import datetime
import numpy as np
from PIL import Image
from os import walk
from os import sep
from os.path import join
from os.path import splitext
from os.path import basename


def parse_size(size=None, def_size='640,480'):
    if size is None:
        size = def_size
    size_lst = list(map(int, size.split(',')))
    return size_lst


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


def convert_pdf_to_image(pdf_file):
    import tempfile
    if isinstance(pdf_file, str):
        from pdf2image import convert_from_path
        with tempfile.TemporaryDirectory() as path:
            image_list = convert_from_path(pdf_file, output_folder=path)
    elif isinstance(pdf_file, bytes):
        from pdf2image import convert_from_bytes
        with tempfile.TemporaryDirectory() as path:
            image_list = convert_from_bytes(pdf_file, output_folder=path)
    else:
        raise TypeError("cannot parse unknown type: {}".format(type(pdf_file)))
    return image_list


def dir_nonempty(dirname):
    # If directory exists and nonempty (ignore hidden files), prompt for action
    return os.path.isdir(dirname) and len([x for x in os.listdir(dirname) if x[0] != '.'])


def mkdir_p(dirname):
    """ Like "mkdir -p", make a dir recursively, but do nothing if the dir exists
    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def empty_dir(dirname, parent=False):
    try:
        # shutil.rmtree(dirname, ignore_errors=True)
        if dir_nonempty(dirname):
            shutil.rmtree(dirname, ignore_errors=False)
    except Exception as e:
        print(e)
    finally:
        if not parent:
            mkdir_p(dirname)


# generate a unique id by datetime now
def generate_unique_id():
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    random_number = random.randint(0, 100)
    random_number_str = str(random_number)
    if random_number < 10:
        random_number_str = str(0) + str(random_number)
    now_random_str = now_str + "-" + random_number_str
    return now_random_str


def get_image_from_io(file):
    try:
        image_array = np.asarray(Image.open(io.BytesIO(file.read())))
        file.seek(0)
    except (OSError, NameError):
        return None
    return image_array


def preprocess_images(file, tmp_folder, ext_format):
    # from werkzeug import secure_filename
    # filename = secure_filename(file.filename)
    filename = file.filename
    file_extention = filename.split('.')[-1]

    file_path_list = []
    if file_extention == "pdf":
        pl_image_list = convert_pdf_to_image(file.read())
        # save_file_name = os.path.splitext(save_file_name)[0]
        for index, img in enumerate(pl_image_list):
            # tmp_name = os.path.join(tmp_folder, "{}_{}.{}".format(save_file_name, index, ext_format))
            ret, buf = cv2.imencode(".jpg", np.asarray(img))
            image_bytes = Image.fromarray(np.uint8(buf)).tobytes()
            file_path_list.append(image_bytes)
            # img.save(tmp_name, ext_format)
    elif file_extention in ["doc", "docx"]:
        from docx import Document
        save_file_name = "{}.{}".format(generate_unique_id(), file_extention)
        os.makedirs(tmp_folder)
        save_file_path = os.path.join(tmp_folder, save_file_name)
        file.save(save_file_path)
        doc = Document(save_file_path)
        for shape in doc.inline_shapes:
            content_id = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
            content_type = doc.part.related_parts[content_id].content_type
            if not content_type.startswith("image"):
                continue
            img_data = doc.part.related_parts[content_id]._blob
            # save_file_name = os.path.basename(doc.part.related_parts[content_id].partname)
            # tmp_name = os.path.join(tmp_folder, save_file_name)
            file_path_list.append(img_data)
            # with open(tmp_name, 'wb') as fp:
            #     fp.write(img_data)
    else:
        file_path_list.append(file.read())
    return file_path_list


def write_to_docx(texts, document):
    for res_dict in texts:
        if "res" in res_dict:
            for res in res_dict["res"]:
                if "text" in res:
                    document.add_paragraph(res["text"])


def getFilePathList(file_dir, recursive=True):
    filePath_list = []
    for w in walk(file_dir):
        part_filePath_list = [join(w[0], file) for file in w[2]]
        filePath_list.extend(part_filePath_list)
        if not recursive:
            break
    return filePath_list


def get_file_base_name(file_path, ignore_postfix=True):
    b_name = basename(file_path)
    if ignore_postfix:
        b_name, ext = splitext(b_name)

    return b_name


def get_files_list(file_dir, postfix='ALL', recursive=True):
    postfix = postfix.split('.')[-1]
    file_list = []
    f_append = file_list.append
    filePath_list = getFilePathList(file_dir, recursive=recursive)
    if postfix == 'ALL':
        file_list = filePath_list
    else:
        for file in filePath_list:
            b_name = basename(file)
            postfix_name = b_name.split('.')[-1]
            if postfix_name == postfix:
                f_append(file)
    file_list.sort()
    return file_list


def gen_files_labels(files_dir, postfix='ALL', recursive=True):
    # filePath_list = getFilePathList(files_dir)
    filePath_list = get_files_list(files_dir, postfix=postfix, recursive=recursive)
    print("files nums:{}".format(len(filePath_list)))
    label_list = []
    for filePath in filePath_list:
        label = filePath.split(sep)[-2]
        label_list.append(label)

    labels_set = list(set(label_list))
    print("labels:{}".format(labels_set))

    return filePath_list, label_list


def read_img_base64(p):
    if isinstance(p, bytes):
        imgString = base64.b64encode(p)
    elif isinstance(p, str):
        with open(p, 'rb') as f:
            imgString = base64.b64encode(f.read())
    else:
        raise TypeError("cannot parse unknown type: {}".format(type(p)))
    imgString = b'data:image/jpeg;base64,' + imgString
    return imgString.decode()


# define the allowed file for uploading
def allowed_file(filename, allow_extensions={'png', 'jpg', 'jpeg', 'gif'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allow_extensions


def get_image_plt(file, logger=None):
    try:
        image_plt = Image.open(io.BytesIO(file.read()))
    except (OSError, NameError):
        message = 'the image : {} has been damaged, cannot be opened...'.format(file.filename)
        if logger is not None:
            logger.error(message)
        else:
            print(message)
        return None
    finally:
        file.seek(0)
    return image_plt


def check_image_size(image_plt, mini_length=500, logger=None):
    min_side_length = min(image_plt.size)
    if min_side_length < mini_length:
        message = "detect invalid image size: {}".format(min_side_length)
        if logger is not None:
            logger.warning(message)
        else:
            print(message)
        return False
    else:
        return True


def get_dir_size(dir):
    size = 0
    file_num = 0
    for root, dirs, files in os.walk(dir):
        file_num += len(files)
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    return size / 1024 / 1024 / 1024, file_num  # GB


def get_file_list_time_sorted(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # os.path.getmtime() last modify time
        # os.path.getctime() last create time
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        # print(dir_list)
        return dir_list


def clear_cache_if_necessary(cache_dir, cache_size=1000, logger=None):
    file_size, file_num = get_dir_size(cache_dir)
    if file_size > 1 or file_num > cache_size:  # > 1GB
        message = "detect cache size is more than 1 GB or file number is more than {}, " \
                  "and will remove the oldest history cache.".format(cache_size)
        if logger is not None:
            logger.info(message)
        else:
            print(message)
        file_list = get_file_list_time_sorted(cache_dir)
        for i in range(len(file_list) // 2 + 1):
            data = os.path.join(cache_dir, file_list[i])
            if os.path.isdir(data):
                empty_dir(data)
                if logger is not None:
                    logger.info("remove directory {} ......".format(data))
                else:
                    print("remove directory {} ......".format(data))
            elif os.path.isfile(data):
                os.remove(data)
                # app.logger.info("remove file {} ......".format(data))
