# @Time: 2022/5/31 9:16
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:custom_log.py

#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
================================================================
Copyright (C) 2019 * Ltd. All rights reserved.
PROJECT      :  BiddingExtraction
FILE_NAME    :  custom_log
AUTHOR       :  DAHAI LU
TIME         :  2021/5/13 下午2:32
PRODUCT_NAME :  PyCharm
================================================================
"""

import os
import sys
import time
import logging
from loguru import logger
from pprint import pformat

# -----------------------系统调试------------------------------------
DEBUG = True
# -----------------------日志-----------------------------------------
LOG_DIR = os.path.join(os.getcwd(), f'application\log\{time.strftime("%Y-%m-%d")}.log')
LOG_FORMAT = '<level>{level: <8}</level>  <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>'


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def format_record(record: dict) -> str:
    format_string = LOG_FORMAT

    if record["extra"].get("payload") is not None:
        record["extra"]["payload"] = pformat(
            record["extra"]["payload"], indent=4, compact=True, width=88
        )
        format_string += "\n<level>{extra[payload]}</level>"

    format_string += "{exception}\n"
    return format_string


def init_logger(configs, logger):
    logging.basicConfig(
        level=configs.log_level,
        format='%(asctime)s %(levelname)s - %(message)s',
        datefmt='[%H:%M:%S]',
    )
    logging.getLogger().handlers = [InterceptHandler()]
    gunicorn_error_logger = logging.getLogger('gunicorn.error')
    gunicorn_access_logger = logging.getLogger('gunicorn.access')
    logging.getLogger('uvicorn.access').handlers = gunicorn_access_logger.handlers
    logging.getLogger('uvicorn.error').handlers = gunicorn_error_logger.handlers

    logger.handlers = gunicorn_error_logger.handlers
    logger.configure(handlers=[{"sink": sys.stdout, "level": configs.log_level,
                                "format": format_record, "serialize": configs.defaults.json_log}])
    logger.add(configs.server_log_file, encoding='utf-8', rotation="23:00")