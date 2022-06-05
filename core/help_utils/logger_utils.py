# @Time: 2022/5/25 14:45
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:logger_utils.py

from core.env_parser import EnvConfigs
from core.help_utils.manager_utils import Logger

logger = None


def internal_logger():
    global logger
    if logger is None:
        logger = Logger(log_file=EnvConfigs().log_file, log_name='FaceAI_Server').logger
    return logger