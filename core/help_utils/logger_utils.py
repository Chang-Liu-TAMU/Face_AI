from core.env_parser import EnvConfigs
from core.help_utils.manager_utils import Logger

logger = None


def internal_logger():
    global logger
    if logger is None:
        logger = Logger(log_file=EnvConfigs().log_file, log_name='FaceAI_Server').logger
    return logger