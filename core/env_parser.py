# @Time: 2022/5/30 9:18
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:env_parser.py
import os
from os import path as osp
from tabulate import tabulate
from core.help_utils.tools import to_bool, parse_size
from core.help_utils.tools import probability_to_dis_threshold as prob_to_dis_thresh

class Defaults:
    def __init__(self):
        # Global parameters
        self.root_path = osp.dirname(osp.dirname(osp.realpath(__file__)))
        print(self.root_path)
        # self.root_path = "/model"
        self.core_path = osp.join(self.root_path, "core")

        self.staff_face_path = osp.join(self.core_path, "dataset/insight_feature_db")
        if not osp.exists(self.staff_face_path):
            os.makedirs(self.staff_face_path, exist_ok=True)

        self.margin = int(os.getenv('MARGIN', 44))
        self.max_size = parse_size(os.getenv('MAX_SIZE'), def_size='960,960')
        self.max_size_str = ",".join(list(map(str, self.max_size)))
        self.min_face_ratio = float(os.getenv('MIN_FACE_RATIO', 0.15))
        self.det_threshold = float(os.getenv('DET_THRESH', 0.6))
        #  60 / 180 * np.pi == 1.0472 --> cos(60°) = 0.5
        self.face_sim_threshold = float(os.getenv('FACE_SIM_THRESH', 0.5))
        self.face_dis_threshold = prob_to_dis_thresh(probability=self.face_sim_threshold, precision=4)
        self.max_faces_per_uid = int(os.getenv('MAX_FACES_PER_UID', 7))
        self.dis_metric = int(os.getenv('DISTANCE_METRIC', 0))
        self.return_face_data = to_bool(os.getenv('DEF_RETURN_FACE_DATA', False))
        self.return_landmarks = to_bool(os.getenv('DEF_RETURN_LANDMARKS', False))
        self.extract_embedding = to_bool(os.getenv('DEF_EXTRACT_EMBEDDING', True))
        self.extract_ga = to_bool(os.getenv('DEF_EXTRACT_GA', False))
        self.detect_angle = to_bool(os.getenv('DEF_DETECT_ANGLE', False))
        self.models_path = str(os.getenv('MODELS_PATH', osp.join(self.root_path, "core/models")))
        self.json_log = True if os.getenv("JSON_LOGS", "0") == "1" else False
        self.api_ver = os.getenv('DEF_API_VER', "2")


class Database:
    def __init__(self):
        # mysql
        self.db_host = os.getenv('DATABASE_HOST', '127.0.0.1')
        self.db_port = os.getenv('DATABASE_PORT', '3306')
        self.db_user = os.getenv('DATABASE_USER', 'root')
        self.db_password = os.getenv('DATABASE_PASSWORD', 'root')
        self.face_db_name = os.getenv('DATABASE_NAME', 'face')
        self.db_table = os.getenv('DATABASE_TABLE', 'face_json')
        self.user_db_name = os.getenv('AUTHORITY_TABLE', 'user')
        self.db_api_table = os.getenv('API_TABLE', 'sys_api')
        self.db_user_table = os.getenv('USER_TABLE', 'sys_user')
        self.db_authority_table = os.getenv('AUTHORITY_TABLE', 'sys_authorities')

        # redis
        self.redis_enabled = to_bool(os.getenv('REDIS_ENABLED', False))
        self.redis_host = os.getenv('REDIS_HOST', '127.0.0.1')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_passwd = os.getenv('REDIS_PASSWORD', 'root12345')
        self.redis_db = int(os.getenv('REDIS_DB', 0))
        self.redis_url = f"redis://:{self.redis_passwd}@{self.redis_host}:{self.redis_port}/{self.redis_db}?encoding=utf-8"
        self.redis_timeout = int(os.getenv('REDIS_TIMEOUT', 5))


class Authority:
    def __init__(self):
        # Ciphertext
        self.encryption_enabled = to_bool(os.getenv('ENCRYPTION_ENABLED', True))

        # token
        self.token_type = os.getenv('TOKEN_TYPE', 'Bearer')
        self.token_expire_minutes = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 60 * 24 * 7))  # minutes, a week
        self.secret_algorithm = os.getenv("SECRET_ALGORITHM", "HS256")
        self.secret_key = os.getenv("SECRET_KEY", "fa5bf890c1c502fbf7460c9ea27c1cfa13f1776c5c4dd6b19bcd6a103c3c6f85")

        # administrator users
        self.admin_name = os.getenv("ADMIN_NAME", "erow")
        self.admin_passwd = os.getenv("ADMIN_PASSWORD", "erow2021")



class Models:
    def __init__(self):
        self.backend_name = os.getenv('INFERENCE_BACKEND', 'trt')
        self.device = os.getenv("DEVICE", 'cuda')
        # retinaface_r50_v1, scrfd_10g_gnkps
        self.det_name = os.getenv("DET_NAME", "retinaface_r50_v1")
        # arcface_r100_v1, glintr100
        self.rec_name = os.getenv("REC_NAME", "glintr100")
        self.ga_name = os.getenv("GA_NAME", "genderage_v1")
        self.rec_batch_size = int(os.getenv('REC_BATCH_SIZE', 1))
        self.det_batch_size = int(os.getenv('DET_BATCH_SIZE', 1))
        self.download_model = to_bool(os.getenv("DOWNLOAD_MODEL", False))
        self.fp16 = to_bool(os.getenv('FORCE_FP16', False))
        self.ga_ignore = to_bool(os.getenv('GA_IGNORE', False))
        self.rec_ignore = to_bool(os.getenv('REC_IGNORE', False))
        self.triton_uri = os.getenv("TRITON_URI", None)

        if self.rec_ignore:
            self.rec_name = None
        if self.ga_ignore:
            self.ga_name = None

class EnvConfigs:
    def __init__(self):
        # version
        self.version = os.getenv("VERSION", "0.1.2.0")
        self.develop = to_bool(os.getenv("DEVELOP", True))

        # gunicorn
        self.num_workers = int(os.getenv("NUM_WORKERS", 1))
        self.port = int(os.getenv('PORT', 9997))
        # must use "0.0.0.0" as defaults other than "127.0.0.1"
        self.host = os.getenv('HOST', "0.0.0.0")
        self.timeout = int(os.getenv("TIMEOUT", 300))
        self.reload = to_bool(os.getenv("RELOAD", False))
        # self.debug = to_bool(os.getenv("DEBUG", False))
        self.debug = False
        self.daemon = to_bool(os.getenv("DAEMON", True))
        self.worker_class = os.getenv("WORKER_CLASS", "uvicorn.workers.UvicornH11Worker")
        self.backlog = int(os.getenv("BACKLOG", 2048))
        self.support_operators = ["add", "delete", "replace", "find"]

        # other configuration
        self.defaults = Defaults()
        self.database = Database()
        self.authority = Authority()
        self.models = Models()

        # log configuration
        self.log_path = osp.join(self.defaults.core_path, "log")
        if not osp.exists(self.log_path):
            os.makedirs(self.log_path, exist_ok=True)
        self.log_file = osp.join(self.log_path, 'face_log.log')
        self.log_request = to_bool(os.getenv("LOG_REQUEST", True))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_distance_top_k = to_bool(os.getenv('LOG_DIS_TOP_K', True))
        self.server_log_path = os.path.join(self.defaults.root_path, "log")
        if not osp.exists(self.server_log_path):
            os.makedirs(self.server_log_path, exist_ok=True)
        self.server_log_file = os.path.join(self.server_log_path, "server_log.log")

    def get_configurations(self):
        config_info = dict()
        config_info.update(self.__dict__)
        # hidden for users
        config_info.pop("database")
        config_info.pop("authority")
        config_info.pop("models")
        config_info.pop("defaults")
        config_info.update(self.defaults.__dict__)
        config_info.update(self.models.__dict__)
        return config_info

    def collect_env_info(self):
        configs_info = self.get_configurations()
        return tabulate(list(configs_info.items()))