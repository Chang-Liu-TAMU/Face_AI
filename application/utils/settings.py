# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/8/7 1:52
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------
import os
from typing import Optional
from pydantic import BaseSettings
from core.env_parser import EnvConfigs


class DefaultUser:
    nickname = "default"
    username = 'default'
    password = '123456'
    authority_id = 100


class Settings(BaseSettings):
    configs = EnvConfigs()

    # version and path configuration
    VERSION: str = configs.version
    BASE_PATH: str = configs.defaults.root_path
    RSA_FILE_PATH: str = os.path.join(BASE_PATH, "rsa")

    # log
    LOG_FILE: str = configs.log_file
    LOG_REQUEST: bool = configs.log_request

    # 开发模式配置
    DEBUG: bool = configs.develop
    # 项目文档
    TITLE: str = "Face-AI"
    DESCRIPTION: str = "FastAPI wrapper for Face AI."

    # 文档地址 默认为docs
    DOCS_URL: Optional[str] = "/docs"
    # 文档关联请求数据接口
    OPENAPI_URL: Optional[str] = "/openapi.json"
    # redoc 文档
    REDOC_URL: Optional[str] = "/redoc"

    # swagger_ui_oauth2_redirect_url
    OAUTH2_REDIRECT_URL: Optional[str] = "/docs/oauth2-redirect"

    # authority
    ENCRYPTION_ENABLED: bool = configs.authority.encryption_enabled
    TOKEN_TYPE: str = configs.authority.token_type
    ACCESS_TOKEN_EXPIRE_MINUTES: int = configs.authority.token_expire_minutes  # minutes, a week
    # openssl rand --hex 32 --> token
    JWT_SK: str = configs.authority.secret_key
    JWT_ALGO: str = configs.authority.secret_algorithm

    # registered users
    USER_INFO_LIST = [
        {"nickname": configs.authority.admin_name,
         "username": configs.authority.admin_name,
         "password": configs.authority.admin_passwd,
         "authority_id": 999
         },
        {"nickname": "测试用户",
         "username": "测试用户",
         "password": "test",
         "authority_id": 100
         },
    ]

    # MySql配置
    MYSQL_USERNAME: str = configs.database.db_user
    MYSQL_PASSWORD: str = configs.database.db_password
    MYSQL_HOST: str = configs.database.db_host
    MYSQL_PORT: int = configs.database.db_port
    MYSQL_DATABASE: str = configs.database.user_db_name
    API_TABLE: str = configs.database.db_api_table
    USER_TABLE: str = configs.database.db_user_table
    AUTHORITY_TABLE: str = configs.database.db_authority_table

    # MySql地址
    SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{MYSQL_USERNAME}:{MYSQL_PASSWORD}@" \
                              f"{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"

    REDIS_ENABLED: bool = configs.database.redis_enabled
    # redis配置
    REDIS_HOST: str = configs.database.redis_host
    REDIS_PASSWORD: str = configs.database.redis_passwd
    REDIS_DB: int = configs.database.redis_db
    REDIS_PORT: int = configs.database.redis_port
    REDIS_URL: str = configs.database.redis_url
    REDIS_TIMEOUT: int = configs.database.redis_timeout  # redis连接超时时间

    CASBIN_MODEL_PATH: str = f"{configs.defaults.root_path}/apphelper/resource/rbac_model.conf"


settings = Settings()

