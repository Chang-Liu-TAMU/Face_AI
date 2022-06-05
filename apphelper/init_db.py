#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/16 14:33
# @Author  : CoderCharm
# @File    : create_user.py
# @Software: PyCharm
# @Github  : github/CoderCharm
# @Email   : wg_python@163.com
# @Desc    :
"""
初始化数据库角色
"""

import os
import sys
import os.path as osp
import mysql.connector
from loguru import logger

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, dir_path)

from apphelper.utils import settings
from apphelper.utils.rsa_utils import RSAendecrypt
from apphelper.utils.sql_utils import update_table
from apphelper.service.sys_user import curd_user
from application.router.v1_router import api_v1_router
from apphelper.common.sys_casbin import get_casbin
from apphelper.database.session import SessionLocal
from apphelper.service.sys_authority import curd_authority
from apphelper.schemas.request import sys_user_schema, sys_authority_schema


def init_authority(db: SessionLocal) -> None:
    """
    创建角色
        999 为超级管理员用户
        100 为普通用户
    :param db:
    :return:
    """
    authority_info_list = [
        {"authority_id": 999, "authority_name": "超级管理员", "parent_id": 0},
        {"authority_id": 100, "authority_name": "普通用户", "parent_id": 999},
    ]

    for authority_info in authority_info_list:
        authority_in = sys_authority_schema.AuthorityCreate(
            authority_id=authority_info["authority_id"],
            authority_name=authority_info["authority_name"],
            parent_id=authority_info["parent_id"],
        )
        authority = curd_authority.create(db, obj_in=authority_in)  # noqa: F841
        if authority.parent_id == -1:
            print(f"角色-{authority.authority_name}-创建失败,角色已经存在")
        else:
            print(f"角色-{authority.authority_name}-创建成功")


def init_user(db: SessionLocal) -> None:
    """
    创建基础的用户
    :param db:
    :return:
    """

    if settings.ENCRYPTION_ENABLED:
        rsadate = RSAendecrypt(settings.RSA_FILE_PATH)
        # 生成公钥和私钥
        if not rsadate.has_key_file():
            rsadate.generate_key()

    for user_info in settings.USER_INFO_LIST:
        user_in = sys_user_schema.UserCreate(
            nickname=user_info["nickname"],
            username=user_info["username"],
            password=user_info["password"],
            authority_id=user_info["authority_id"],  # 权限id
        )
        user = curd_user.create(db, obj_in=user_in)  # noqa: F841
        if settings.ENCRYPTION_ENABLED:
            encrypted_passwd = rsadate.encrypt(user_info["password"])
        else:
            encrypted_passwd = user_info["password"]
        if user.is_active == 0:
            print(f"用户-{user.nickname}-创建失败 用户已存在,更新用户名: {user.username} 密码: {encrypted_passwd}")
        else:
            print(f"用户-{user.nickname}-创建成功 设置新用户名: {user.username} 密码: {encrypted_passwd}")


def init_casbin():
    """
    初始化casbin的基本API数据

    把 api_v1_router 分组的所有路由都添加到 casbin里面
    :return:
    """
    e = get_casbin()

    for route in api_v1_router.routes:
        if route.name == "登录":
            # 登录不验证权限
            continue
        for method in route.methods:
            # 添加casbin规则
            e.add_policy("999", route.path, method)


def update_tables(db_handler, reset=False):
    sql_path = osp.join(osp.dirname(osp.realpath(__file__)), "sql")

    logger.info("Create tables named {} now...".format(settings.API_TABLE))
    sys_api_sql_file = osp.join(sql_path, "{}.sql".format(settings.API_TABLE))
    update_table(db_handler=db_handler, sql_file=sys_api_sql_file, reset=reset)

    logger.info("Create tables named {} now...".format(settings.AUTHORITY_TABLE))
    sys_authorities_sql_file = osp.join(sql_path, "{}.sql".format(settings.AUTHORITY_TABLE))
    update_table(db_handler=db_handler, sql_file=sys_authorities_sql_file, reset=reset)

    logger.info("Create tables named {} now...".format(settings.USER_TABLE))
    sys_user_sql_file = osp.join(sql_path, "{}.sql".format(settings.USER_TABLE))
    update_table(db_handler=db_handler, sql_file=sys_user_sql_file, reset=reset)

    try:
        db_handler.commit()
    except Exception as e:
        print(e.args)
        db_handler.rollback()

    db_handler.close()


def init_database(reset=False):
    try:
        db_handler = mysql.connector.connect(host=settings.MYSQL_HOST,
                                             port=settings.MYSQL_PORT,
                                             user=settings.MYSQL_USERNAME,
                                             password=settings.MYSQL_PASSWORD,
                                             database=settings.MYSQL_DATABASE)
    except Exception as e:
        try:
            logger.warning("Cannot find database named {} and creat it now...".
                           format(settings.MYSQL_DATABASE))
            db_handler = mysql.connector.connect(host=settings.MYSQL_HOST,
                                                 port=settings.MYSQL_PORT,
                                                 user=settings.MYSQL_USERNAME,
                                                 password=settings.MYSQL_PASSWORD)
            cmd_sql = "create database {}".format(settings.MYSQL_DATABASE)
            db_handler.cursor().execute(cmd_sql)
            db_handler = mysql.connector.connect(host=settings.MYSQL_HOST,
                                                 port=settings.MYSQL_PORT,
                                                 user=settings.MYSQL_USERNAME,
                                                 password=settings.MYSQL_PASSWORD,
                                                 database=settings.MYSQL_DATABASE)
            update_tables(db_handler, reset=reset)
        except Exception as e:
            logger.warning(e)
    else:
        update_tables(db_handler, reset=reset)


is_reset = False
db = SessionLocal()
init_database(reset=is_reset)
init_authority(db)
init_user(db)
init_casbin()
