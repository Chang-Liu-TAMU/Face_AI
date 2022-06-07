#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/16 13:41
# @Author  : CoderCharm
# @File    : user.py
# @Software: PyCharm
# @Github  : github/CoderCharm
# @Email   : wg_python@163.com
# @Desc    :
"""
管理员用户表的CRUD
"""

from typing import Optional
from loguru import logger
from sqlalchemy.orm import Session

from .security import get_password_hash, verify_password
from .curd_base import CRUDBase
from application.models.sys_auth import SysUser
from apphelper.schemas.request import sys_user_schema


class CRUDUser(CRUDBase[SysUser, sys_user_schema.UserCreate, sys_user_schema.UserUpdate]):

    @staticmethod
    def get_by_username(db: Session, *, username: str) -> Optional[SysUser]:
        """
        通过email获取用户
        参数里面的* 表示 后面调用的时候 要用指定参数的方法调用
        正确调用方式
            curd_user.get_by_username(db, username="xxx")
        错误调用方式
            curd_user.get_by_username(db, "xxx")
        :param db:
        :param username:
        :return:
        """
        return db.query(SysUser).filter(SysUser.username == username).first()


    def create(self, db: Session, *, obj_in: sys_user_schema.UserCreate) -> SysUser:
        db_obj = SysUser(
            username=obj_in.username,
            nickname=obj_in.nickname,
            hashed_password=get_password_hash(obj_in.password),
            authority_id=obj_in.authority_id,
            is_active=obj_in.is_active
        )

        try:
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
        except Exception as e:
            db.rollback()
            db_obj.is_active = 0
            logger.warning(e.args)
        return db_obj

    def authenticate(self, db: Session, *, username: str, password: str) -> Optional[SysUser]:
        user = self.get_by_username(db, username=username)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    @staticmethod
    def is_active(user: SysUser) -> bool:
        return user.is_active == 1


curd_user = CRUDUser(SysUser)
