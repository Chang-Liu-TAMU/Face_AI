#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/16 11:51
# @Author  : CoderCharm
# @File    : deps.py
# @Software: PyCharm
# @Github  : github/CoderCharm
# @Email   : wg_python@163.com
# @Desc    :
"""

一些通用的依赖功能

"""
from typing import Generator, Any, Union, Optional, Dict

import jose
from jose import jwt
from fastapi import Header, Depends, Request, Query
from sqlalchemy.orm import Session
from pydantic import ValidationError

from apphelper.database.session import SessionLocal
from apphelper.common import custom_exc, sys_casbin
from application.models.sys_auth import SysUser
from apphelper.utils import settings
from apphelper.service.sys_user import curd_user


def get_db() -> Generator:
    """
    获取sqlalchemy会话对象
    :return:
    """
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


def check_jwt_token(
        token: Optional[str] = Query(..., description="登录token")
) -> Union[str, Any]:
    """
    解析验证token  默认验证headers里面为token字段的数据
    可以给 headers 里面token替换别名, 以下示例为 X-Token
    token: Optional[str] = Header(None, alias="X-Token")
    :param token:
    :return:
    """

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SK, algorithms=[settings.JWT_ALGO]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise custom_exc.TokenExpired()
    except (jwt.JWTError, jose.exceptions.JWSError, ValidationError, UnicodeDecodeError, AttributeError):
        raise custom_exc.TokenAuthError()


def get_current_user(
        db: Session = Depends(get_db),
        token: Optional[str] = Depends(check_jwt_token)
) -> SysUser:
    """
    根据header中token 获取当前用户
    :param db:
    :param token:
    :return:
    """
    user = curd_user.get(db, user_id=token.get("sub"))
    if not user:
        raise custom_exc.TokenAuthError(err_desc="User not found")
    return user


def find_user_by_id(
        user_id: Any,
        db: Session = Depends(get_db),
) -> bool:
    """
    根据header中token和data更新当前用户
    :param data:
    :param db:
    :param token:
    :return:
    """
    user = curd_user.get(db, user_id=user_id)
    return user


def find_user_by_username(
        username: Any,
        db: Session = Depends(get_db),
) -> bool:
    """
    根据header中token和data更新当前用户
    :param data:
    :param db:
    :param token:
    :return:
    """
    user = curd_user.get_user_by_name(db=db, username=username)
    return user


def update_user(
        data: Dict[str, Any],
        db: Session = Depends(get_db),
        token: Optional[str] = Depends(check_jwt_token)
) -> SysUser:
    """
    根据header中token和data更新当前用户
    :param data:
    :param db:
    :param token:
    :return:
    """
    current_user = get_current_user(db=db, token=token)
    updated_user = curd_user.update(db=db, db_obj=current_user, obj_in=data)
    if not updated_user:
        raise custom_exc.TokenAuthError(err_desc="User not found")
    return updated_user


def remove_user(
        db: Session = Depends(get_db),
        token: Optional[str] = Depends(check_jwt_token)
) -> SysUser:
    """
    根据header中token和data更新当前用户
    :param data:
    :param db:
    :param token:
    :return:
    """

    to_be_deleted_user = get_current_user(db=db, token=token)
    flag = curd_user.remove(db=db, user_id=to_be_deleted_user.user_id)
    if not flag:
        raise custom_exc.TokenAuthError(err_desc="User not found")

    return to_be_deleted_user


def check_authority(
        request: Request,
        token: Optional[str] = Depends(check_jwt_token)
):
    """
    权限验证 依赖于 JWT token
    :param request:
    :param token:
    :return:
    """
    authority_id = token.get("authority_id")
    path = request.url.path
    method = request.method

    e = sys_casbin.get_casbin()
    if not e.enforce(str(authority_id), path, method):
        # 注意 字段类型都是字符串
        # 根据token中的 authority_id  请求路径  方法 判断路径
        raise custom_exc.AuthenticationError()
