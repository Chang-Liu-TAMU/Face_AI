# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/8/7 2:09
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------
from loguru import logger
from typing import Optional
from fastapi import Depends
from datetime import timedelta
from sqlalchemy.orm import Session

from apphelper.common import deps
from apphelper.common import custom_exc
from application.models.response import CRUD
from apphelper.utils import settings
from apphelper.service import security
from apphelper.service.sys_user import curd_user
from application.models.sys_auth import SysUser
from apphelper.schemas.response import response_code
from apphelper.schemas.request import sys_user_schema


def get_response(user_item: SysUser):
    return CRUD(data={
        "user_id": user_item.user_id,
        "nickname": user_item.nickname,
        "username": user_item.username,
        "hashed_password": user_item.hashed_password,
        "is_active": user_item.is_active,
        "authority_id": user_item.authority_id,
    })


def db_create(user_info: dict, db: Session):
    """
    - 对password进行了哈希存储
    - 为scopes增加了默认赋值（权限）
    - 为_id做了逻辑判断（mongodb）
    :param kwargs:
    :return:
    """
    if "username" not in user_info:
        return response_code.resp_500(message="missing fields")
    if deps.find_user_by_username(username=user_info["username"], db=db):
        return response_code.resp_5002(message="user existed")

    user_in = sys_user_schema.UserCreate(
        nickname=user_info["nickname"],
        username=user_info["username"],
        password=user_info["password"],
        authority_id=user_info["authority_id"],  # 权限id
    )
    user_item = curd_user.create(db, obj_in=user_in)  # noqa: F841
    return get_response(user_item=user_item)


def db_authenticate(username: str, password: str, db: Session) -> Optional[SysUser]:
    """
    :param username:
    :param password:
    :return:
    """
    return curd_user.authenticate(db=db, username=username, password=password)


def db_is_active(user_item: SysUser) -> bool:
    return curd_user.is_active(user_item)


def db_get_token(username: str, password: str, db: Session):
    """
    有了获取token这个环节，其实登录就等于获取token+读取user信息两步

    所以在数据库环节没必要多写一个登录函数，在前端写即可

    此外，基于token的认证，在数据库环节也无需多写一个基于id查找用户的函数

    :param username:
    :param password:
    :return:
    """
    user_item = db_authenticate(db=db, username=username, password=password)
    if not user_item:
        message = f"用户认证错误: username: {username} password: {password}"
        logger.warning(message)
        return response_code.resp_4003(message=message)
    elif not db_is_active(user_item):
        message = f"User: {user_item.username} not activated"
        logger.warning(message)
        return response_code.resp_4003(message=message)

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    # 登录token 存储了user.id 和 authority_id
    return CRUD(data={"token": security.create_access_token(user_item.user_id,
                                                            str(user_item.authority_id),
                                                            expires_delta=access_token_expires),
                      "token_type": settings.TOKEN_TYPE})


def db_test_read(token: str, db: Session):
    try:
        user_item = deps.get_current_user(db=db, token=deps.check_jwt_token(token))
    except Exception as e:
        if isinstance(e, custom_exc.TokenAuthError):
            return response_code.resp_4003(message=e.err_desc)
        elif isinstance(e, custom_exc.TokenExpired):
            return response_code.resp_4002(message=e.err_desc)
        else:
            return response_code.resp_500(message=str(e.args))
    else:
        return get_response(user_item=user_item)


def db_test_write(token: str, data: dict, db: Session):
    try:
        user_item = deps.update_user(data=data, db=db, token=deps.check_jwt_token(token))
    except Exception as e:
        if isinstance(e, custom_exc.TokenAuthError):
            return response_code.resp_4003(message=e.err_desc)
        elif isinstance(e, custom_exc.TokenExpired):
            return response_code.resp_4002(message=e.err_desc)
        else:
            return response_code.resp_500(message=str(e.args))
    else:
        return get_response(user_item=user_item)


def db_test_delete(token: str, db: Session):
    try:
        user_item = deps.remove_user(db=db, token=deps.check_jwt_token(token))
    except Exception as e:
        if isinstance(e, custom_exc.TokenAuthError):
            return response_code.resp_4003(message=e.err_desc)
        elif isinstance(e, custom_exc.TokenExpired):
            return response_code.resp_4002(message=e.err_desc)
        else:
            return response_code.resp_500(message=str(e.args))
    else:
        return CRUD(message=f"successfully deleted user: {user_item.username}")
