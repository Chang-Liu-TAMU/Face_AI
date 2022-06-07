# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/8/7 2:02
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------
from typing import Any
from loguru import logger
from sqlalchemy.orm import Session
from fastapi import Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Header, Body, Query

from apphelper.common import deps
from application.models import sys_auth
from apphelper.utils import settings
from apphelper.utils import DefaultUser
from apphelper.utils.rsa_utils import RSAendecrypt
from apphelper.schemas.response import response_code
from application.models.token_model import TokenModel
from apphelper.schemas.request.sys_user_schema import UserInDB, UserReturn
from apphelper.service.curd_custom import db_create, db_get_token, \
    db_authenticate, db_is_active, db_test_read, db_test_write, db_test_delete

router = APIRouter()

#  only add when in debugging mode
if settings.DEBUG:
    @router.post("/create", summary='用户注册', name="用户注册")
    async def user_create(nickname: str = Query(default=DefaultUser.nickname, description="注册用户昵称"),
                          username: str = Query(default=DefaultUser.username, description="注册用户名"),
                          password: str = Query(default=DefaultUser.password, description="注册密码"),
                          authority_id: int = Query(default=DefaultUser.authority_id, description="权限ID"),
                          db: Session = Depends(deps.get_db)):
        """
        用户创建
        :param nickname:
        :param username:
        :param password:
        :param authority_id:
        :param db:
        :return:
        """
        user_info = {
            "nickname": nickname,
            "username": username,
            "password": password,
            "authority_id": authority_id
        }
        res = db_create(user_info=user_info, db=db)
        if isinstance(res, JSONResponse):
            return res
        if res.data:
            return response_code.resp_200(data=UserReturn(**res.data).dict())
        else:
            raise response_code.resp_4003(message=res.message)

if settings.ENCRYPTION_ENABLED:
    @router.post("/update_key", summary='更新秘钥', name="更新秘钥")
    async def update_key(username: str = Query(..., description="用户名"),
                         password: str = Query(..., description="密码"),
                         db: Session = Depends(deps.get_db)):
        """
        更新秘钥
        :return:
        """
        rsadate = RSAendecrypt(settings.RSA_FILE_PATH)
        try:
            password = rsadate.decrypt(password)
        except Exception as e:
            logger.warning(e.args)

        user_item = db_authenticate(db=db, username=username, password=password)
        if not user_item:
            message = f"Update key failed: 用户认证错误: username: {username} password: {password}"
            logger.warning(message)
            return response_code.resp_4003(message=message)
        elif not db_is_active(user_item):
            message = f"Update key failed: User: {user_item.username} not activated"
            logger.warning(message)
            return response_code.resp_4003(message=message)

        try:
            rsadate.generate_key()
            updated_password = rsadate.encrypt(password)
        except Exception as e:
            message = f"Update key failed: {e.args}"
            logger.warning(message)
            raise response_code.resp_4003(message=message)
        else:
            return response_code.resp_200(data={"username": username, "password": updated_password})


@router.post("/login/access-token", summary="用户登录认证", name="登录")
async def login_access_token(
        username: str = Query(..., description="用户名"),
        password: str = Query(..., description="密码"),
        db: Session = Depends(deps.get_db)
) -> Any:
    """
    用户JWT登录
    :param username:
    :param password:
    :param db:
    :return:
    """

    # decrypt password if necessary
    print("checking encrption")
    if settings.ENCRYPTION_ENABLED:
        try:
            rsadate = RSAendecrypt(settings.RSA_FILE_PATH)
            password = rsadate.decrypt(password)
        except Exception as e:
            print("I am logging")
            logger.warning(e.args)
    print("db get token")
    print("password: ", password)
    # get token by given username and password
    res_token = db_get_token(username=username, password=password, db=db)
    if isinstance(res_token, JSONResponse):
        return res_token
    print("db get token finished")
    if res_token.data:
        token_model = TokenModel(token=res_token.data['token'], token_type=res_token.data["token_type"])
        return response_code.resp_200(data=token_model.dict())
    else:
        raise response_code.resp_4001(message=res_token.message)


@router.get("/user/info", summary="获取用户信息", name="获取用户信息", description="此API没有验证权限")
async def get_user_info(current_user: sys_auth.SysUser = Depends(deps.get_current_user)) -> Any:
    return response_code.resp_200(data=current_user.__dict__)


@router.get("/home", summary='用户查看', name="查看用户信息", description="此API没有验证权限")
async def user_home(authorization_: str = Query(..., description="token验证"),
                    db: Session = Depends(deps.get_db)):
    res = db_test_read(token=authorization_, db=db)
    if isinstance(res, JSONResponse):
        return res
    if res.data:
        return response_code.resp_200(data=UserInDB(**res.data).dict())
    else:
        raise response_code.resp_4001(message=res.message)


@router.post("/update", summary='用户修改', name="修改用户信息", description="此API没有验证权限")
async def user_update(authorization_: str = Query(..., description="token验证"),
                      data: dict = Body(..., example={"nickname": "dummy"}),
                      db: Session = Depends(deps.get_db)):
    res = db_test_write(token=authorization_, data=data, db=db)
    if isinstance(res, JSONResponse):
        return res
    if res.data:
        return response_code.resp_200(data=UserReturn(**res.data).dict())
    else:
        raise response_code.resp_4001(message=res.message)


@router.delete("/delete", summary='用户注销', name="删除用户信息", description="此API没有验证权限")
async def user_delete(authorization_: str = Query(..., description="token验证"),
                      db: Session = Depends(deps.get_db)):
    res = db_test_delete(token=authorization_, db=db)
    if isinstance(res, JSONResponse):
        return res
    if res.data:
        return response_code.resp_200(message=res.message)
    else:
        raise response_code.resp_4001(message=res.message)
