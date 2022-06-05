#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/16 13:43
# @Author  : CoderCharm
# @File    : sys_user_schema.py
# @Software: PyCharm
# @Github  : github/CoderCharm
# @Email   : wg_python@163.com
# @Desc    :
"""
管理员表的 字段model模型 验证 响应(没写)等
"""

from typing import Optional
from pydantic import BaseModel


# Shared properties
class UserBase(BaseModel):
    is_active: Optional[bool] = True


class UserAuth(BaseModel):
    username: str


class UserLogin(UserAuth):
    password: str


# 手机号登录认证 验证数据字段都叫username
class UserPhoneAuth(UserAuth):
    password: int


# 创建账号需要验证的条件
class UserCreate(UserBase):
    username: str
    nickname: str
    password: str
    authority_id: int = 1


class UserReturn(UserBase, UserAuth):
    user_id: str
    nickname: str
    hashed_password: str


# Properties to receive via API on update
class UserUpdate(UserBase):
    password: Optional[str] = None


class UserInDBBase(UserBase):
    user_id: Optional[str] = None
    nickname: Optional[str] = None

    class Config:
        orm_mode = True


class UserInDB(UserInDBBase):
    hashed_password: str


# 返回的用户信息
class UserInfo(BaseModel):
    role_id: int
    role: str
    nickname: str
    username: str
