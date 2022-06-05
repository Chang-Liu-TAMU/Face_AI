# @Time: 2022/5/25 14:42
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:face_mysql.py

import datetime
import numpy as np
import logging
import mysql.connector
# from ..help_utils.logger_utils import internal_logger

def internal_logger():
    return logging.getLogger(__name__)


class FaceMysql:
    def __init__(self, host='127.0.0.1', port='3306', username='root',
                 password="root", database_name='face', table_name='face_json'):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database_name
        self.table_name = table_name
        self.sql = "replace into {}(features,pic_name,date,state,uname,uid,ugroup) " \
                   "values(%s,%s,%s,%s,%s,%s,%s);".format(self.table_name)
        self.update_sql = "update %s set features='{}',pic_name='{}'," \
                          "date='{}',state='{}',uname='{}',uid='{}',ugroup='{}' where pic_name='{}'" % self.table_name

    def conn_mysql(self):

        db = None

        try:
            db = mysql.connector.connect(host=self.host, port=self.port, user=self.username,
                                         password=self.password, database=self.database_name)
        except Exception as e:
            try:
                internal_logger().warning("Cannot find database named {} and creat it now...".
                                          format(self.database_name))
                db = mysql.connector.connect(host=self.host, port=self.port,
                                             user=self.username, password=self.password)
                cursor = db.cursor()
                cmd_sql = "create database {}".format(self.database_name)
                cursor.execute(cmd_sql)
                internal_logger().info("Create tables named {} now...".format(self.table_name))
                db = mysql.connector.connect(host=self.host, port=self.port, user=self.username,
                                             password=self.password, database=self.database_name)
                import os.path as Path
                sql_path = Path.join(Path.dirname(Path.realpath(__file__)), "database.sql")
                create_tb = str(open(sql_path, 'r', encoding="utf-8").read())
                db.cursor().execute(create_tb)
            except Exception as e:
                internal_logger().warning(e)

        return db

    def remove_faces_by_ids(self, uids):
        db = self.conn_mysql()
        if db is None:
            internal_logger().warning("Unabled to connnect face mysql tables!")
            return "Unabled to connnect face mysql tables!"

        cursor = db.cursor()
        result = {}
        try:
            for uid in set(uids):
                affect_rows = self.remove_by_id(cursor, uid)
                result[uid] = -1 if affect_rows > 0 else 0
            db.commit()
        except Exception as e:
            # Rollback in case there is any error
            db.rollback()
            return str(e)
        finally:
            db.close()
        return result

    async def remove_faces_by_ids_async(self, uids):
        return self.remove_faces_by_ids(uids=uids)

    def remove_by_id(self, cursor, uid):
        mysql_code = "delete from {} where uid='{}'".format(self.table_name, uid)
        cursor.execute(mysql_code)
        return cursor.rowcount

    def remove_duplicated(self, cursor):
        cmd_sql = "delete from {} where Id not in (select Id from (select min(Id) as Id " \
                  "from {} group by pic_name) as p);".format(self.table_name, self.table_name)
        cursor.execute(cmd_sql)

    def find_faces_by_ids(self, u_ids, return_embedding=False):
        result = {}
        for uid in u_ids:
            res = self.find_byuid_facejson(uid)
            if len(res) > 0:
                res_array = np.asarray(res)[:, 1:-1]
                if not return_embedding:
                    res_array = np.delete(res_array, 3, axis=1)
                res = res_array.tolist()
            for item in res:
                if isinstance(item[-1], datetime.datetime):
                    item[-1] = datetime.datetime.strftime(item[-1], '%Y-%m-%d %H:%M:%S')
            info = dict()
            info["match_number"] = len(res)
            info["match_items"] = res
            result[uid] = info
        return result

    def update_by_pic_name(self, cursor, items, pic_name):
        if len(items) != 7:
            return -1
        sql_udate_code = self.update_sql.format(items[0], items[1], items[2], items[3],
                                                items[4], items[5], items[6], pic_name)
        cursor.execute(sql_udate_code)
        return int(cursor.lastrowid)

    def update_facejson(self, pic_names, features, unames, uids, ugroups):
        db = self.conn_mysql()
        if db is None:
            internal_logger().warning("Unabled to connnect face mysql tables!")
            return "Unabled to connnect face mysql tables!"

        cursor = db.cursor()
        # clear cached faces with uids
        self.remove_faces_by_ids(uids)

        try:
            last_id = []
            item_num = len(features)
            dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dates = item_num * [dt]
            states = item_num * ['1']
            zipped = list(zip(features, pic_names, dates, states, unames, uids, ugroups))
            cursor.executemany(self.sql, zipped)
            last_id.append(int(cursor.lastrowid))
            db.commit()
        except Exception as e:
            # Rollback in case there is any error
            db.rollback()
            last_id = str(e)
        finally:
            db.close()
        return last_id

    def update_facejson_deprecated(self, pic_names, features, unames, uids,
                                   ugroups, operator='add', max_faces_per_uid=7):
        db = self.conn_mysql()
        if db is None:
            internal_logger().warning("Unabled to connnect face mysql tables!")
            return "Unabled to connnect face mysql tables!"

        cursor = db.cursor()
        item_num = len(features)

        if operator == 'replace':
            for i in range(item_num):
                self.remove_by_id(cursor, uids[i])

        try:
            last_id = []
            if item_num <= 100:
                for i in range(item_num):
                    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    item = (features[i], pic_names[i], dt, '1', unames[i], uids[i], ugroups[i])
                    target_items = self.find_byuid_facejson(uids[i])
                    if len(target_items) >= max_faces_per_uid and operator == 'add':
                        # sorted by datetime
                        sorted_target_items = sorted(target_items, key=lambda data: data[6])
                        # get the oldest picture name with this uid
                        oldest_pic_name = sorted_target_items[0][5]
                        # update the item with the oldest datetime by pic_name(as the pic_name is unique in database)
                        uid = self.update_by_pic_name(cursor, item, oldest_pic_name)
                    else:
                        cursor.execute(self.sql, item)
                        uid = int(cursor.lastrowid)

                    last_id.append(uid)
            else:  # TODO batch inserting data does not consider replace oldest items
                dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dates = item_num * [dt]
                states = item_num * ['1']
                zipped = list(zip(features, pic_names, dates, states, unames, uids, ugroups))
                cursor.executemany(self.sql, zipped)
                last_id.append(int(cursor.lastrowid))
            db.commit()
        except Exception as e:
            # Rollback in case there is any error
            db.rollback()
            last_id = str(e)
        finally:
            db.close()
        return last_id


    def find_byugroup_facejson(self, ugroup):
        db = self.conn_mysql()
        if db is None:
            internal_logger().warning("Unabled to connnect face mysql tables!")
            return []
        cursor = db.cursor()

        sql = "select * from %s where state=1 and ugroup= '%s' ;" % (self.table_name, ugroup)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
        except:
            internal_logger().warning("Error:unable to fecth data")
        db.close()

    def find_byuid_facejson(self, uid):
        db = self.conn_mysql()
        if db is None:
            internal_logger().warning("Unabled to connnect face mysql tables!")
            return []

        cursor = db.cursor()

        sql = "select * from %s where state=1 and uid= '%s' ;" % (self.table_name, uid)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
        except:
            internal_logger().warning("Error:unable to fecth data")
        db.close()

    def findall_facejson(self):
        db = self.conn_mysql()
        if db is None:
            import time
            n = 0
            while db is None:
                time.sleep(10)
                n += 1
                if db is None and n > 3:
                    break

                internal_logger().warning("[findall_facejson] Try connect to MySql {} time...".format(n))
                db = self.conn_mysql()
        if db is None:
            internal_logger().error("[findall_facejson] Failed to connect to MySql, "
                                    "please start mysql and restart this service again!")
            return []

        cursor = db.cursor()

        sql = "select * from %s where state=1;" % self.table_name
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
        except:
            internal_logger().warning("Error:unable to fecth data")
        db.close()

if __name__ == "__main__":
    db = FaceMysql()
    db_conn = db.conn_mysql()
    print(db_conn)