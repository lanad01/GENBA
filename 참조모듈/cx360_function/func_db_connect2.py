########################################################################################################################
# 라이브러리 선언                                                                                   
########################################################################################################################

import sqlite3
from sqlalchemy import create_engine
import mysql.connector
import psycopg2
import os
from urllib.parse import quote_plus
import pandas as pd

########################################################################################################################
# DB 연결 Class                                                                               
########################################################################################################################

class dbConnect():
    def __init__(self):
        # DBMS 종류에 따른 연결 문자열 패턴을 저장
        self.engine_str = {
            'sqlite': "sqlite:///{db_dir}",
            'mysql': "mysql+mysqlconnector://{username}:{password}@{host}:{port}/{dbname}",
            'postgresql': "postgresql://{username}:{password}@{host}:{port}/{dbname}"
        }

    def dbConn(self, db_type, **kwargs):
        if db_type not in self.engine_str:
            raise ValueError(f"Unsupported database type: {db_type}")

        if db_type == "sqlite":
            if "db_dir" not in kwargs:
                raise ValueError("For sqlite, 'db_dir' parameter is required.")

            conn = sqlite3.connect(kwargs['db_dir'], isolation_level=None)
            cur = conn.cursor()
            engine = create_engine(self.engine_str[db_type].format(db_dir=kwargs['db_dir']))
        
        else :
            required_args = ["username", "password", "host", "port", "dbname"]

            if not all(arg in kwargs for arg in required_args):
                raise ValueError(f"For {db_type}, the following parameters are required: {', '.join(required_args)}")

            kwargs['username'] = quote_plus(kwargs['username'])
            kwargs['password'] = quote_plus(kwargs['password'])

            engine = create_engine(self.engine_str[db_type].format(**kwargs))
            conn = engine.connect()
            cur = conn.execution_options(autocommit=True).execute

        return (cur, conn, engine)

# 예제 사용 방법:
# sqlite_conn = dbConnect().dbConn("sqlite", db_dir="your_sqlite_db_path")
# mysql_conn = dbConnect().dbConn("mysql", username="user", password="pass", host="localhost", port=3306, dbname="testdb")
# postgres_conn = dbConnect().dbConn("postgresql", username="user", password="pass", host="localhost", port=5432, dbname="testdb")

########################################################################################################################
# DB 연결 Test
########################################################################################################################

# if __name__ == '__main__':
#     # print(f'{os.getcwd()}/data/dev/source/pine.db')
#     # SQLite 연결 테스트
#     try:
#         sqlite_conn = dbConnect().dbConn("sqlite", db_dir=f'{os.getcwd()}/data/dev/source/pine.db')
#         print("SQLite 연결 성공!")
#     except Exception as e:
#         print(f"SQLite 연결 실패: {e}")

#     # MySQL 연결 테스트
#     try:
#         mysql_conn = dbConnect().dbConn("mysql", username="root", password="dhdudgh123!@#", host="localhost", port=3306, dbname="pinedb")
#         print("MySQL 연결 성공!")

#     except Exception as e:
#         print(f"MySQL 연결 실패: {e}")
    
#     # PostgreSQL 연결 테스트
#     try:
#         postgres_conn = dbConnect().dbConn("postgresql", username="postgres", password="postgres", host="localhost", port=5432, dbname="postgres")
#         print("PostgreSQL 연결 성공!")
#     except Exception as e:
#         print(f"PostgreSQL 연결 실패: {e}")
    
    