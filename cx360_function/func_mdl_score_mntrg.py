########################################################################################################################
# 라이브러리 선언 
########################################################################################################################
import pandas as pd
import numpy as np
import sqlite3
import os
import sys
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# print(os.getcwd())

func_dir = f"{os.getcwd()}/data/dev/program/function"
# DB 호출함수
exec(open(f"{func_dir}/func_db_connect2.py"    , encoding='utf-8').read())

################################################################################
## DB 연결
################################################################################

db_dir       = f"{os.getcwd()}/data/dev/source/pine.db"
cur, conn, _ = dbConnect().dbConn("sqlite", db_dir=db_dir)


print("[Log] DB connection Completed")

########################################################################################################################
# 모델 스코어 함수                                                                                     
########################################################################################################################

def func_mdl_score_mntrg(
          list_mdl
        , baseYm   
    ):
    ranges = [f'{i:.1f}이하' for i in np.arange(0.1, 1.1, 0.1)]
    list_score = []    
    for tar, cutoff in list_mdl:
        df_score_interval = pd.DataFrame({'TARGET': [tar] * len(ranges) , '스코어_구간': ranges})

        # 운영_스코어_분포 추출 쿼리
        df_score_result = pd.read_sql(
        f"""
            SELECT
                  기준년월
                , 모델ID
                , TARGET
                , 스코어_구간
                , count(*)                                                       AS 운영_스코어_분포
            FROM (
                SELECT 
                    *
                    , CASE 
                        WHEN Y_Prob <= 0.1                  THEN '0.1이하'
                        WHEN Y_Prob > 0.1 and Y_Prob <= 0.2 THEN '0.2이하'
                        WHEN Y_Prob > 0.2 and Y_Prob <= 0.3 THEN '0.3이하'
                        WHEN Y_Prob > 0.3 and Y_Prob <= 0.4 THEN '0.4이하'
                        WHEN Y_Prob > 0.4 and Y_Prob <= 0.5 THEN '0.5이하'
                        WHEN Y_Prob > 0.5 and Y_Prob <= 0.6 THEN '0.6이하'
                        WHEN Y_Prob > 0.6 and Y_Prob <= 0.7 THEN '0.7이하'       
                        WHEN Y_Prob > 0.7 and Y_Prob <= 0.8 THEN '0.8이하'
                        WHEN Y_Prob > 0.8 and Y_Prob <= 0.9 THEN '0.9이하'                  
                        WHEN Y_Prob > 0.9 and Y_Prob <= 1   THEN '1.0이하'   END AS 스코어_구간
                FROM TBL_CHMP_SCORE_DIST
                WHERE 기준년월 = {baseYm}
                AND TARGET = '{tar}'
            ) A
            GROUP BY 1,2,3,4
        """
        , conn
        )  
        
        # 모델_스코어_구간 병합    
        df_score_result = pd.merge(
              df_score_result
            , df_score_interval
            , how = 'outer'
            , on  = ['스코어_구간', 'TARGET']
        )
    
        # 모델_스코어_구간 NULL처리  
        df_score_result['운영_스코어_분포'] = df_score_result['운영_스코어_분포'].fillna(0)
        df_score_result['기준년월'] = df_score_result['기준년월'].fillna(baseYm)

        # 데이터 합치기
        list_score.append(df_score_result) 

    # 하나의 데이터 프레임으로 병합
    df_score = pd.concat(list_score)
    # print(df_score)
             

    ########################################################################################################################
    # 모델 스코어 데이터 적재                                                                                     
    ########################################################################################################################

    # MDL_SCORE_MONITORING 테이블 미존재 시 테이블 생성
    cur.execute(
        """
        create table if not exists MDL_SCORE_MONITORING(
              기준년월               TEXT
            , 모델ID                 TEXT
            , TARGET                 TEXT    
            , 스코어_구간            TEXT
            , 운영_스코어_분포       NUM
            , PRIMARY KEY (기준년월, 모델ID, TARGET, 스코어_구간)
        )
        """
    )
    cur.execute(f"""delete from MDL_SCORE_MONITORING where 기준년월 = {baseYm}""")
    # 모델_스코어 적재
    df_score.to_sql(
          name      = 'MDL_SCORE_MONITORING'
        , con       = conn2
        , if_exists = 'append'
        , index     = False
        , method    = "multi"
        , chunksize = 10000
    ) 

# if __name__ == '__main__':
    
#     baseYm = '202306'
#     # 모델ID 추출
#     df_mdl_list = pd.read_sql(
#     f"""
#         select TARGET, cutOff
#         from PINE_MDL_CATALOG
#         where 모델상태정보 = '챔피언'
#     """     
#     , con = conn
#     )

#     df_mdl_list = df_mdl_list.values.tolist()
#     print(df_mdl_list)
#     ranges = [f'{i:.1f}이하' for i in np.arange(0.1, 1.1, 0.1)]

#     list_score = []    
#     for tar, cutoff in df_mdl_list:
#         df_score_interval = pd.DataFrame({'TARGET': [tar] * len(ranges) , '스코어_구간': ranges})

#         # 운영_스코어_분포 추출 쿼리
#         df_score_result = pd.read_sql(
#         f"""
#             SELECT
#                   기준년월
#                 , 모델ID
#                 , TARGET
#                 , 스코어_구간
#                 , count(*)                                                       AS 운영_스코어_분포
#             FROM (
#                 SELECT 
#                     *
#                     , CASE 
#                         WHEN Y_Prob <= 0.1                  THEN '0.1이하'
#                         WHEN Y_Prob > 0.1 and Y_Prob <= 0.2 THEN '0.2이하'
#                         WHEN Y_Prob > 0.2 and Y_Prob <= 0.3 THEN '0.3이하'
#                         WHEN Y_Prob > 0.3 and Y_Prob <= 0.4 THEN '0.4이하'
#                         WHEN Y_Prob > 0.4 and Y_Prob <= 0.5 THEN '0.5이하'
#                         WHEN Y_Prob > 0.5 and Y_Prob <= 0.6 THEN '0.6이하'
#                         WHEN Y_Prob > 0.6 and Y_Prob <= 0.7 THEN '0.7이하'       
#                         WHEN Y_Prob > 0.7 and Y_Prob <= 0.8 THEN '0.8이하'
#                         WHEN Y_Prob > 0.8 and Y_Prob <= 0.9 THEN '0.9이하'                  
#                         WHEN Y_Prob > 0.9 and Y_Prob <= 1   THEN '1.0이하'   END AS 스코어_구간
#                 FROM TBL_CHMP_SCORE_DIST
#                 WHERE 기준년월 = {baseYm}
#                 AND TARGET = '{tar}'
#             ) A
#             GROUP BY 1,2,3,4
#         """
#         , conn
#         )  
        
#         # 모델_스코어_구간 병합    
#         df_score_result = pd.merge(
#               df_score_result
#             , df_score_interval
#             , how = 'outer'
#             , on  = ['스코어_구간', 'TARGET']
#         )
    
#         # 모델_스코어_구간 NULL처리  
#         df_score_result['운영_스코어_분포'] = df_score_result['운영_스코어_분포'].fillna(0)
#         df_score_result['기준년월'] = df_score_result['기준년월'].fillna(baseYm)

#         # 데이터 합치기
#         list_score.append(df_score_result) 

#     # 하나의 데이터 프레임으로 병합
#     df_score = pd.concat(list_score)
#     print(df_score)


#     # MDL_SCORE_MONITORING 테이블 미존재 시 테이블 생성
#     cur.execute(
#         """
#         create table if not exists MDL_SCORE_MONITORING(
#               기준년월               TEXT
#             , 모델ID                 TEXT
#             , TARGET                 TEXT    
#             , 스코어_구간            TEXT
#             , 운영_스코어_분포       NUM
#             , PRIMARY KEY (기준년월, 모델ID, TARGET, 스코어_구간)
#         )
#         """
#     )

#     # 모델_스코어 적재
#     df_score.to_sql(
#           name      = 'MDL_SCORE_MONITORING'
#         , con       = conn2
#         , if_exists = 'append'
#         , index     = False
#         , method    = "multi"
#         , chunksize = 10000
#     ) 