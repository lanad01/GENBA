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
# 모델 성능지표 함수                                                                                     
########################################################################################################################

def func_mdl_prfmnc_mntrg(
          list_mdl
        , baseYm   
    ):
        
    list_prfmnc = []
    for idx, (tar, cutoff) in enumerate(list_mdl, start = 1):
        print(f'[LOG] TARGET = {tar}, 순서 = {idx} / {len(list_mdl)}')
        
        # 운영_성능지표_분포 추출 쿼리
        df_prfmnc_result = pd.read_sql(
        f"""
            SELECT 
                  기준년월
                , 모델ID
                , TARGET
                , 정밀도
                , 재현율
                , F1Score
            FROM TBL_CHMP_MTC
            WHERE 
                 TARGET  = '{tar}'
            AND 기준년월 = {baseYm}
            AND cutOff   = {cutoff}
        """
        , con = conn
        )  
        
        # 성능_재학습_여부 컬럼 추가
        df_prfmnc_result["성능_재학습_여부"] = np.where(
            df_prfmnc_result["F1Score"] >= 0.9, "유지", np.where(
                df_prfmnc_result["F1Score"] < 0.8, "재학습", "주의"
            )
        )
        
        # 데이터 합치기
        list_prfmnc.append(df_prfmnc_result) 
    # 하나의 데이터 프레임으로 병합
    df_prfmnc = pd.concat(list_prfmnc) 
    print(df_prfmnc)
    ########################################################################################################################
    # 모델 성능지표 데이터 적재                                                                                     
    ########################################################################################################################

    # MDL_PRFMNC_MONITORING 테이블 미존재 시 테이블 생성
    cur.execute(
        """
        create table if not exists MDL_PRFMNC_MONITORING (
              기준년월              text
            , 모델ID                text
            , TARGET                text
            , 정밀도                real
            , 재현율                real
            , F1Score               real
            , 성능_재학습_여부      text
            , PRIMARY KEY(기준년월, TARGET)
            )
        """
    )
    cur.execute(f"""delete from MDL_PRFMNC_MONITORING where 기준년월 = {baseYm}""")
    # 모델_스코어 적재
    df_prfmnc.to_sql(
          name      = 'MDL_PRFMNC_MONITORING'
        , con       = conn2
        , if_exists = 'append'
        , index     = False
        , method    = "multi"
        , chunksize = 10000
    )

# if __name__ == '__main__':
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
#     baseYm = '201102'
    
#     list_prfmnc = []
#     for idx, (tar, cutoff) in enumerate(df_mdl_list, start = 1):
#         print(f'[LOG] TARGET = {tar}, 순서 = {idx} / {len(df_mdl_list)}')
        
#         # 운영_성능지표_분포 추출 쿼리
#         df_prfmnc_result = pd.read_sql(
#         f"""
#             SELECT 
#                   기준년월
#                 , 모델ID
#                 , TARGET
#                 , 정밀도
#                 , 재현율
#                 , F1Score
#             FROM TBL_CHMP_MTC
#             WHERE 
#                  TARGET = '{tar}'
#             AND 기준년월 = {baseYm}
#             AND cutOff  = {cutoff}
#         """
#         , con = conn
#         )  
        
#         # 성능_재학습_여부 컬럼 추가
#         df_prfmnc_result["성능_재학습_여부"] = np.where(
#             df_prfmnc_result["F1Score"] >= 0.9, "유지", np.where(
#                 df_prfmnc_result["F1Score"] < 0.8, "재학습", "주의"
#             )
#         )
        
#         # 데이터 합치기
#         list_prfmnc.append(df_prfmnc_result) 
#     # 하나의 데이터 프레임으로 병합
#     df_prfmnc = pd.concat(list_prfmnc)
#     print(df_prfmnc)