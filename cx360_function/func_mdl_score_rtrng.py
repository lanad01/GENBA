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
# 모델 재학습 함수                                                                                     
########################################################################################################################

def func_mdl_score_rtrng(
          df_mdl_list
        , baseYm
    ):

    baseYm_bfr_1m = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months =  1), '%Y%m')
    df_results = []
    for tar, cutoff in df_mdl_list:
        # 1개월 전_스코어 추출 쿼리
        df_bf_score = pd.read_sql(
        f"""
            SELECT 
                  기준년월                        AS bf_기준년월
                -- , 모델ID
                , TARGET
                , 스코어_구간
                , 운영_스코어_분포                AS 기존_스코어_분포
            FROM MDL_SCORE_MONITORING
            WHERE 
                기준년월 = {baseYm_bfr_1m}
            AND TARGET   = '{tar}'
        """
        , con=conn)

        # 값이 없는 경우에 대한 처리
        if df_bf_score.empty:

            # 기준년월이 없어 데이터가 없는 경우, test_result 결과 가져오기
            df_bf_score = pd.read_sql(
            f"""
                SELECT 
                      기준년월                    AS bf_기준년월
                    -- , 모델ID                      
                    , TARGET                      
                    , 스코어_구간                 
                    , 운영_스코어_분포            AS 기존_스코어_분포
                FROM MDL_SCORE_MONITORING
                WHERE 
                    기준년월 = {baseYm}
                AND TARGET   = '{tar}'
            """
            , con=conn)
            
        # 현재_스코어_분포 추출 쿼리
        df_rct_score = pd.read_sql(
        f"""
            SELECT 
                  기준년월
                , 모델ID
                , TARGET
                , 스코어_구간
                , 운영_스코어_분포                AS 현재_스코어_분포
            FROM MDL_SCORE_MONITORING
            WHERE 
                기준년월 = {baseYm}
            AND TARGET   = '{tar}'
            """
            , con=conn)
            
        # 스코어_구간 병합 & 필요 컬럼 추출
        df_score_cnt = pd.merge(
            df_bf_score,
            df_rct_score,
            how='left',
            on=['스코어_구간', 'TARGET']
        )
        
        df_score_cnt = df_score_cnt.loc[:, ['기준년월', '모델ID', 'TARGET', '스코어_구간', '기존_스코어_분포', '현재_스코어_분포']]
                    
        # 데이터 합치기
        df_results.append(df_score_cnt)   
        
    # 하나의 데이터프레임으로 병합
    df_results = pd.concat(df_results)
    
    # psi산출_분포 비율(모델ID별로 PSI산출)
    df_results['기존_스코어_비율'] = df_results['기존_스코어_분포'] / df_results['기존_스코어_분포'].sum()
    df_results['현재_스코어_비율'] = df_results['현재_스코어_분포'] / df_results['현재_스코어_분포'].sum()
    df_results['로그비율']         = np.log(df_results['현재_스코어_비율'] / df_results['기존_스코어_비율'])
        
    # psi산출(모델ID별로 PSI산출)                   
    df_results['psi값']            = df_results['로그비율'] * (df_results['현재_스코어_비율'] - df_results['기존_스코어_비율'])  
    df_psi_result = df_results.groupby(['기준년월', '모델ID', 'TARGET'])['psi값'].sum().reset_index()  
    # 스코어_재학습_여부 컬럼 초기화
    df_psi_result['스코어_재학습_여부'] = np.where(
        df_psi_result['psi값']  <= 0.25, '유지', np.where(
            df_psi_result['psi값']  > 0.5, '재학습', '주의'
        )
    )

    ########################################################################################################################
    # 모델 스코어 데이터 적재                                                                                     
    ########################################################################################################################

    # MDL_SCORE_RETRAINING 테이블 미존재 시 테이블 생성
    cur.execute(
        """
        create table if not exists MDL_SCORE_RETRAINING (
              기준년월               text
            , 모델ID                 text
            , TARGET                 text
            , psi값                  real
            , 스코어_재학습_여부     text
            , PRIMARY KEY (기준년월, 모델ID, TARGET)
        )
        """
    )
    cur.execute(f"""delete from MDL_SCORE_RETRAINING where 기준년월 = {baseYm}""")
    # 모델_스코어 적재
    df_psi_result.to_sql(
          name      = 'MDL_SCORE_RETRAINING'
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

#     baseYm_bfr_1m = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months =  1), '%Y%m')
#     df_results = []
#     for tar, cutoff in df_mdl_list:
#         # 1개월 전_스코어 추출 쿼리
#         df_bf_score = pd.read_sql(
#         f"""
#             SELECT 
#                   기준년월
#                 , 모델ID
#                 , TARGET
#                 , 스코어_구간
#                 , 운영_스코어_분포                AS 기존_스코어_분포
#             FROM MDL_SCORE_MONITORING
#             WHERE 
#                 기준년월 = {baseYm_bfr_1m}
#             AND TARGET   = '{tar}'
#         """
#         , con=conn)

#         # 값이 없는 경우에 대한 처리
#         if df_bf_score.empty:
#             # 기준년월이 없어 데이터가 없는 경우, test_result 결과 가져오기
#             df_bf_score = pd.read_sql(
#             f"""
#                 SELECT 
#                       기준년월                    AS bf_기준년월
#                     -- , 모델ID                      
#                     , TARGET                      
#                     , 스코어_구간                 
#                     , 운영_스코어_분포            AS 기존_스코어_분포
#                 FROM MDL_SCORE_MONITORING
#                 WHERE 
#                     기준년월 = {baseYm}
#                 AND TARGET   = '{tar}'
#             """
#             , con=conn)
            
#         # 현재_스코어_분포 추출 쿼리
#         # df_rct_score = pd.DataFrame()
#         df_rct_score = pd.read_sql(
#         f"""
#             SELECT 
#                   기준년월
#                 , 모델ID
#                 , TARGET
#                 , 스코어_구간
#                 , 운영_스코어_분포                AS 현재_스코어_분포
#             FROM MDL_SCORE_MONITORING
#             WHERE 
#                 기준년월 = {baseYm}
#             AND TARGET   = '{tar}'
#             """
#             , con=conn)
            
#         # 스코어_구간 병합 & 필요 컬럼 추출
#         df_score_cnt = pd.merge(
#             df_bf_score,
#             df_rct_score,
#             how='left',
#             on=['스코어_구간', 'TARGET']
#         )
#         # print(df_score_cnt)
#         df_score_cnt = df_score_cnt.loc[:, ['기준년월', '모델ID', 'TARGET', '스코어_구간', '기존_스코어_분포', '현재_스코어_분포']]
#         # print(df_score_cnt)
            
#         # 데이터 합치기
#         df_results.append(df_score_cnt)   
#         # print(df_results)
#     # 하나의 데이터프레임으로 병합
#     df_results = pd.concat(df_results)
#     # print(df_results)

#     # psi산출_분포 비율(모델ID별로 PSI산출)
#     df_results['기존_스코어_비율'] = df_results['기존_스코어_분포'] / df_results['기존_스코어_분포'].sum()
#     df_results['현재_스코어_비율'] = df_results['현재_스코어_분포'] / df_results['현재_스코어_분포'].sum()
#     df_results['로그비율']         = np.log(df_results['현재_스코어_비율'] / df_results['기존_스코어_비율'])
        
#     # # psi산출(모델ID별로 PSI산출)                   
#     df_results['psi값']            = df_results['로그비율'] * (df_results['현재_스코어_비율'] - df_results['기존_스코어_비율'])  
#     # print(df_results)
#     df_psi_result = df_results.groupby(['기준년월', '모델ID', 'TARGET'])['psi값'].sum().reset_index()  
#     # # 스코어_재학습_여부 컬럼 초기화
#     df_psi_result['스코어_재학습_여부'] = np.where(
#         df_psi_result['psi값']  <= 0.25, '유지', np.where(
#             df_psi_result['psi값']  > 0.5, '재학습', '주의'
#         )
#     )
#     # print(df_psi_result)
