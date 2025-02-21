########################################################################################################################
# 라이브러리 선언 
########################################################################################################################
import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import re
import json
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
# 모델_변수중요도 분포 함수                                                                                     
########################################################################################################################

def func_mdl_vrb_imptnc_mntrg(
          df_mdl_list # 타겟 리스트
        , list_PK     # PK 리스트
        , baseYm      # 기준년월
    ):
    list_key = list(set(list_PK) - set(['기준년월']))

    for idx, (tar, cutoff) in enumerate(df_mdl_list, start = 1):
        print(f'[LOG] 순서 : {idx} / {len(df_mdl_list)}, TARGET : {tar}')

        # 변수중요도 상위 Top 10 변수명 추출
        df_col_info = pd.read_sql(
        f'''
            SELECT 변수명
            FROM TBL_CHMP_FI
            WHERE 기준년월 = {baseYm}
            AND TARGET = '{tar}'
            ORDER BY 변수중요도 DESC
        '''
        , conn
        )
        list_cols_10 = df_col_info['변수명'][:10].tolist()

        # 챔피언 모델이 사용한 피처마트명 불러오기
        v_ftr_mart_nm = pd.read_sql(
        f'''
            SELECT 피처마트
            FROM PINE_MDL_CATALOG
            WHERE 기준년월 = {baseYm}
            AND TARGET = '{tar}'
        '''
        , conn
        ).values[0][0]
        print(f'[LOG] 순서 : {idx} / {len(df_mdl_list)}, TARGET : {tar}, 피처마트명 : {v_ftr_mart_nm}, 변수중요도 Top 10 : {list_cols_10}')

        ####################################
        # 파생변수 변수중요도 분포 추출
        ####################################
        list_drv_cols = [col for col in list_cols_10 if re.search(r'_\d+m_(sum|avg)', col)]
        list_rmn_cols = list(set(list_cols_10) - set(list_drv_cols))
        baseYm0 = baseYm   # 수정해야 함.

        df_tmp_mart = pd.read_sql(
            f"""
                SELECT {", ".join(list_PK)}
                    , {", ".join(list_rmn_cols)}
                FROM {v_ftr_mart_nm}
                WHERE 기준년월 = '{baseYm0}'
            """, conn)

        for idx2, col in enumerate(list_drv_cols):
            match = re.match(r'(.*)_(\d+)m_(sum|avg)', col)
            if match:
                col_name   = match.group(1)       # 영화 이름 (예: "위인_관심영화_예산평균")
                period     = int(match.group(2))  # 숫자 (예: "3")
                flag       = match.group(3)       # 연산 (예: "sum")
                baseYm_bfr = datetime.strftime(datetime.strptime(baseYm0 + "01", '%Y%m%d') - relativedelta(months = period - 1), '%Y%m')
                
                df_tmp = pd.read_sql(
                    f'''
                        SELECT
                              AA.기준년월
                            , '{tar}'                                                      AS TARGET
                            , '{col_name}_{period}m_{flag}'                                AS 컬럼명
                            , AA.속성값
                            , count(*)                                                     AS 속성건수
                        FROM (
                            SELECT 
                                  A.기준년월
                                , case
                                    when {col_name}_{period}m_{flag} <=   1000 then '01'
                                    when {col_name}_{period}m_{flag} <=   3000 then '02'
                                    when {col_name}_{period}m_{flag} <=   5000 then '03'
                                    when {col_name}_{period}m_{flag} <=  10000 then '04'
                                    when {col_name}_{period}m_{flag} <=  30000 then '05'
                                    when {col_name}_{period}m_{flag} <= 100000 then '06'
                                    else                                            '07'
                                end                                                        AS 속성값
                            FROM (
                                SELECT
                                    '{baseYm0}'                                            AS 기준년월
                                    , {','.join(list_key)}
                                    , {flag}({col_name})                                   AS {col_name}_{period}m_{flag}
                                FROM {v_ftr_mart_nm}
                                WHERE 기준년월 BETWEEN '{baseYm_bfr}' AND '{baseYm0}'
                                GROUP BY {','.join(list_key)}
                            ) A
                        ) AA
                        GROUP BY 1,4
                    ''', conn)
                
                print(f'[LOG] 순서 : {idx} / {len(df_mdl_list)}, TARGET : {tar}, 파생변수 컬럼명 : {col}, 건수 : {len(df_tmp)}')

                if idx2 == 0 :
                    df_drv_mart = df_tmp
                else:
                    df_drv_mart = pd.concat([df_drv_mart, df_tmp])
        
        ####################################
        # 파생변수 외 변수중요도 분포 추출
        ####################################

        qty_keyword = ['건수', '개수']
        per_keyword = ['비율']         # 수익률, '_률' 수익률
        age_keyword = ['나이', '연령']

        for idx3, col2 in enumerate(list_rmn_cols):
            if df_tmp_mart[col2].dtypes == "object":
                str_select = f"CASE WHEN {col2} IS NULL THEN 'Null' ELSE {col2} END"
            else:
                if any(keyword in col2 for keyword in qty_keyword):
                    str_select = f"""
                            case
                                when {col2}  =  0 then '01'
                                when {col2} <=  2 then '02'
                                when {col2} <=  4 then '03'                            
                                when {col2} <=  6 then '04'
                                when {col2} <=  8 then '05'
                                when {col2} <= 10 then '06'
                                else                   '07'
                            end 
                        """
                elif any(keyword in col2 for keyword in per_keyword):
                    str_select = f"""
                            case
                                when {col2}  = 0   then '01'
                                when {col2} <= 0.2 then '02'
                                when {col2} <= 0.4 then '03'
                                when {col2} <= 0.6 then '04'
                                when {col2} <= 0.8 then '05'
                                when {col2} <= 1.0 then '06'
                                else                    '07'
                            end 
                        """                            
                elif any(keyword in col2 for keyword in age_keyword):
                    str_select = f"""
                            case
                                when {col2} < 20 then '01'
                                when {col2} < 30 then '02'
                                when {col2} < 40 then '03'
                                when {col2} < 50 then '04'
                                when {col2} < 60 then '05'
                                when {col2} < 70 then '06'
                                else                  '07'
                            end 
                        """                            
         
                else : 
                    str_select = f"""
                            case
                                when {col2} <=   1000 then '01'
                                when {col2} <=   3000 then '02'
                                when {col2} <=   5000 then '03'
                                when {col2} <=  10000 then '04'
                                when {col2} <=  30000 then '05'
                                when {col2} <= 100000 then '06'
                                else                       '07'
                            end 
                        """                            
            
            df_tmp2 = pd.read_sql( 
                f'''
                    SELECT 
                          A.기준년월
                        , '{tar}'                        AS TARGET
                        , '{col2}'                       AS 컬럼명
                        , A.{col2}                       AS 속성값
                        , count(*)                       AS 속성건수
                    FROM (
                        select 
                            기준년월
                            , {str_select}               AS {col2}
                        from {v_ftr_mart_nm}
                        WHERE 기준년월 = '{baseYm0}'
                    ) A
                    group by 1, 4
                '''                                                             
                , conn                                                          
            )
            
            print(f'[LOG] 순서 : {idx} / {len(df_mdl_list)}, TARGET : {tar}, 파생변수외 컬럼명 : {col2}, 건수 : {len(df_tmp2)}')
            if idx3 == 0 :
                df_rmn_mart = df_tmp2
            else:
                df_rmn_mart = pd.concat([df_rmn_mart, df_tmp2])

        df_var_dist = pd.concat([df_drv_mart, df_rmn_mart])
        # print(df_var_dist)
        ########################################################################################################################
        # 모델 변수중요도 분포 데이터 적재                                                                                     
        ########################################################################################################################

        # MDL_IMPTNC_MONITORING 테이블 미존재 시 테이블 생성
        cur.execute(
            """
            create table if not exists MDL_IMPTNC_MONITORING (
                  기준년월              text
                , TARGET                text
                , 컬럼명                text
                , 속성값                text
                , 속성건수              NUM
                , PRIMARY KEY (기준년월, TARGET, 컬럼명, 속성값)
                )
            """
        )
        cur.execute(f"""delete from MDL_IMPTNC_MONITORING where 기준년월 = {baseYm} and TARGET = '{tar}'""")
        # 모델_스코어 적재
        df_var_dist.to_sql(
              name      = 'MDL_IMPTNC_MONITORING'
            , con       = conn2
            , if_exists = 'append'
            , index     = False
            , method    = "multi"
            , chunksize = 10000
        )
        print(f'[LOG] 순서 : {idx} / {len(df_mdl_list)}, TARGET : {tar}, MDL_IMPTNC_MONITORING 적재완료')
# func_mdl_vrb_imptnc_mntrg(df_mdl_list, list_PK, baseYm)
# if __name__ == '__main__':
#     with open(f"{os.getcwd()}/data/dev/json/3.Modeling.json", encoding="UTF-8") as f:
#         data = json.load(f)

#     baseYm   = data["modeling_parameter"]["baseYm"]
#     list_PK  = data["modeling_parameter"]["list_PK"]
#     list_key = list(set(list_PK) - set(['기준년월']))
#     print(baseYm, list_key)
#     # baseYm = '201102' 
#     # TARGET LIST 추출
#     # df_mdl_list = pd.read_sql(
#     # f"""
#     #     select TARGET, cutOff
#     #     from PINE_MDL_CATALOG
#     #     where 모델상태정보 = '챔피언'
#     # """     
#     # , con = conn
#     # )
#     # df_mdl_list = df_mdl_list.values.tolist()
#     df_mdl_list = [['스릴러','61']]
#     list_col = []
#     for idx, (tar, cutoff) in enumerate(df_mdl_list):
#         # print(f'[LOG] 순서 = {i} / {len(df_mdl_list)}  모델ID = {x}')

#         # 변수중요도 기준 변수명 추출
#         df_col_info = pd.read_sql(
#         f'''
#             SELECT 변수명
#             FROM TBL_CHMP_FI
#             WHERE 기준년월 = {baseYm}
#             AND TARGET = '{tar}'
#             ORDER BY 변수중요도 DESC
#         '''
#         , conn
#         )
#         list_cols_10 = df_col_info['변수명'][:10].tolist()
#         # print(list_cols_10)
#         # 챔피언 모델이 사용한 피처마트명 불러오기
#         tmp_feature_mart_nm = pd.read_sql(
#         f'''
#             SELECT 피처마트
#             FROM PINE_MDL_CATALOG
#             WHERE 기준년월 = {baseYm}
#             AND TARGET = '{tar}'
#         '''
#         , conn
#         ).values[0][0]

#         ####################################
#         # 파생변수 컬럼 추출
#         ####################################
#         list_drv_cols = [col for col in list_cols_10 if re.search(r'_\d+m_(sum|avg)', col)]

#         list_rmn_cols = list(set(list_cols_10) - set(list_drv_cols))
#         baseYm0 = baseYm

#         df_tmp_mart = pd.read_sql(
#             f"""
#                 SELECT {", ".join(list_PK)}
#                     , {", ".join(list_rmn_cols)}
#                 FROM {tmp_feature_mart_nm}
#                 WHERE 기준년월 = '{baseYm0}'
#             """, conn)
#         # print(list_drv_cols)
#         for idx2, col in enumerate(list_drv_cols):
#             match = re.match(r'(.*)_(\d+)m_(sum|avg)', col)
#             if match:
#                 col_name   = match.group(1)       # 영화 이름 (예: "위인_관심영화_예산평균")
#                 period     = int(match.group(2))  # 숫자 (예: "3")
#                 flag       = match.group(3)       # 연산 (예: "sum")
#                 baseYm_bfr = datetime.strftime(datetime.strptime(baseYm0 + "01", '%Y%m%d') - relativedelta(months = period - 1), '%Y%m')
                
#                 df_tmp = pd.read_sql(
#                     f'''
#                         SELECT
#                               AA.기준년월
#                             , '{tar}'                                                      AS TARGET
#                             , '{col_name}_{period}m_{flag}'                                AS 컬럼명
#                             , AA.속성값
#                             , count(*)                                                     AS 속성건수
#                         FROM (
#                             SELECT 
#                                   A.기준년월
#                                 , case
#                                     when {col_name}_{period}m_{flag} <=   1000 then '01'
#                                     when {col_name}_{period}m_{flag} <=   3000 then '02'
#                                     when {col_name}_{period}m_{flag} <=   5000 then '03'
#                                     when {col_name}_{period}m_{flag} <=  10000 then '04'
#                                     when {col_name}_{period}m_{flag} <=  30000 then '05'
#                                     when {col_name}_{period}m_{flag} <= 100000 then '06'
#                                     else                                            '07'
#                                 end                                                        AS 속성값
#                             FROM (
#                                 SELECT
#                                     '{baseYm0}'                                            AS 기준년월
#                                     , {','.join(list_key)}
#                                     , {flag}({col_name})                                   AS {col_name}_{period}m_{flag}
#                                 FROM {tmp_feature_mart_nm}
#                                 WHERE 기준년월 BETWEEN '{baseYm_bfr}' AND '{baseYm0}'
#                                 GROUP BY {','.join(list_key)}
#                             ) A
#                         ) AA
#                         GROUP BY 1,4
#                     ''', conn)

#                 # print(df_tmp)
#                 # print('-'*100)

#                 if idx2 == 0 :
#                     tmp = df_tmp
#                 else:
#                     tmp = pd.concat([tmp, df_tmp])

#         # df_tmp_mart = df_tmp_mart.merge(df_mart_tmp
#         #                                 , on = list_PK
#         #                                 , how = 'left')
#         # print(df_tmp_mart)

#         qty_keyword = ['건수', '개수']
#         per_keyword = ['비율', '수익률']
#         age_keyword = ['나이', '연령']
#         list_col_cnt_1 = []
#         # print(list_rmn_cols)
#         for idx3, col2 in enumerate(list_rmn_cols):
#             # print(f"순서 : {idx2}, 컬럼명 : {col2}")
#             if df_tmp_mart[col2].dtypes == "object":
#                 # print(f"TEXT TYPE COLUMNS : {col2}")
#                 str_select = f"CASE WHEN {col2} IS NULL THEN 'Null' ELSE {col2} END"

#             else:
#                 # print(f"FLOAT OR INT TYPE COLUMNS : {col2}")
#                 if any(keyword in col2 for keyword in qty_keyword):
#                     # print(f"건수, 개수와 관련된 col {col2}")
#                     str_select = f"""
#                             case
#                                 when {col2}  =  0 then '01'
#                                 when {col2} <=  2 then '02'
#                                 when {col2} <=  4 then '03'                            
#                                 when {col2} <=  6 then '04'
#                                 when {col2} <=  8 then '05'
#                                 when {col2} <= 10 then '06'
#                                 else                   '07'
#                             end 
#                         """
#                 elif any(keyword in col2 for keyword in per_keyword):
#                     # print(f"비율과 관련된 col {col2}")
#                     str_select = f"""
#                             case
#                                 when {col2}  = 0   then '01'
#                                 when {col2} <= 0.2 then '02'
#                                 when {col2} <= 0.4 then '03'
#                                 when {col2} <= 0.6 then '04'
#                                 when {col2} <= 0.8 then '05'
#                                 when {col2} <= 1.0 then '06'
#                                 else                    '07'
#                             end 
#                         """                            
#                 elif any(keyword in col2 for keyword in age_keyword):
#                     # print(f"나이와 관련된 col {col2}")
#                     str_select = f"""
#                             case
#                                 when {col2} < 20 then '01'
#                                 when {col2} < 30 then '02'
#                                 when {col2} < 40 then '03'
#                                 when {col2} < 50 then '04'
#                                 when {col2} < 60 then '05'
#                                 when {col2} < 70 then '06'
#                                 else                  '07'
#                             end 
#                         """                            
         
#                 else : 
#                     # print(f"나머지 col : {col2}")
#                     str_select = f"""
#                             case
#                                 when {col2} <=   1000 then '01'
#                                 when {col2} <=   3000 then '02'
#                                 when {col2} <=   5000 then '03'
#                                 when {col2} <=  10000 then '04'
#                                 when {col2} <=  30000 then '05'
#                                 when {col2} <= 100000 then '06'
#                                 else                       '07'
#                             end 
#                         """                            
            
#             tmp_col_dtl_1 = pd.read_sql( 
#                 f'''
#                     SELECT 
#                             A.기준년월
#                         , '{tar}'                        AS TARGET
#                         , '{col2}'                       AS 컬럼명
#                         , A.{col2}                       AS 속성값
#                         , count(*)                       AS 속성건수
#                     FROM (
#                         select 
#                             기준년월
#                             , {str_select}               AS {col2}
#                         from {tmp_feature_mart_nm}
#                         WHERE 기준년월 = '{baseYm0}'
#                     ) A
#                     group by 1, 4
#                 '''                                                             
#                 , conn                                                          
#             )

#             if idx3 == 0 :
#                 tmp2 = tmp_col_dtl_1
#             else:
#                 tmp2 = pd.concat([tmp2, tmp_col_dtl_1])
#         # print(tmp2)

#         df_var_dist = pd.concat([tmp, tmp2])
#         print(df_var_dist['컬럼명'].unique())


