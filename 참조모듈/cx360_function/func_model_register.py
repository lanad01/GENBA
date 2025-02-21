################################################################################
# 패키지 IMPORT
################################################################################
import numpy as np
import pandas as pd
import sqlite3
import os
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz
import pickle
import json
KST = pytz.timezone('Asia/Seoul')

func_dir = f"{os.getcwd()}/data/dev/program/function"

# DB 호출함수
exec(open(f"{func_dir}/func_db_connect2.py"    , encoding='utf-8').read())

################################################################################
## DB 연결
################################################################################


cur, conn, _ = dbConnect().dbConn("sqlite", db_dir=db_dir)
print("[Log] DB connection Completed")


################################################################################
# 모델 등록 함수 정의
################################################################################
def func_model_register(
          df_mdl_test_tmp      # PINE_CHMP_TEST 테이블에서 로드한 챔피언 모델 메타데이터 
        , v_target             # 타겟
    ):
    if not df_mdl_test_tmp.empty:
        # PINE_MDL_CATALOG 테이블 미존재 시 테이블 생성
        cur.execute(
            """
                create table if not exists PINE_MDL_CATALOG (
                      기준년월      text
                    , 모델ID        text
                    , TARGET        text
                    , USER          text
                    , 수행시점      text
                    , 알고리즘      text
                    , cutOff        text
                    , 정밀도        REAL
                    , 재현율        REAL
                    , F1Score       REAL
                    , 모델경로      text
                    , 모델상태정보  text
                    , 모델등록일시  text
                    , 샘플링마트    text
                    , 피처마트      text
                    , PRIMARY KEY(기준년월, 모델ID, TARGET)
                )
            """)
                
        # 저장할 모델이 챔피언 모델이면, 기존 모델을 챔피언에서 챌린지로 변경 
        if (df_mdl_test_tmp['TARGET'] == v_target).any():
            
            # 기존 챔피언 모델 메타 정보 불러오기
            df_pre_chm_mdl = pd.read_sql(
                f"""
                    SELECT *
                    FROM PINE_MDL_CATALOG
                    WHERE 
                              TARGET = '{v_target}'
                    AND 모델상태정보 = '챔피언'
                """, conn)
            df_pre_chm_mdl['모델상태정보'] = '챌린지'

            # 기존 챔피언 모델 메타 정보 삭제
            cur.execute(
                f"""
                    delete from PINE_MDL_CATALOG
                    where 
                              TARGET = '{v_target}'
                    AND 모델상태정보 = '챔피언'
                """
                )

            # 기존 챔피언 모델 재.적재(챔피언 -> 챌린지)
            df_pre_chm_mdl.to_sql(
                  name      = f'PINE_MDL_CATALOG'
                , con       = conn2
                , if_exists = 'append'
                , index     = False
                , method    = "multi"
                , chunksize = 10000
            )

            with open(f"{os.getcwd()}/data/dev/json/json_mdling.json", encoding="UTF-8") as f:
                data = json.load(f)

            df_mdl_test_tmp = df_mdl_test_tmp.reindex(columns = df_pre_chm_mdl.columns)
            df_mdl_test_tmp = df_mdl_test_tmp.fillna(np.nan)
            df_mdl_test_tmp['모델상태정보'] = '챔피언'
            df_mdl_test_tmp['모델등록일시'] = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
            df_mdl_test_tmp['샘플링마트']   = data['modeling_parameter']["smp_mart_nm"]
            df_mdl_test_tmp['피처마트']     = data['modeling_parameter']["ftr_mart_nm"]

            # 새로운 챔피언 모델 적재
            df_mdl_test_tmp.to_sql(
                  name      = f'PINE_MDL_CATALOG'
                , con       = conn2
                , if_exists = 'append'
                , index     = False
                , method    = "multi"
                , chunksize = 10000
            )
        # 저장할 모델이 챌린지 모델이면 모델 카타로그에 그냥 저장 X
        else:
            print("[LOG] No metadata information to save")

    else:
        print("[LOG] Empty DataFrame")
