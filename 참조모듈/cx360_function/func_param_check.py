########################################################################################################################
# 라이브러리 선언                                                                                   
########################################################################################################################
import pandas as pd
import sqlite3
import os
from sqlalchemy import create_engine


def func_param_check (
      param    # 체크하고자 하는 데이터
    , check    # 체크 형태 - df,db : 건수 검증 / list : 데이터 타입, 인스턴스 타입, 개수 검증 / str, int, float : 데이터 타입 검증
    , instance # 리스트 인스턴스 타입 검증
    ):

    # JSON 파라미터 Key값 추출
    json_param = [k for k, v in json_smpl[0].items() if v == param]
    exec(open(f"{os.getcwd()}/data/dev/program/function/func_db_connect2.py"     , encoding='utf-8').read()) # DB호출함수
    
    ################################################################################
    ## DB 연결
    ################################################################################
    
    db_dir       = f"{os.getcwd()}/data/dev/source/pine.db"
    cur, conn, _ = dbConnect().dbConn("sqlite", db_dir=db_dir)
    
    # SQLAlchemy 연결
    engine = create_engine(f"sqlite:///{db_dir}")
    conn2 = engine.connect()
    

    
    # DB, DataFrame 데이터 건수 검증
    if check in ('df', 'db'):
        df_mart = pd.read_sql(
              f"""
              select *
              from {v_mart_tgt_nm}
              """
              , con = conn)

        v_len_df_mart = len(df_mart)

        if v_len_df_mart != 0:
            print(f"[LOG] {btch_nm} > 파라미터 검증 > {json_param[0]:<{max_json_param}}, 건수 : {format(v_len_df_mart, ',')}건")
        else:
            raise ValueError (f"[LOG] 파라미터 오류 > {json_param[0]} , 데이터 미존재")
    
    # LIST 데이터 검증 (건수, 인스턴스)
    elif check == list:
        if isinstance(param, check):
            print(f"[LOG] {btch_nm} > 파라미터 검증 > {json_param[0]:<{max_json_param}}, 데이터 타입 list 일치")
            
            if all(type(d) == instance for d in param):
                print(f"[LOG] {btch_nm} > 파라미터 검증 > {json_param[0]:<{max_json_param}}, 인스턴스 타입 {instance} 모두 일치")
                
                if len(param) != 0:
                    v_len_param = len(param)
                    print(f"[LOG] {btch_nm} > 파라미터 검증 > {json_param[0]:<{max_json_param}}, 개수 : {format(v_len_param, ',')}건")
                else:
                    raise ValueError (f"[LOG] 파라미터 오류> {json_param[0]:<{max_json_param}}, 데이터 미존재")
            
            else:
                raise ValueError (f"[LOG] 파라미터 검증 > {json_param[0]:<{max_json_param}}, 인스턴스 타입 {instance} 불일치")
        
        else:
            raise ValueError (f"[LOG] 파라미터 오류 > {json_param[0]:<{max_json_param}}, 데이터 타입 list 형태 불일치")

    # STR, INT, FLOAT 데이터 검증
    elif check in (str, int, float):
        if isinstance(param, check):
            print(f"[LOG] {btch_nm} > 파라미터 검증 > {json_param[0]:<{max_json_param}}, 데이터 타입 {check} 모두 일치")
        else:
            raise ValueError (f"[LOG] 파라미터 오류 > {json_param[0]:<{max_json_param}}, 데이터 타입 {check} 불일치")