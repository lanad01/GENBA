########################################################################################################################
# 라이브러리 선언
########################################################################################################################
import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import warnings

from sqlalchemy import create_engine
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from pickle import dump, load


def make_tbl_mdl_info(v_tgt_var, tbl_nm, df_prfmnc_cmp_test, mdl_baseYm = None, chmp_mdl_path = None):
    ########################################################################################################################
    # DB연결                                                                                            
    ########################################################################################################################
    func_dir = f"{os.getcwd()}{json_params['setting_parameter']['func_dir']}"
    # DB호출함수
    exec(open(f"{func_dir}/func_db_connect2.py"      , encoding='utf-8').read())

    # DB Directory
    db_dir       = f"{os.getcwd()}/data/dev/source/pine.db"
    cur, conn, _ = dbConnect().dbConn("sqlite", db_dir=db_dir)

    
    ####################################################################################################
    # 기준년월 
    ####################################################################################################
    if mdl_baseYm != None:
        baseYm = mdl_baseYm
    else:
        baseYm = datetime.strftime(datetime.now() - relativedelta(months = 1), '%Y%m')
        
        
    ########################################################################################################################
    # 모델 정보 데이터프레임 생성                                                                                         
    ########################################################################################################################
    
    # 챔피언 테이블 생성
    if chmp_mdl_path != None:
        df_new_info = df_prfmnc_cmp_test
        df_mdl_info = pd.DataFrame(
            columns = [
                   '기준년월'
                ,  '모델ID'  
                ,  'TARGET'  
                ,  'USER'
                ,  '모델생성년월'
                ,  '수행시점' 
                ,  '모델링회차'
                ,  '샘플링배수'
                ,  '알고리즘' 
                ,  '변수개수'
                ,  'cutOff'   
                ,  '정밀도'   
                ,  '재현율'   
                ,  'F1Score' 
                ,  '모델경로'
            ]
        )
        # 값 지정
        df_mdl_info['모델링회차']     = df_new_info['모델링회차']
        df_mdl_info['기준년월']       = baseYm
        df_mdl_info['모델ID']         = f'mdl_{v_tgt_var}_chmp'   
        df_mdl_info['TARGET']         = df_new_info['TARGET']    
        df_mdl_info['USER']           = df_new_info['USER']
        df_mdl_info['모델생성년월']   = df_new_info['모델생성년월']
        df_mdl_info['수행시점']       = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        df_mdl_info['샘플링배수']     = df_new_info['샘플링배수']
        df_mdl_info['알고리즘']       = df_new_info['알고리즘']  
        df_mdl_info['변수개수']       = df_new_info['변수개수']  
        df_mdl_info['cutOff']         = df_new_info['cutOff']    
        df_mdl_info['정밀도']         = df_new_info['정밀도']    
        df_mdl_info['재현율']         = df_new_info['재현율']    
        df_mdl_info['F1Score']        = df_new_info['F1Score']   
        df_mdl_info['모델경로']       = chmp_mdl_path


        # cur.execute(f"""drop table if exists {tbl_nm}""")

        # cur.execute(f"""
        #     create table if not exists {tbl_nm} (
        #           기준년월                           VARCHAR
        #         , 모델ID                             VARCHAR          
        #         , TARGET                             VARCHAR
        #         , USER                               VARCHAR
        #         , 수행시점                           VARCHAR           
        #         , 모델링회차                         VARCHAR          
        #         , 샘플링배수                         NUM           
        #         , 알고리즘                           VARCHAR
        #         , 변수개수                           NUM
        #         , cutOff                             VARCHAR
        #         , 정밀도                             NUM
        #         , 재현율                             NUM
        #         , F1Score                            NUM
        #         , 모델경로                           VARCHAR
        #         , PRIMARY KEY(기준년월, 모델ID, USER, 수행시점, cutOff)
        #     )
        # """
        # )

        # 데이터 초기화
        cur.execute(f"""delete from {tbl_nm} WHERE 기준년월 = {baseYm} AND TARGET = '{v_tgt_var}'""")

        # 데이터 적재
        df_mdl_info.to_sql(
              name      = f'{tbl_nm}'
            , con       = engine
            , if_exists = 'append'
            , index     = False
            , method    = "multi"
            , chunksize = 10000
        )

        DF건수 = len(df_mdl_info)

        #데이터프레임 건수확인
        DB건수 = pd.read_sql(
            f"""
                select count(1) as 건수
                from {tbl_nm}
                WHERE 기준년월 = {baseYm} 
                AND TARGET = '{v_tgt_var}'                
            """
        , conn
        ).values[0][0]

        print(f"[LOG] 생성테이블명 = {tbl_nm} DF건수 = {DF건수:,}, DB건수 = {DB건수:,}")
    else:
        df_mdl_info = pd.DataFrame(
            columns = [
                   '기준년월'
                ,  '모델ID'  
                ,  'TARGET'  
                ,  'USER'
                ,  '수행시점' 
                ,  '모델링회차'
                ,  '샘플링배수'
                ,  '알고리즘' 
                ,  '변수개수'
                ,  'cutOff'   
                ,  '정밀도'   
                ,  '재현율'   
                ,  'F1Score' 
            ]
        )
        # 값 지정
        df_mdl_info['모델링회차']     = df_prfmnc_cmp_test['Mdl_step']
        df_mdl_info['기준년월']       = baseYm
        df_mdl_info['모델ID']         = df_prfmnc_cmp_test['모델ID']
        df_mdl_info['TARGET']         = v_tgt_var
        df_mdl_info['USER']           = df_prfmnc_cmp_test['USER']
        df_mdl_info['모델생성년월']   = baseYm
        df_mdl_info['수행시점']       = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        df_mdl_info['샘플링배수']     = df_prfmnc_cmp_test['샘플링배수']
        df_mdl_info['알고리즘']       = df_prfmnc_cmp_test['Model']
        df_mdl_info['변수개수']       = df_prfmnc_cmp_test['변수개수']
        df_mdl_info['cutOff']         = df_prfmnc_cmp_test['cutOff']
        df_mdl_info['정밀도']         = df_prfmnc_cmp_test['정밀도']
        df_mdl_info['재현율']         = df_prfmnc_cmp_test['재현율']
        df_mdl_info['F1Score']        = df_prfmnc_cmp_test['F1Score']


        #cur.execute(f"""drop table if exists {tbl_nm}""")

        cur.execute(f"""
            create table if not exists {tbl_nm} (
                  기준년월                           VARCHAR
                , 모델ID                             VARCHAR          
                , TARGET                             VARCHAR
                , USER                               VARCHAR
                , 모델생성년월                       VARCHAR
                , 수행시점                           VARCHAR           
                , 모델링회차                         VARCHAR          
                , 샘플링배수                         NUM           
                , 알고리즘                           VARCHAR
                , 변수개수                           NUM
                , cutOff                             VARCHAR
                , 정밀도                             NUM
                , 재현율                             NUM
                , F1Score                            NUM
                , PRIMARY KEY(기준년월, 모델ID, USER, 모델생성년월, 수행시점, 모델링회차, 알고리즘, 샘플링배수, cutOff)
            )
        """
        )

        # 데이터 초기화
        cur.execute(f"""delete from {tbl_nm} WHERE 기준년월 = {baseYm} AND TARGET = '{v_tgt_var}'""")

        # 데이터 적재
        df_mdl_info.to_sql(
              name      = f'{tbl_nm}'
            , con       = engine
            , if_exists = 'append'
            , index     = False
            , method    = "multi"
            , chunksize = 10000
        )

        DF건수 = len(df_mdl_info)

        #데이터프레임 건수확인
        DB건수 = pd.read_sql(
            f"""
                select count(1) as 건수
                from {tbl_nm}
                WHERE 기준년월 = {baseYm} 
                AND TARGET = '{v_tgt_var}'                
            """
        , conn
        ).values[0][0]

        print(f"[LOG] 생성테이블명 = {tbl_nm} DF건수 = {DF건수:,}, DB건수 = {DB건수:,}")
    
    return df_mdl_info


def make_tbl_mdl_fi(v_tgt_var, tbl_nm, df_prfmnc_cmp_fi, mdl_baseYm = None, chmp_mdl_path = None):
    ########################################################################################################################
    # DB연결                                                                                            
    ########################################################################################################################
    func_dir = f"{os.getcwd()}{json_params['setting_parameter']['func_dir']}"
    # DB호출함수
    exec(open(f"{func_dir}/func_db_connect.py"      , encoding='utf-8').read())

    # DB Directory
    db_dir = f"{os.getcwd()}/data/dev/source/pine.db"
    dbConn = dbConnect()
    cur, conn, conn2 = dbConn.dbConn(db_dir)
    
    ####################################################################################################
    # 기준년월 
    ####################################################################################################
    if mdl_baseYm != None:
        baseYm = mdl_baseYm
    else:
        baseYm = datetime.strftime(datetime.now() - relativedelta(months = 1), '%Y%m')
        
    #########################################################################################################################
    # 변수중요도 테이블 생성                                                             
    #########################################################################################################################
    df_mdl_fi = pd.DataFrame(
        columns = [
               '기준년월'
            ,  '모델ID'  
            ,  'TARGET'  
            ,  'USER'
            ,  '수행시점' 
            ,  '모델링회차'
            ,  '샘플링배수'
            ,  '알고리즘' 
            ,  '변수명'   
            ,  '변수중요도'   
        ]
    )

    # 값 지정
    if chmp_mdl_path != None:    
        df_new_fi = df_prfmnc_cmp_fi
        df_mdl_fi['모델링회차']     = df_new_fi['모델링회차']
        df_mdl_fi['기준년월']       = baseYm
        df_mdl_fi['모델ID']         = f'mdl_{v_tgt_var}_chmp' 
        df_mdl_fi['TARGET']         = v_tgt_var
        df_mdl_fi['USER']           = df_new_fi['USER']
        df_mdl_fi['모델생성년월']   = df_new_fi['모델생성년월']
        df_mdl_fi['수행시점']       = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        df_mdl_fi['샘플링배수']     = df_new_fi['샘플링배수']
        df_mdl_fi['알고리즘']       = df_new_fi['알고리즘']
        df_mdl_fi['변수명']         = df_new_fi['변수명']
        df_mdl_fi['변수중요도']     = df_new_fi['변수중요도']
    else:
        df_mdl_fi['모델링회차']     = df_prfmnc_cmp_fi['Mdl_step']
        df_mdl_fi['기준년월']       = baseYm
        df_mdl_fi['모델ID']         = df_prfmnc_cmp_fi['모델ID']
        df_mdl_fi['TARGET']         = v_tgt_var
        df_mdl_fi['USER']           = df_prfmnc_cmp_fi['USER']
        df_mdl_fi['모델생성년월']   = baseYm
        df_mdl_fi['수행시점']       = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        df_mdl_fi['샘플링배수']     = df_prfmnc_cmp_fi['샘플링배수']
        df_mdl_fi['알고리즘']       = df_prfmnc_cmp_fi['algorithm']
        df_mdl_fi['변수명']         = df_prfmnc_cmp_fi['var']
        df_mdl_fi['변수중요도']     = df_prfmnc_cmp_fi['중요도']
        
    # cur.execute(f"""drop table if exists {tbl_nm}""")

    # 테이블 생성
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {tbl_nm} (
              기준년월                        VARCHAR
            , 모델ID                          VARCHAR
            , TARGET                          VARCHAR
            , USER                            VARCHAR
            , 모델생성년월                    VARCHAR
            , 수행시점                        VARCHAR
            , 모델링회차                      NUM
            , 샘플링배수                      VARCHAR
            , 알고리즘                        VARCHAR
            , 변수명                          VARCHAR
            , 변수중요도                      NUM
            , PRIMARY KEY(기준년월, 모델ID, TARGET, USER, 모델생성년월, 수행시점,모델링회차, 샘플링배수, 알고리즘, 변수명)       
        )
    """
    )
    # 데이터 초기화
    cur.execute(f"""delete from {tbl_nm} WHERE 기준년월 = {baseYm} AND TARGET = '{v_tgt_var}'""")

    # 데이터 적재
    df_mdl_fi.to_sql(
          name      = f'{tbl_nm}'
        , con       = engine
        , if_exists = 'append'
        , index     = False
        , method    = "multi"
        , chunksize = 10000
    )

    DF건수 = len(df_mdl_fi)

    #데이터프레임 건수확인
    DB건수 = pd.read_sql(
        f"""
            select count(1) as 건수
            from {tbl_nm}
            WHERE 기준년월 = {baseYm} 
            AND TARGET = '{v_tgt_var}'
        """
    , conn
    ).values[0][0]

    print(f"[LOG] 생성테이블명 = {tbl_nm} DF건수 = {DF건수:,}, DB건수 = {DB건수:,}")
    
    return df_mdl_fi



def make_tbl_mdl_score_dist(v_tgt_var, tbl_nm, df_prfmnc_cmp_score, mdl_baseYm = None, chmp_mdl_path = None):
    ########################################################################################################################
    # DB연결                                                                                            
    ########################################################################################################################
    func_dir = f"{os.getcwd()}{json_params['setting_parameter']['func_dir']}"
    # DB호출함수
    exec(open(f"{func_dir}/func_db_connect.py"      , encoding='utf-8').read())

    # DB Directory
    db_dir = f"{os.getcwd()}/data/dev/source/pine.db"
    dbConn = dbConnect()
    cur, conn, conn2 = dbConn.dbConn(db_dir)
    
    
    ####################################################################################################
    # 기준년월 
    ####################################################################################################
    if mdl_baseYm != None:
        baseYm = mdl_baseYm
    else:
        baseYm = datetime.strftime(datetime.now() - relativedelta(months = 1), '%Y%m')
        
        
    ########################################################################################################################
    # score 분포 데이터프레임 생성                                                                                        
    ########################################################################################################################
    df_mdl_score_dist = pd.DataFrame(
        columns = [
               '기준년월'
            ,  '모델ID'  
            ,  'TARGET'  
            ,  'USER'
            ,  '모델생성년월'
            ,  '수행시점' 
            ,  '모델링회차'
            ,  '샘플링배수'
            ,  '알고리즘' 
            ,  '고객식별번호'
            ,  'Y_Real'   
            ,  'Y_Prob'   
        ]
    )
    # 값 지정
    if df_prfmnc_cmp_score['기준년월'].dtypes =='float64':
        df_prfmnc_cmp_score['기준년월'] = df_prfmnc_cmp_score['기준년월'].astype(int).astype(str)
        
    if df_prfmnc_cmp_score['고객식별번호'].dtypes =='float64':
        df_prfmnc_cmp_score['고객식별번호'] = df_prfmnc_cmp_score['고객식별번호'].astype(int).astype(str)     
        
    if chmp_mdl_path != None:    
        df_new_score = df_prfmnc_cmp_score
        df_mdl_score_dist['모델링회차']     = df_new_score['모델링회차']
        df_mdl_score_dist['기준년월']       = baseYm
        df_mdl_score_dist['모델ID']         = f'mdl_{v_tgt_var}_chmp' 
        df_mdl_score_dist['TARGET']         = v_tgt_var
        df_mdl_score_dist['USER']           = df_new_score['USER']
        df_mdl_score_dist['모델생성년월']   = df_new_score['모델생성년월']
        df_mdl_score_dist['수행시점']       = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        df_mdl_score_dist['샘플링배수']     = df_new_score['샘플링배수']
        df_mdl_score_dist['알고리즘']       = df_new_score['알고리즘']
        df_mdl_score_dist['고객식별번호']   = df_new_score['고객식별번호']
        df_mdl_score_dist['Y_Real']         = df_new_score['Y_Real']
        df_mdl_score_dist['Y_Prob']         = df_new_score['Y_Prob']
   
    else:
        df_mdl_score_dist['모델링회차']     = df_prfmnc_cmp_score['Mdl_step']
        df_mdl_score_dist['기준년월']       = baseYm
        df_mdl_score_dist['모델ID']         = df_prfmnc_cmp_score['모델ID']
        df_mdl_score_dist['TARGET']         = v_tgt_var
        df_mdl_score_dist['USER']           = df_prfmnc_cmp_score['USER']
        df_mdl_score_dist['모델생성년월']   = baseYm
        df_mdl_score_dist['수행시점']       = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        df_mdl_score_dist['샘플링배수']     = df_prfmnc_cmp_score['샘플링배수']
        df_mdl_score_dist['알고리즘']       = df_prfmnc_cmp_score['Model']
        df_mdl_score_dist['고객식별번호']   = df_prfmnc_cmp_score['고객식별번호']
        df_mdl_score_dist['Y_Real']         = df_prfmnc_cmp_score['Y_Real']
        df_mdl_score_dist['Y_Prob']         = df_prfmnc_cmp_score['Y_Prob']


    # cur.execute(f"""drop table if exists {tbl_nm}""")

    cur.execute(f"""
        create table if not exists {tbl_nm} (
              기준년월                           VARCHAR
            , 모델ID                             VARCHAR          
            , TARGET                             VARCHAR
            , USER                               VARCHAR
            , 모델생성년월                       VARCHAR
            , 수행시점                           VARCHAR           
            , 모델링회차                         VARCHAR          
            , 샘플링배수                         NUM           
            , 알고리즘                           VARCHAR
            , 고객식별번호                       VARCHAR
            , Y_Real                             NUM
            , Y_Prob                             NUM   
            , PRIMARY KEY(기준년월, 모델ID, USER, 모델생성년월, 수행시점,모델링회차, 샘플링배수,알고리즘,고객식별번호)
        )
    """
    )

    # 데이터 초기화
    cur.execute(f"""delete from {tbl_nm} WHERE 기준년월 = {baseYm} AND TARGET = '{v_tgt_var}'""")

    # 데이터 적재
    df_mdl_score_dist.to_sql(
          name      = f'{tbl_nm}'
        , con       = engine
        , if_exists = 'append'
        , index     = False
        , method    = "multi"
        , chunksize = 10000
    )

    DF건수 = len(df_mdl_score_dist)

    #데이터프레임 건수확인
    DB건수 = pd.read_sql(
        f"""
            select count(1) as 건수
            from {tbl_nm}
        """
    , conn
    ).values[0][0]

    print(f"[LOG] 생성테이블명 = {tbl_nm} DF건수 = {DF건수:,}, DB건수 = {DB건수:,}")

    return df_mdl_score_dist
    
    
    
def make_tbl_mdl_lift(v_tgt_var, tbl_nm, df_prfmnc_cmp_lift, mdl_baseYm = None, chmp_mdl_path = None):    
    ########################################################################################################################
    # DB연결                                                                                            
    ########################################################################################################################
    func_dir = f"{os.getcwd()}{json_params['setting_parameter']['func_dir']}"
    # DB호출함수
    exec(open(f"{func_dir}/func_db_connect.py"      , encoding='utf-8').read())

    # DB Directory
    db_dir = f"{os.getcwd()}/data/dev/source/pine.db"
    dbConn = dbConnect()
    cur, conn, conn2 = dbConn.dbConn(db_dir)
    
    
    ####################################################################################################
    # 기준년월 
    ####################################################################################################
    if mdl_baseYm != None:
        baseYm = mdl_baseYm
    else:
        baseYm = datetime.strftime(datetime.now() - relativedelta(months = 1), '%Y%m')
        
    ########################################################################################################################
    # LIFT 데이터프레임 생성                                                                                        
    ########################################################################################################################        
    df_mdl_lift = pd.DataFrame(
        columns = [
               '기준년월'
            ,  '모델ID'  
            ,  'TARGET'  
            ,  'USER'
            ,  '모델생성년월'
            ,  '수행시점' 
            ,  '모델링회차'
            ,  '샘플링배수'
            ,  '알고리즘' 
            ,  'cutOff'
            ,  '마케팅대상고객수'
            ,  'mdl_반응률'   
            ,  'rnd_반응률'   
            ,  'Lift'   
        ]
    )
    if chmp_mdl_path != None:    
        df_new_lift = df_prfmnc_cmp_lift
        df_mdl_lift['모델링회차']       = df_new_lift['모델링회차']
        df_mdl_lift['기준년월']         = baseYm
        df_mdl_lift['모델ID']           = f'mdl_{v_tgt_var}_chmp' 
        df_mdl_lift['TARGET']           = v_tgt_var
        df_mdl_lift['USER']             = df_new_lift['USER']
        df_mdl_lift['모델생성년월']     = df_new_lift['모델생성년월']
        df_mdl_lift['수행시점']         = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        df_mdl_lift['샘플링배수']       = df_new_lift['샘플링배수']
        df_mdl_lift['알고리즘']         = df_new_lift['알고리즘']
        df_mdl_lift['cutOff']           = df_new_lift['cutOff']
        df_mdl_lift['마케팅대상고객수'] = df_new_lift['마케팅대상고객수']
        df_mdl_lift['mdl_반응률']       = df_new_lift['mdl_반응률']
        df_mdl_lift['rnd_반응률']       = df_new_lift['rnd_반응률']
        df_mdl_lift['Lift']             = df_new_lift['Lift']
    else:
        df_mdl_lift['모델링회차']       = df_prfmnc_cmp_lift['Mdl_step']
        df_mdl_lift['기준년월']         = baseYm
        df_mdl_lift['모델ID']           = df_prfmnc_cmp_lift['모델ID']
        df_mdl_lift['TARGET']           = v_tgt_var
        df_mdl_lift['USER']             = df_prfmnc_cmp_lift['USER']
        df_mdl_lift['모델생성년월']     = baseYm
        df_mdl_lift['수행시점']         = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        df_mdl_lift['샘플링배수']       = df_prfmnc_cmp_lift['샘플링배수']
        df_mdl_lift['알고리즘']         = df_prfmnc_cmp_lift['Model']
        df_mdl_lift['cutOff']           = df_prfmnc_cmp_lift['cutOff']
        df_mdl_lift['마케팅대상고객수'] = df_prfmnc_cmp_lift['마케팅대상고객수']
        df_mdl_lift['mdl_반응률']       = df_prfmnc_cmp_lift['mdl_반응률']
        df_mdl_lift['rnd_반응률']       = df_prfmnc_cmp_lift['rnd_반응률']
        df_mdl_lift['Lift']             = df_prfmnc_cmp_lift['Lift']

    # cur.execute(f"""drop table if exists {tbl_nm}""")

    cur.execute(f"""
        create table if not exists {tbl_nm} (
              기준년월                           VARCHAR
            , 모델ID                             VARCHAR          
            , TARGET                             VARCHAR
            , USER                               VARCHAR
            , 모델생성년월                       VARCHAR
            , 수행시점                           VARCHAR           
            , 모델링회차                         VARCHAR          
            , 샘플링배수                         NUM           
            , 알고리즘                           VARCHAR
            , cutOff                             NUM
            , 마케팅대상고객수                   NUM
            , mdl_반응률                         NUM
            , rnd_반응률                         NUM
            , Lift                               NUM        
            , PRIMARY KEY(기준년월, 모델ID, USER, 모델생성년월, 수행시점,모델링회차, 샘플링배수,알고리즘, cutOff, Lift)
        )
    """
    )

    # 데이터 초기화
    cur.execute(f"""delete from {tbl_nm} WHERE 기준년월 = {baseYm} AND TARGET = '{v_tgt_var}'""")

    # 데이터 적재
    df_mdl_lift.to_sql(
          name      = f'{tbl_nm}'
        , con       = engine
        , if_exists = 'append'
        , index     = False
        , method    = "multi"
        , chunksize = 10000
    )

    DF건수 = len(df_mdl_lift)

    #데이터프레임 건수확인
    DB건수 = pd.read_sql(
        f"""
            select count(1) as 건수
            from {tbl_nm}
        """
    , conn
    ).values[0][0]

    print(f"[LOG] 생성테이블명 = {tbl_nm} DF건수 = {DF건수:,}, DB건수 = {DB건수:,}")
    
    return df_mdl_lift

    
def make_tbl_mdl_mtc(v_tgt_var, tbl_nm, df_prfmnc_cmp_mtc, mdl_baseYm = None, chmp_mdl_path = None):
    ########################################################################################################################
    # DB연결                                                                                            
    ########################################################################################################################
    func_dir = f"{os.getcwd()}{json_params['setting_parameter']['func_dir']}"
    # DB호출함수
    exec(open(f"{func_dir}/func_db_connect.py"      , encoding='utf-8').read())

    # DB Directory
    db_dir = f"{os.getcwd()}/data/dev/source/pine.db"
    dbConn = dbConnect()
    cur, conn, conn2 = dbConn.dbConn(db_dir)
    
    ####################################################################################################
    # 기준년월 
    ####################################################################################################
    if mdl_baseYm != None:
        baseYm = mdl_baseYm
    else:
        baseYm = datetime.strftime(datetime.now() - relativedelta(months = 1), '%Y%m')
        
        
    ########################################################################################################################
    # 모델 정보 데이터프레임 생성                                                                                         
    ########################################################################################################################
    
    # 챔피언 테이블 생성
    if chmp_mdl_path != None:
        df_new_info = df_prfmnc_cmp_mtc
        df_mdl_info = pd.DataFrame(
            columns = [
                   '기준년월'
                ,  '모델ID'  
                ,  'TARGET'  
                ,  'USER'
                ,  '모델생성년월'
                ,  '수행시점' 
                ,  '모델링회차'
                ,  '샘플링배수'
                ,  '알고리즘' 
                ,  '변수개수'
                ,  'cutOff'   
                ,  '정밀도'   
                ,  '재현율'   
                ,  'F1Score' 
            ]
        )
        # 값 지정
        df_mdl_info['모델링회차']     = df_new_info['모델링회차']
        df_mdl_info['기준년월']       = baseYm 
        df_mdl_info['모델ID']         = f'mdl_{v_tgt_var}_chmp'   
        df_mdl_info['TARGET']         = df_new_info['TARGET']    
        df_mdl_info['USER']           = df_new_info['USER']   
        df_mdl_info['모델생성년월']   = df_new_info['모델생성년월'] 
        df_mdl_info['수행시점']       = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        df_mdl_info['샘플링배수']     = df_new_info['샘플링배수']
        df_mdl_info['알고리즘']       = df_new_info['알고리즘']  
        df_mdl_info['변수개수']       = df_new_info['변수개수']  
        df_mdl_info['cutOff']         = df_new_info['cutOff']    
        df_mdl_info['정밀도']         = df_new_info['정밀도']    
        df_mdl_info['재현율']         = df_new_info['재현율']    
        df_mdl_info['F1Score']        = df_new_info['F1Score']   

        cur.execute(f"""
            create table if not exists {tbl_nm} (
                  기준년월                           VARCHAR
                , 모델ID                             VARCHAR          
                , TARGET                             VARCHAR
                , USER                               VARCHAR
                , 모델생성년월                       VARCHAR
                , 수행시점                           VARCHAR           
                , 모델링회차                         VARCHAR          
                , 샘플링배수                         NUM           
                , 알고리즘                           VARCHAR
                , 변수개수                           NUM
                , cutOff                             VARCHAR
                , 정밀도                             NUM
                , 재현율                             NUM
                , F1Score                            NUM
                , PRIMARY KEY(기준년월, 모델ID, TARGET,USER, 모델생성년월, 수행시점,모델링회차, 샘플링배수,알고리즘, cutOff)
            )
        """
        )

        # 데이터 초기화
        cur.execute(f"""delete from {tbl_nm} WHERE 기준년월 = {baseYm} AND TARGET = '{v_tgt_var}'""")

        # 데이터 적재
        df_mdl_info.to_sql(
              name      = f'{tbl_nm}'
            , con       = engine
            , if_exists = 'append'
            , index     = False
            , method    = "multi"
            , chunksize = 10000
        )

        DF건수 = len(df_mdl_info)

        #데이터프레임 건수확인
        DB건수 = pd.read_sql(
            f"""
                select count(1) as 건수
                from {tbl_nm}
                WHERE TARGET = '{v_tgt_var}'                
            """
        , conn
        ).values[0][0]

        print(f"[LOG] 생성테이블명 = {tbl_nm} DF건수 = {DF건수:,}, DB건수 = {DB건수:,}")
    else:
        df_mdl_info = pd.DataFrame(
            columns = [
                   '기준년월'
                ,  '모델ID'  
                ,  'TARGET'  
                ,  'USER'
                ,  '모델생성년월'
                ,  '수행시점' 
                ,  '모델링회차'
                ,  '샘플링배수'
                ,  '알고리즘' 
                ,  '변수개수'
                ,  'cutOff'   
                ,  '정밀도'   
                ,  '재현율'   
                ,  'F1Score' 
            ]
        )
        # 값 지정
        df_mdl_info['모델링회차']     = df_prfmnc_cmp_mtc['Mdl_step']
        df_mdl_info['기준년월']       = baseYm
        df_mdl_info['모델ID']         = df_prfmnc_cmp_mtc['모델ID']
        df_mdl_info['TARGET']         = v_tgt_var
        df_mdl_info['USER']           = df_prfmnc_cmp_mtc['USER']
        df_mdl_info['모델생성년월']   = baseYm
        df_mdl_info['수행시점']       = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        df_mdl_info['샘플링배수']     = df_prfmnc_cmp_mtc['샘플링배수']
        df_mdl_info['알고리즘']       = df_prfmnc_cmp_mtc['Model']
        df_mdl_info['변수개수']       = df_prfmnc_cmp_mtc['변수개수']
        df_mdl_info['cutOff']         = df_prfmnc_cmp_mtc['cutOff']
        df_mdl_info['정밀도']         = df_prfmnc_cmp_mtc['정밀도']
        df_mdl_info['재현율']         = df_prfmnc_cmp_mtc['재현율']
        df_mdl_info['F1Score']        = df_prfmnc_cmp_mtc['F1Score']


        #cur.execute(f"""drop table if exists {tbl_nm}""")

        cur.execute(f"""
            create table if not exists {tbl_nm} (
                  기준년월                           VARCHAR
                , 모델ID                             VARCHAR          
                , TARGET                             VARCHAR
                , USER                               VARCHAR
                , 모델생성년월                       VARCHAR
                , 수행시점                           VARCHAR           
                , 모델링회차                         VARCHAR          
                , 샘플링배수                         NUM           
                , 알고리즘                           VARCHAR
                , 변수개수                           NUM
                , cutOff                             VARCHAR
                , 정밀도                             NUM
                , 재현율                             NUM
                , F1Score                            NUM
                , PRIMARY KEY(기준년월, 모델ID, TARGET,USER, 모델생성년월, 수행시점,모델링회차, 샘플링배수,알고리즘, cutOff)
            )
        """
        )

        # 데이터 초기화
        cur.execute(f"""delete from {tbl_nm} WHERE 기준년월 = {baseYm} AND TARGET = '{v_tgt_var}'""")

        # 데이터 적재
        df_mdl_info.to_sql(
              name      = f'{tbl_nm}'
            , con       = engine
            , if_exists = 'append'
            , index     = False
            , method    = "multi"
            , chunksize = 10000
        )

        DF건수 = len(df_mdl_info)

        #데이터프레임 건수확인
        DB건수 = pd.read_sql(
            f"""
                select count(1) as 건수
                from {tbl_nm}
                WHERE 기준년월 = {baseYm} 
                AND TARGET = '{v_tgt_var}'                
            """
        , conn
        ).values[0][0]

        print(f"[LOG] 생성테이블명 = {tbl_nm} DF건수 = {DF건수:,}, DB건수 = {DB건수:,}")
    
    return df_mdl_info
    
    