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

from pickle import dump, load


def func_make_data(v_model_turn, json_params, mdl_baseYm = None):
    ########################################################################################################################
    # DB연결                 
    ########################################################################################################################
    func_dir = f"{os.getcwd()}{json_params['setting_parameter']['func_dir']}"
    # DB호출함수
    exec(open(f"{func_dir}/func_db_connect2.py", encoding='utf-8').read())

    # DB Directory
    # db_dir = f"{os.getcwd()}/data/dev/source/pine.db"
    # dbConn = dbConnect()
    # cur, conn, conn2 = dbConn.dbConn(db_dir)
    db_dir       = f"{os.getcwd()}/data/dev/source/pine.db"
    cur, conn, _ = dbConnect().dbConn("sqlite", db_dir=db_dir)
    
    
    ####################################################################################################
    # 기준년월 
    ####################################################################################################
    if mdl_baseYm != None:
        baseYm = mdl_baseYm
    else:
        baseYm = datetime.strftime(datetime.now() - relativedelta(months = 1), '%Y%m')
        
    # json_params = pd.read_json("C:/PythonProject/MLOps솔루션/data/dev/김규형/json/3.Modeling.json")
    
    df_mdl_params  = json_params['modeling_parameter'].dropna()
    setting_params = json_params['setting_parameter'].dropna()
    
    ####################################################################################################
    # 저장 디렉토리 확인 및 함수 호출
    ####################################################################################################
    st_dir = os.getcwd()
    
    save_dir = []
    save_dir.append(setting_params['1st_dir'])
    save_dir.append(setting_params['2nd_dir'])
    save_dir.append(setting_params['3rd_dir'])
    save_dir.append(setting_params['4th_dir'])
    
    if v_model_turn not in [1,2,3,4]:
        raise Exception("v_model_turn 값을 확인하세요 1,2,3 셋중에 하나를 입력하세요")

    # 마트명
    v_smpl_mart_nm = df_mdl_params['smp_mart_nm']
    v_ftr_mart_nm  = df_mdl_params['ftr_mart_nm']
    v_tgt_mart_nm  = df_mdl_params['tgt_mart_nm']

    # 생성할 파생변수 기간 ex) n개월 평균
    v_period_derv = df_mdl_params['period_derv']
    
    v_tgt_var = df_mdl_params['v_tgt_var']
    
    # 시작년월 기준년월 형식확인
    df_temp = pd.Series([df_mdl_params['train_st_baseYm']
                       , df_mdl_params['train_ed_baseYm']
                       , df_mdl_params['test_st_baseYm']
                       , df_mdl_params['test_ed_baseYm']])
    # 각 값이 6자리문자열 및 시작일자 < 종료일자 일때 해당 json 파라미터 값을 대입
    if ((sum(df_temp.apply(lambda x : len(x)) != 6) == 0) & (df_temp[0] <= df_temp[1]) & (df_temp[2] <= df_temp[3])) & (sum([1 if isinstance(x, str) else 0  for x in df_temp]) == 4):
        v_train_st_baseYm = df_mdl_params['train_st_baseYm']
        v_train_ed_baseYm = df_mdl_params['train_ed_baseYm']
        v_test_st_baseYm  = df_mdl_params['test_st_baseYm']
        v_test_ed_baseYm  = df_mdl_params['test_ed_baseYm']
    
    else:        
        # 기준년월시점에서 답지 없음 (Y값이 존재하지않아 훈련 및 검증하는데 사용 할 수 없음)
        # 기준년월시점 1달전 테스트마트 (1달)
        # 시준년월시점 13달전~2달전 훈련마트 (12달)
        v_train_st_baseYm = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months = 13), '%Y%m')
        v_train_ed_baseYm = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months =  2), '%Y%m')
        v_test_st_baseYm  = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months =  1), '%Y%m')
        v_test_ed_baseYm  = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months =  1), '%Y%m')
    
    print(f"[LOG] 기준년월: {baseYm}, 훈련마트 생성기간: {v_train_st_baseYm} ~ {v_train_ed_baseYm}, 평가마트 생성기간 {v_test_st_baseYm} ~ {v_test_ed_baseYm}")
    
    list_mult  = pd.read_sql(f"""
    SELECT DISTINCT 샘플링배수
    FROM {v_smpl_mart_nm}
    """, conn)['샘플링배수'].tolist()
    
    print(f"[LOG] 사용가능 샘플링 비율: {list_mult}")
    
    # PRIMARY KEY
    list_PK       = df_mdl_params['list_PK']
    
    # 2차모델 사용 컬럼
    if v_model_turn in [2,3,4]:
        # 2차모델
        # num_slctVar : 변수선택시 중요도 상위 n개
        # list_FTR : 변수선택할 변수 리스트
        num_slctVar     = df_mdl_params['num_slctVar']
        num_dev_slctVar = df_mdl_params['num_dev_slctVar']
        list_Cust_FTR   = df_mdl_params['list_Cust_FTR']
        ################################################
        ## 2차 모델링 변수 상위 변수중요도 또는 json file내 선택변수 리스트
        ################################################
        if len(df_mdl_params['list_FTR']) == 0:
            list_FTR = pd.read_pickle(f'{save_dir[0]}/df_mdl_FI_{v_tgt_var}_{baseYm}_1_grpby.pkl')['var'].tolist()[:num_slctVar]
            print(f"[LOG] 선택 변수 리스트 : {list_FTR}")
        
        else:
            list_FTR = df_mdl_params['list_FTR']
            
        ################################################
        ## 3차 모델링 변수 전체 파생변수 생성
        ################################################    
        if v_model_turn == 3:
            list_DERV_FTR = list_FTR
            list_DERV_FTR = list(set(list_DERV_FTR) - set(list_Cust_FTR))
        
        ################################################
        ## 3차 모델링 변수 선택 파생변수 생성
        ################################################
        if v_model_turn == 4:
            if len(df_mdl_params['list_DERV_FTR']) == 0:
                tmp = pd.read_pickle(f'{save_dir[2]}/df_mdl_FI_{v_tgt_var}_{baseYm}_3_grpby.pkl')
                list_DERV_FTR = (
                                    tmp.loc[tmp['var'].str.contains('m_sum|m_avg'), 'var']
                                    .str
                                    .rsplit(pat = '_', n = 2)
                                    .str[0]
                                    .drop_duplicates()
                                    .reset_index(drop = True)[:num_dev_slctVar]
                                    .tolist()
                                )
            else:
                list_DERV_FTR = df_mdl_params['list_DERV_FTR']
                
    
    target_yn = 'Y'
    
 
    ##################################################################################################
    ########## 훈련마트
    ##################################################################################################
    for v_mult in list_mult:
    
        ################################################
        ## 1차 모델링 변수
        ################################################
        if v_model_turn == 1:
            
            df_TRAIN_MART = pd.read_sql(
            f"""
            SELECT
                  A20.*
                , A10.{target_yn} AS Y
            FROM {v_smpl_mart_nm} A10
            
            LEFT JOIN(
                SELECT
                    *
                FROM {v_ftr_mart_nm}
                WHERE 기준년월 BETWEEN '{v_train_st_baseYm}' AND '{v_train_ed_baseYm}'
            ) A20
            ON {' AND '.join([('A10.'+ x + ' = ' + 'A20.' + x) for x in list_PK])}
            
            WHERE
                A10.샘플링배수 = {v_mult}
            AND A10.타겟       = '{v_tgt_var}'
            AND A10.기준년월 BETWEEN '{v_train_st_baseYm}' AND '{v_train_ed_baseYm}'
            """, conn)

            
        ##################################################################
        ########## 2차모델링 변수 -- 변수선택
        ##################################################################
        else:
            # 1차모델링 마트 존재할시
            try:
                # 날짜 추가해야할듯.
                df_TRAIN_MART = pd.read_pickle(f"{save_dir[0]}/trn_mart_{v_tgt_var}_{baseYm}_{v_mult}_1.pkl")[list_PK + list_FTR + ['Y']]
                
            # 1차모델링 마트 존재하지 않을시 쿼리를 통한 데이터 추출
            except Exception as error:
                ################################################
                ## 훈련마트
                ################################################
                df_TRAIN_MART = pd.read_sql(f"""
                SELECT
                      A20.*
                    , A10.{target_yn} AS Y
                FROM {v_smpl_mart_nm} A10
                
                LEFT JOIN(
                    SELECT
                        {', '.join(list_PK + list_FTR)}
                    FROM {v_ftr_mart_nm}
                    WHERE 기준년월 BETWEEN '{v_train_st_baseYm}' AND '{v_train_ed_baseYm}'
                ) A20
                ON {' AND '.join([('A10.'+ x + ' = ' + 'A20.' + x) for x in list_PK])}
                
                WHERE
                    A10.샘플링배수 = {v_mult}
                AND A10.타겟       = '{v_tgt_var}'
                AND A10.기준년월 BETWEEN '{v_train_st_baseYm}' AND '{v_train_ed_baseYm}'
                """, conn)
            
        ##################################################################
        ########## 3차모델링 변수 -- 파생변수 추가
        ##################################################################
        ##################################################################
        ########## 현재 sqlite상 평균과 합만 계산 가능하여 다른 엔진에서 사용할 수 있는 추가 적인 파생변수 개발 필요함 ex) std, cv(=std/평균)
        ##################################################################
            if v_model_turn in [3,4]:
                # 파생변수를 만들기위해 group by 할 컬럼
                list_key = list(set(list_PK) - set(['기준년월']))

                # 훈련 시작 기준년월 설정
                v_tmp = v_train_st_baseYm
                
                #############################
                ## 훈련마트를 만들 기준년월 리스트 생성
                #############################
                list_train_ym = []
                i = 0
                while 1:
                    # 최대 100번 시행
                    if i == 100:
                        break
                    # max 값줘서 break 걸게끔 처리
                    if v_tmp == v_train_ed_baseYm:
                        list_train_ym.append(v_tmp)
                        break
                    list_train_ym.append(v_tmp)
                    # 기준년월 증가
                    v_tmp = datetime.strftime(datetime.strptime(v_tmp +'01','%Y%m%d') + relativedelta(months = 1),'%Y%m')  
                    i = i + 1

                #################################################
                ## 훈련마트 파생변수
                #################################################
                
                # 생성할 훈련마트 기준년월수 반복
                list_TRAIN_MART = []
                for ym in list_train_ym:
                    df_mart_tmp = pd.DataFrame()
                    
                    # 생성할 파생변수 반복
                    for coli,colx in enumerate(list_DERV_FTR):
                        # n개월 파생변수를 구하기위해 n-1개월전 기준년월 계산
                        ym_3m = datetime.strftime(datetime.strptime(ym +'01','%Y%m%d') - relativedelta(months = v_period_derv),'%Y%m')
                        df_tmp = pd.read_sql(f"""
                        SELECT
                              '{ym}' as 기준년월
                            , {','.join([('A20.' + x) for x in list_key])}
                            , A20.{colx}_{v_period_derv}m_sum
                            , A20.{colx}_{v_period_derv}m_avg
                            
                        FROM(
                            SELECT DISTINCT
                                  {','.join(list_PK)}
                                , 샘플링배수
                                , 타겟
                            FROM {v_smpl_mart_nm}
                        ) A10
                        
                        LEFT JOIN(
                            SELECT
                                {', '.join(list_key)}
                                , SUM({colx}) as {colx}_{v_period_derv}m_sum
                                , AVG({colx}) as {colx}_{v_period_derv}m_avg
                            FROM {v_ftr_mart_nm}
                            WHERE
                                기준년월 BETWEEN '{ym_3m}' AND '{ym}'
                            GROUP BY {','.join(list_key)}
                        ) A20
                        ON {' AND '.join([('A10.'+ x + ' = ' + 'A20.' + x) for x in list_key])}

                        WHERE
                            A10.샘플링배수 = {v_mult}
                        AND A10.타겟       = '{v_tgt_var}'
                        AND A10.기준년월   = '{ym}'
                        """, conn)
                        
                        if coli == 0:
                            df_mart_tmp = df_tmp
                        else:
                            df_mart_tmp = df_mart_tmp.merge(df_tmp
                                                          , on  = list_PK
                                                          , how = 'inner')
                                                          
                    list_TRAIN_MART.append(df_mart_tmp)
                    
                df_TRAIN_MART_2 = pd.concat(list_TRAIN_MART, axis = 0)
                # 2차모델링에 사용한 마트에 join하여 3차모델링 마트 구성
                df_TRAIN_MART  = df_TRAIN_MART.merge(df_TRAIN_MART_2
                                                   , on  = list_PK
                                                   , how = 'left')
        print(f"[LOG] {v_model_turn}차 마트생성 >> 샘플링배수: {v_mult} TRAIN 마트 추출 ROW 건수 : {df_TRAIN_MART.shape[0]}, TRAIN 마트 COLUMN 추출 건수 : {df_TRAIN_MART.shape[1]}")
        df_TRAIN_MART.to_pickle(f"{save_dir[v_model_turn - 1]}/df_train_mart_{v_tgt_var}_{baseYm}_{v_mult}_{v_model_turn}.pkl")
        
  
    ##################################################################################
    ## 테스트마트
    ##################################################################################
    ##################################################################
    ########## 1차모델링 변수
    ##################################################################
    if v_model_turn == 1:
        
        df_TEST_MART = pd.read_sql(
        f"""
        SELECT
              A10.*
            , COALESCE(A20.Y, 0) AS Y
        FROM {v_ftr_mart_nm} A10
        
        LEFT JOIN(
            SELECT DISTINCT
                  {','.join(list_PK)}
                , Y_{v_tgt_var} AS Y
            FROM {v_tgt_mart_nm}
        ) A20
        ON {' AND '.join([('A10.'+ x + ' = ' + 'A20.' + x) for x in list_PK])}
        
        WHERE
            A10.기준년월 BETWEEN '{v_test_st_baseYm}' AND '{v_test_ed_baseYm}'
        """, conn)
    else:
        ##################################################################
        ########## 2차모델링 변수 -- 변수선택
        ##################################################################
        try:
            
            df_TEST_MART  = pd.read_pickle(f"{save_dir[0]}/df_test_mart_{v_tgt_var}_{baseYm}_1.pkl")[list_PK + list_FTR + ['Y']]
            
        # 1차모델링 마트 존재하지 않을시 쿼리를 통한 데이터 추출
        except Exception as error:
            df_TEST_MART = pd.read_sql(f"""
            SELECT
                 {', '.join('A10.' + x for x in (list_PK + list_FTR))}
                , COALESCE(A20.Y, 0) AS Y
            FROM {v_ftr_mart_nm} A10
            
            LEFT JOIN(
                SELECT DISTINCT
                      {','.join(list_PK)}
                    , Y_{v_tgt_var} AS Y
                FROM {v_tgt_mart_nm}
            ) A20
            ON {' AND '.join([('A10.'+ x + ' = ' + 'A20.' + x) for x in list_PK])}
            
            WHERE
                A10.기준년월 BETWEEN '{v_test_st_baseYm}' AND '{v_test_ed_baseYm}'
            """, conn)
            
        ##################################################################
        ########## 3차모델링 변수 -- 파생변수 추가
        ##################################################################
        
        if v_model_turn in [3,4]:
            # 파생변수를 만들기위해 group by 할 컬럼
            list_key = list(set(list_PK) - set(['기준년월']))

            # 테스트 시작 기준년월 설정
            v_tmp = v_test_st_baseYm
            
            #############################
            ## 테스트마트를 만들 기준년월 리스트 생성
            #############################            
            list_test_ym = [] 
            i = 0
            while 1:
                # 최대 100번 시행
                if i == 100:
                    break
                # max 값줘서 break 걸게끔 처리
                if v_tmp == v_test_ed_baseYm:
                    list_test_ym.append(v_tmp)
                    break
                list_test_ym.append(v_tmp)
                # 기준년월 증가
                v_tmp = datetime.strftime(datetime.strptime(v_tmp +'01','%Y%m%d') + relativedelta(months = 1),'%Y%m')
                i = i + 1
            ##################################################
            ## 테스트마트 파생변수
            ##################################################
            # 생성할 훈련마트 기준년월수 반복
            list_TEST_MART  = []
            for ym in list_test_ym:
                # df_mart_tmp = pd.DataFrame()
                
                # n개월 파생변수를 구하기위해 n-1개월전 기준년월 계산
                ym_3m = datetime.strftime(datetime.strptime(ym +'01','%Y%m%d') - relativedelta(months = v_period_derv),'%Y%m')
                for coli, colx in enumerate(list_DERV_FTR):
                    # 3개월 rolling 변수 생성
                    df_tmp  = pd.read_sql(
                    f"""
                    SELECT
                        '{ym}' as 기준년월
                        , {','.join(list_key)}
                        , SUM({colx}) as {colx}_{v_period_derv}m_sum
                        , AVG({colx}) as {colx}_{v_period_derv}m_avg
                    FROM {v_ftr_mart_nm}
                    WHERE
                        기준년월 BETWEEN '{ym_3m}' AND '{ym}'
                    GROUP BY {','.join(list_key)}
                    """, conn)
                    
                    if coli == 0:
                        df_mart_tmp = df_tmp
                    else:
                        df_mart_tmp = df_mart_tmp.merge(df_tmp
                                                      , on  = list_PK
                                                      , how = 'inner')
                list_TEST_MART.append(df_mart_tmp)
                
            df_TEST_MART_2 = pd.concat(list_TEST_MART, axis = 0).reset_index(drop = True)
            df_TEST_MART   = df_TEST_MART.merge(df_TEST_MART_2
                                              , on = list_PK
                                              , how = 'left')

    print(f"[LOG] {v_model_turn}차 마트생성 >> TEST 마트 추출 ROW 건수 : {df_TEST_MART.shape[0]}, TEST 마트 COLUMN 추출 건수  : {df_TEST_MART.shape[1]}")
    df_TEST_MART.to_pickle(f"{save_dir[v_model_turn - 1]}/df_test_mart_{v_tgt_var}_{baseYm}_{v_model_turn}.pkl")
    
    
# 챔피언마트 생성 함수
def func_make_data_chmp(list_use_col, json_params, mdl_baseYm = None):

    ####################################################################################################
    # 기준년월 
    ####################################################################################################
    if mdl_baseYm != None:
        baseYm = mdl_baseYm
    else:
        baseYm = datetime.strftime(datetime.now() - relativedelta(months = 1), '%Y%m')
        
    baseYm_bfr = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months = 2), '%Y%m')
    
    
    ####################################################################################################
    # 저장 디렉토리 확인
    ####################################################################################################
    # json_params   = pd.read_json("C:/PythonProject/MLOps솔루션/data/dev/김규형/json/3.Modeling.json")
    st_dir = os.getcwd()
    df_mdl_params  = json_params['modeling_parameter'].dropna()
    setting_params = json_params['setting_parameter'].dropna()
    
    
    # 시작년월 기준년월 형식확인
    df_temp = pd.Series([df_mdl_params['train_st_baseYm']
                       , df_mdl_params['train_ed_baseYm']
                       , df_mdl_params['test_st_baseYm']
                       , df_mdl_params['test_ed_baseYm']])
    if ((sum(df_temp.apply(lambda x : len(x)) != 6) == 0) & (df_temp[0] <= df_temp[1]) & (df_temp[2] <= df_temp[3])) | (sum([1 if isinstance(x, str) else 0  for x in df_temp]) == 4):
        v_train_st_baseYm = df_mdl_params['train_st_baseYm']
        v_train_ed_baseYm = df_mdl_params['train_ed_baseYm']
        v_test_st_baseYm  = df_mdl_params['test_st_baseYm']
        v_test_ed_baseYm  = df_mdl_params['test_ed_baseYm']
    
    else:        
        # 기준년월시점에서 답지 없음 (Y값이 존재하지않아 훈련 및 검증하는데 사용 할 수 없음)
        # 기준년월시점 1달전 테스트마트 (1달)
        # 시준년월시점 13달전~2달전 훈련마트 (12달)
        v_train_st_baseYm = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months = 13), '%Y%m')
        v_train_ed_baseYm = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months = 2 ), '%Y%m')
        v_test_st_baseYm  = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months = 1 ), '%Y%m')
        v_test_ed_baseYm  = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months = 1 ), '%Y%m')
    
    # if flag = 1:
    #     v_test_st_baseYm = baseYm
    #     v_test_ed_baseYm = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months = 5), '%Y%m')
    test_dir = f"{os.getcwd()}{setting_params['test_dir']}"
    
    
    v_tgt_var = df_mdl_params['v_tgt_var']
    list_PK   = df_mdl_params['list_PK']
    target_yn = 'Y'
    
    v_ftr_mart_nm   = df_mdl_params['ftr_mart_nm']
    v_smpl_mart_nm  = df_mdl_params['smp_mart_nm']
    v_tgt_mart_nm   = df_mdl_params['tgt_mart_nm']
    
    
    v_tmp = 0
    if len([x for x in list_use_col if "m_sum" in x]) != 0:
        # 파생변수를 생성한 기간 추출
        v_period_derv = int([x.split(f"m_sum")[0] for x in list_use_col if  f"m_sum"     in x][0].split("m_sum")[0][-1])
        
        # 파생변수를 생성한 변수 목록 추출
        list_DERV_FTR = [x.split(f"_{v_period_derv}m_sum")[0] for x in list_use_col if  f"_{v_period_derv}m_sum"     in x]
        
        # 사용한 변수 목록 추출
        list_FTR      = [x                                    for x in list_use_col if (f"_{v_period_derv}m_sum" not in x) & (f"{v_period_derv}m_avg" not in x)]
        
        baseYm_bfr = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months = v_period_derv - 1), '%Y%m')
        v_tmp_dev = 1
    
    else:
        list_FTR = list_use_col
    
   
    df_TEST_MART = pd.read_sql(
    f"""
    SELECT
          A10.*
        , COALESCE(A20.Y, 0) AS Y
    FROM {v_ftr_mart_nm} A10
    
    LEFT JOIN(
        SELECT DISTINCT
              {','.join(list_PK)}
            , Y_{v_tgt_var} AS Y
        FROM {v_tgt_mart_nm}
    ) A20
    ON {' AND '.join([('A10.'+ x + ' = ' + 'A20.' + x) for x in list_PK])}
    
    WHERE
        A10.기준년월 BETWEEN '{v_test_st_baseYm}' AND '{v_test_ed_baseYm}'

    """, conn)
    
    
    list_key = list(set(list_PK) - set(['기준년월']))
    
    # 파생변수 생성 필요시
    if v_tmp_dev == 1:
        list_key = list(set(list_PK) - set(['기준년월']))

        # 테스트 시작 기준년월 설정
        v_tmp = v_test_st_baseYm
        list_test_ym = []
        i = 0
        while 1:
            # 최대 100번 시행
            if i == 100:
                break
            # max 값줘서 break 걸게끔 처리
            if v_tmp == v_test_ed_baseYm:
                list_test_ym.append(v_tmp)
                break
            list_test_ym.append(v_tmp)
            # 기준년월 증가
            v_tmp = datetime.strftime(datetime.strptime(v_tmp +'01','%Y%m%d') + relativedelta(months = 1),'%Y%m')
            i = i + 1
            
        ##################################################
        ## 테스트마트 파생변수
        ##################################################
        # 생성할 훈련마트 기준년월수 반복
        list_TEST_MART  = []
        for ym in list_test_ym:
            # df_mart_tmp = pd.DataFrame()
            
            # n개월 파생변수를 구하기위해 n-1개월전 기준년월 계산
            ym_3m = datetime.strftime(datetime.strptime(ym +'01','%Y%m%d') - relativedelta(months = v_period_derv),'%Y%m')
            for coli, colx in enumerate(list_DERV_FTR):
                # 3개월 rolling 변수 생성
                df_tmp  = pd.read_sql(
                f"""
                SELECT
                    '{ym}' as 기준년월
                    , {','.join(list_key)}
                    , SUM({colx}) as {colx}_{v_period_derv}m_sum
                    , AVG({colx}) as {colx}_{v_period_derv}m_avg
                FROM {v_ftr_mart_nm}
                WHERE
                    기준년월 BETWEEN '{ym_3m}' AND '{ym}'
                GROUP BY {','.join(list_key)}
                """, conn)
                
                if coli == 0:
                    df_mart_tmp = df_tmp
                else:
                    df_mart_tmp = df_mart_tmp.merge(df_tmp
                                                  , on  = list_PK
                                                  , how = 'inner')
            list_TEST_MART.append(df_mart_tmp)
                  
        df_TEST_MART_2 = pd.concat(list_TEST_MART, axis = 0).reset_index(drop = True)
    
        # 2차모델링에 사용한 마트에 join하여 3차모델링 마트 구성
        df_TEST_MART   = df_TEST_MART.merge(df_TEST_MART_2
                                          , on = list_PK
                                          , how = 'left')
                                      
    
    df_TEST_MART = df_TEST_MART[list_PK + list_use_col + ['Y']]
    df_TEST_MART.to_pickle(f"{test_dir}/df_test_chmp_mart_{baseYm}.pkl")
    
    
    
# 예측마트 생성 함수
def func_make_data_prd(list_use_col, json_params, mdl_baseYm = None):

    ####################################################################################################
    # 기준년월 
    ####################################################################################################
    if mdl_baseYm != None:
        baseYm = mdl_baseYm
    else:
        baseYm = datetime.strftime(datetime.now() - relativedelta(months = 1), '%Y%m')
    
    
    ####################################################################################################
    # 저장 디렉토리 확인
    ####################################################################################################
    # json_params   = pd.read_json("C:/PythonProject/MLOps솔루션/data/dev/김규형/json/3.Modeling.json")
    st_dir = os.getcwd()
    df_mdl_params  = json_params['modeling_parameter'].dropna()
    setting_params = json_params['setting_parameter'].dropna()
    
    v_period_derv = df_mdl_params['period_derv']
    
    pred_dir = f"{os.getcwd()}{setting_params['pred_dir']}"
    
    list_PK       = df_mdl_params['list_PK']
    v_ftr_mart_nm = df_mdl_params['ftr_mart_nm']
    
    # 파생변수 생성기간 생성
    baseYm_bfr = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months = v_period_derv - 1), '%Y%m')
    
    v_tmp = 0
    if len([x for x in list_use_col if "m_sum" in x]) != 0:
        v_period_derv = int([x.split(f"m_sum")[0] for x in list_use_col if  f"m_sum"     in x][0].split("m_sum")[0][-1])
        
        list_DERV_FTR = [x.split(f"_{v_period_derv}m_sum")[0] for x in list_use_col if  f"_{v_period_derv}m_sum"     in x]
        list_FTR      = [x                                    for x in list_use_col if (f"_{v_period_derv}m_sum" not in x) & (f"{v_period_derv}m_avg" not in x)]
        
        baseYm_bfr = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') - relativedelta(months = v_period_derv - 1), '%Y%m')
        v_tmp_dev = 1
    
    else:
        list_FTR = list_use_col
    
    
    df_PRED_MART = pd.read_sql(f"""
    SELECT
        {', '.join(list_PK + list_FTR)}
    FROM {v_ftr_mart_nm} A10
    WHERE
        A10.기준년월 == '{baseYm}'
    """, conn)
    
    
    list_key = list(set(list_PK) - set(['기준년월']))
    
    # 파생변수 생성 필요시
    if v_tmp_dev == 1:
        for coli, colx in enumerate(list_DERV_FTR):
            df_tmp = pd.read_sql(f"""
            SELECT
                '{baseYm}'    as 기준년월
                , {','.join(list_key)}
                , SUM({colx}) as {colx}_{v_period_derv}m_sum
                , AVG({colx}) as {colx}_{v_period_derv}m_avg
            FROM {v_ftr_mart_nm}
            WHERE
                기준년월 BETWEEN '{baseYm_bfr}' AND '{baseYm}'
            GROUP BY {','.join(list_key)}
            """, conn)
            if coli == 0:
                df_mart_tmp = df_tmp
            else:
                df_mart_tmp = df_mart_tmp.merge(df_tmp
                                              , on  = list_PK
                                              , how = 'inner')
        df_PRED_MART   = df_PRED_MART.merge(df_mart_tmp
                                          , on = list_PK
                                          , how = 'left')
                                          
                                          
    df_PRED_MART = df_PRED_MART[list_PK + list_use_col]
    df_PRED_MART.to_pickle(f"{pred_dir}/pred_mart_{baseYm}.pkl")
    