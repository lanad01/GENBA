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

from sklearn.preprocessing import OneHotEncoder
from pickle import dump, load

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import lightgbm as lgbm

def func_model_test(df_mart, v_model_turn, json_params, mdl_baseYm = None):

    if mdl_baseYm != None:
        baseYm = mdl_baseYm
    else:
        baseYm = datetime.strftime(datetime.now() - relativedelta(months = 1), '%Y%m')
    
    # json_params = pd.read_json("C:/PythonProject/MLOps솔루션/data/dev/김규형/json/3.Modeling.json")
    
    df_mdl_params  = json_params['modeling_parameter'].dropna()
    setting_params = json_params['setting_parameter'].dropna()
    
    ####################################################################################################
    # 저장 디렉토리 확인
    ####################################################################################################
    
    st_dir = os.getcwd()
    save_dir = []
    save_dir.append(setting_params['1st_dir'])
    save_dir.append(setting_params['2nd_dir'])
    save_dir.append(setting_params['3rd_dir'])
    save_dir.append(setting_params['4th_dir'])
    
    test_dir = setting_params['test_dir']
    chmp_dir = setting_params['chmp_dir']
    func_dir = setting_params['func_dir']
    
    # 사용자 정의 함수 호출
    exec(open(f"{st_dir}{func_dir}/func_mdl_prfmnc.py", encoding='utf-8').read())
    
    if v_model_turn not in [1,2,3,4,'C']:
        raise Exception("v_model_turn 값을 확인하세요 1, 2, 3, 4, 'C' 셋중에 하나를 입력하세요")
        
    ####################################################################################################
    # 초기 파라미터 정의
    ####################################################################################################
    
    v_algorithm_nm = setting_params['v_algorithm_nm']
                   
    v_eval_index   = df_mdl_params['v_eval_index']
    v_tgt_var      = df_mdl_params['v_tgt_var']
    list_PK        = df_mdl_params['list_PK']

    ####################################################################################################
    # 차수별 최종모델 LOAD
    ####################################################################################################
    
    if v_model_turn in [1,2,3,4]:
        df_model_info = pd.read_pickle(f"{save_dir[v_model_turn - 1]}/df_info_mdl_{v_tgt_var}_{baseYm}_{v_model_turn}.pkl")
    elif v_model_turn == 'C':
        df_model_info = pd.read_pickle(f"{chmp_dir}/chmp_info_mdl_{v_tgt_var}_{v_model_turn}.pkl")

    v_cutoff = df_model_info['cutOff'].values[0]
    v_mult   = df_model_info['샘플링배수'].values[0]
    v_model  = df_model_info['Model'].values[0]



    ########################################################################################################################
    # 배치작업 시작
    ########################################################################################################################
    ####################################################################################################
    # 모델 TEST 지표 계산
    ####################################################################################################
    
    # 삭제 필요
    df_mart = df_mart.fillna(0).astype(float)
    
    v_cust = list(set(list_PK) - set(['기준년월']))[0]
    ################################################################################
    # TEST
    ################################################################################
    
    # lightGBM에서 특수문자보유컬럼 존재시 오류 -> 특수문자 제거
    df_mart = df_mart.rename(columns = lambda x:re.sub('[^\uAC00-\uD7A30-9a-zA-Z\s_]', '', x))
    
    # 마트데이터를 X와 Y로 분리
    df_PK        = df_mart[list_PK]
    df_X         = df_mart.drop(columns = list_PK+['Y'])
    df_Y         = df_mart['Y'].astype(int)


    # 최종 모델 LOAD
    if v_model_turn in [1,2,3,4]:
        mdl_final = load(open(f'{save_dir[v_model_turn - 1]}/mdl_eff_{v_tgt_var}_{baseYm}_{v_model_turn}.pkl', 'rb')) 
    elif v_model_turn == 'C':
        mdl_final = load(open(f'{chmp_dir}/mdl_chmp_{v_tgt_var}_{v_model_turn}.pkl', 'rb')) 

    # 모델 학습결과 생성
    df_Y_test         = pd.concat([df_Y.reset_index(drop = True), pd.Series([x[1] for x in mdl_final.predict_proba(df_X)])], axis = 1)
    df_Y_test.columns = ['Y_Real', 'Y_Prob'] 
    
    # validation 최종모델 정보 가져오기
    
    df_Y_test[list_PK]      = df_PK
    df_Y_test['Model']      = v_model
    df_Y_test['Mdl_step']   = v_model_turn
    df_Y_test['샘플링배수'] = v_mult
    df_Y_test = df_Y_test[list_PK + ['Mdl_step', '샘플링배수', 'Model', 'Y_Real', 'Y_Prob']]
    df_Y_test.to_pickle(test_dir + f"/df_score_dist_test_{v_tgt_var}_{v_model_turn}_{baseYm}.pkl")
    
    # 모델 성능 측정
    df_prfmnc = func_mdl_prfmnc(df_Y_test[['Y_Real', 'Y_Prob']])
    
    
    df_prfmnc['Model']      = v_model
    df_prfmnc['Mdl_step']   = v_model_turn
    df_prfmnc['샘플링배수'] = v_mult
    df_prfmnc = df_prfmnc[['Mdl_step', '샘플링배수', 'Model', 'cutOff', 'R1', 'P1', 'N11', 'N10', 'N01', 'N00', '정밀도', '재현율', 'F1Score']]
    df_prfmnc.to_pickle(test_dir + f"/df_prfmnc_test_{v_tgt_var}_{v_model_turn}_{baseYm}.pkl")
    
    
    # 모델 Lift 측정
    df_prfmnc_lift = func_mdl_prfmnc_lift(df_Y_test[['Y_Real', 'Y_Prob']])
    
    
    df_prfmnc_lift['Model']      = v_model
    df_prfmnc_lift['Mdl_step']   = v_model_turn
    df_prfmnc_lift['샘플링배수'] = v_mult
    df_prfmnc_lift = df_prfmnc_lift[['Mdl_step', '샘플링배수', 'Model', 'cutOff', 'R1', 'P1', 'N11', 'N10', 'N01', 'N00', '마케팅대상고객수', 'mdl_반응률', 'rnd_반응률', 'Lift']]
    df_prfmnc_lift.to_pickle(test_dir + f"/df_prfmnc_lift_test_{v_tgt_var}_{v_model_turn}_{baseYm}.pkl")