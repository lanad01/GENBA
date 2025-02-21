########################################################################################################################
# 라이브러리 선언
########################################################################################################################
import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import re
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import OneHotEncoder
from pickle import dump, load

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import lightgbm as lgbm


def func_model_train(df_mart, v_model_turn, v_mult, json_params, mdl_baseYm):
    # 배치작업/작업명 선언
    # btch_nm = "2_3_modeling"
    # job_nm  = f"ModelTrain: {str(v_model_turn)}차 모델링" 
    baseYm = mdl_baseYm
        
    
    # json_params = pd.read_json("C:/PythonProject/MLOps솔루션/data/dev/김규형/json/3.Modeling.json")
    
    df_mdl_params  = json_params['modeling_parameter'].dropna()
    setting_params = json_params['setting_parameter'].dropna()
    rf_params      = json_params['rf_parameter_c'].dropna()
    xgb_params     = json_params['xgb_parameter_c'].dropna()
    lgb_params     = json_params['lgb_parameter_c'].dropna()
    
    ####################################################################################################
    # 저장 디렉토리 확인 및 함수호출
    ####################################################################################################
    st_dir = os.getcwd()
    
    save_dir = []
    save_dir.append(setting_params['1st_dir'])
    save_dir.append(setting_params['2nd_dir'])
    save_dir.append(setting_params['3rd_dir'])
    save_dir.append(setting_params['4th_dir'])
    func_dir = f"{st_dir}{setting_params['func_dir']}"
    
    # 사용자 정의 함수 호출
    exec(open(func_dir + f"/func_mdl_prfmnc.py", encoding='utf-8').read())
    
    if v_model_turn not in [1,2,3,4]:
        raise Exception("v_model_turn 값을 확인하세요 1,2,3 셋중에 하나를 입력하세요")
    
    ####################################################################################################
    # 초기 파라미터 정의
    ####################################################################################################
    v_algorithm_nm   = setting_params['v_algorithm_nm']
    v_valid_size     = df_mdl_params['valid_size']
    v_eval_index     = df_mdl_params['v_eval_index']
    v_tgt_var        = df_mdl_params['v_tgt_var']
    
    list_PK = df_mdl_params['list_PK']

    ####################################################################################################
    # 알고리즘 파라미터
    ####################################################################################################

    # json 파일 내 문자열 None을 None 객체로 변경 및 파라미터 정보 dictionary 형태로 할당
    rf_params[rf_params == "None"]   = None
    xgb_params[xgb_params == "None"] = None
    lgb_params[lgb_params == "None"] = None

    rf_params  = (
        rf_params
        .dropna()
        .to_dict()
    )
    xgb_params  = (
        xgb_params
        .dropna()
        .to_dict()
    )
    lgb_params  = (
        lgb_params
        .dropna()
        .to_dict()
    )
    ########################################################################################################################
    # 배치작업 시작
    ########################################################################################################################
    list_df_prfmnc = []

    ####################################################################################################
    # 모델 학습
    ####################################################################################################    
    ################################################################################
    # 데이터 추출
    ################################################################################

    # 컬럼 object 오류방지
    df_mart = df_mart.drop(list(set(df_mart.columns[df_mart.dtypes == object]) - set(list_PK + ['Y'])), axis = 1)
    df_mart = df_mart.fillna(0).astype(float)

    ################################################################################
    # 학습
    ################################################################################
        
    # lightGBM에서 특수문자보유컬럼 존재시 오류 -> 특수문자 제거
    df_mart = df_mart.rename(columns = lambda x:re.sub('[^\uAC00-\uD7A30-9a-zA-Z\s_]', '', x))
    
    # 마트데이터를 X와 Y로 분리
    df_X = df_mart.drop(columns = list_PK + ['Y'])
    df_Y = df_mart['Y'].astype(int)
    # 마트데이터를 학습과 검증으로 분리        
    df_X_train, df_X_valid, df_Y_train, df_Y_valid = train_test_split(df_X
                                                                    , df_Y
                                                                    , test_size    = v_valid_size
                                                                    , random_state = 0)
    
    
    
    # 모델파라미터 json 형태로 입력
    # RF모델 학습 및 저장 
    mdl_rf = RandomForestClassifier().set_params(**rf_params)
    mdl_rf.fit(df_X_train, df_Y_train.values.ravel())
    dump(mdl_rf,   open(save_dir[v_model_turn - 1] + f'/mdl_{v_tgt_var}_rf_{baseYm}_{v_mult}_{v_model_turn}.pkl', 'wb'))

    # XGBOOST모델 학습 및 저장
    mdl_xgb  = XGBClassifier().set_params(**xgb_params)
    mdl_xgb.fit(df_X_train, df_Y_train.values.ravel())
    dump(mdl_xgb,  open(save_dir[v_model_turn - 1] + f'/mdl_{v_tgt_var}_xgb_{baseYm}_{v_mult}_{v_model_turn}.pkl', 'wb'))

    # LIGHTGBM모델 학습 및 저장
    mdl_lgb  = lgbm.LGBMClassifier().set_params(**lgb_params)
    mdl_lgb.fit(df_X_train, df_Y_train.values.ravel())
    dump(mdl_lgb, open(save_dir[v_model_turn - 1] + f'/mdl_{v_tgt_var}_lgb_{baseYm}_{v_mult}_{v_model_turn}.pkl', 'wb'))



    # RF모델 학습결과 생성
    df_Y_valid_rf         = pd.concat([df_Y_valid.reset_index(drop = True), pd.Series([x[1] for x in mdl_rf.predict_proba(df_X_valid)])], axis = 1)
    df_Y_valid_rf.columns = ['Y_Real', 'Y_Prob'] 

    # xgboost모델 학습결과 생성
    df_Y_valid_xgb         = pd.concat([df_Y_valid.reset_index(drop = True), pd.Series([x[1] for x in mdl_xgb.predict_proba(df_X_valid)])], axis = 1)
    df_Y_valid_xgb.columns = ['Y_Real', 'Y_Prob'] 

    # lightgbm모델 학습결과 생성
    df_Y_valid_lgb         = pd.concat([df_Y_valid.reset_index(drop = True), pd.Series([x[1] for x in mdl_lgb.predict_proba(df_X_valid)])], axis = 1)
    df_Y_valid_lgb.columns = ['Y_Real', 'Y_Prob'] 

    ################################################################################
    # 모델 성능 측정
    ################################################################################
    # 함수호출
    df_prfmnc_rf  = func_mdl_prfmnc(df_Y_valid_rf)
    df_prfmnc_xgb = func_mdl_prfmnc(df_Y_valid_xgb)
    df_prfmnc_lgb = func_mdl_prfmnc(df_Y_valid_lgb)
    
    # 사용 알고리즘명 입력
    df_prfmnc_rf['Model']  = 'rf'
    df_prfmnc_xgb['Model'] = 'xgb'
    df_prfmnc_lgb['Model'] = 'lgb'

    # 통합
    df_prfmnc = pd.concat([
          df_prfmnc_rf
        , df_prfmnc_xgb
        , df_prfmnc_lgb], axis = 0)
    
    # 모델 차수, 샘플링 배수 입력
    df_prfmnc['Mdl_step']   = v_model_turn
    df_prfmnc['샘플링배수'] = v_mult
    
    #데이터 컬럼 정렬
    df_prfmnc = df_prfmnc[['Mdl_step', '샘플링배수', 'Model', 'cutOff', 'R1', 'P1', 'N11', 'N10', 'N01', 'N00', '정밀도', '재현율', 'F1Score']]
    # list_df_prfmnc.append(df_prfmnc)

    # 해당 차수 모델링 모든 알고리즘 cutoff별 스코어분포
    # pd.concat(list_df_prfmnc).to_pickle(save_dir[v_model_turn - 1] + f"/df_prfmnc_{v_tgt_var}_{baseYm}_{v_model_turn}.pkl")
    
    
    ##########################################
    # 변수중요도 - RF
    ##########################################
    
    df_ftr_imp = pd.DataFrame({
          'var'       : df_X_train.columns
        , 'algorithm' : 'rf'
        , '중요도'    : mdl_rf.feature_importances_/mdl_rf.feature_importances_.sum()
    }).sort_values(['중요도'], ascending = False)
    
    df_ftr_imp.to_pickle(save_dir[v_model_turn - 1] + f"/df_mdl_FI_rf_{v_tgt_var}_{baseYm}_{v_mult}_{v_model_turn}.pkl")


    # one-hot encoding된 카테고리 변수를 통합하여 변수 중요도 집계 후 저장 ex) col_가, col_나 => col
    df_ftr_imp.loc[(df_ftr_imp['var'].str.rsplit('_').str[0].isin(df_onehot_encdr['변수명'])) & (df_ftr_imp['var'].str.contains('_')), 'tmp'] = df_ftr_imp['var'].str.rsplit('_').str[0]
    df_ftr_imp.loc[df_ftr_imp['tmp'].isnull(), 'tmp'] = df_ftr_imp.loc[df_ftr_imp['tmp'].isnull(), 'var'] 
    (
        df_ftr_imp
        .groupby(['algorithm', 'tmp'])['중요도']
        .sum()
        .reset_index()
        .sort_values('중요도', ascending = False)
        .reset_index(drop = True)
        .rename(columns = {'tmp' : 'var'})
        .to_pickle(save_dir[v_model_turn - 1] + f"/df_mdl_FI_rf_{v_tgt_var}_{baseYm}_{v_mult}_{v_model_turn}_grpby.pkl")
    )
    
    ##########################################
    # 변수중요도 - xgb
    ##########################################
    
    df_ftr_imp = pd.DataFrame({
          'var'       : df_X_train.columns
        , 'algorithm' : 'xgb'
        , '중요도'    : mdl_xgb.feature_importances_/mdl_xgb.feature_importances_.sum()
    }).sort_values(['중요도'], ascending = False)
    
    df_ftr_imp.to_pickle(save_dir[v_model_turn - 1] + f"/df_mdl_FI_xgb_{v_tgt_var}_{baseYm}_{v_mult}_{v_model_turn}.pkl")


    # one-hot encoding된 카테고리 변수를 통합하여 변수 중요도 집계 후 저장 ex) col_가, col_나 => col
    df_ftr_imp.loc[(df_ftr_imp['var'].str.rsplit('_').str[0].isin(df_onehot_encdr['변수명'])) & (df_ftr_imp['var'].str.contains('_')), 'tmp'] = df_ftr_imp['var'].str.rsplit('_').str[0]
    df_ftr_imp.loc[df_ftr_imp['tmp'].isnull(), 'tmp'] = df_ftr_imp.loc[df_ftr_imp['tmp'].isnull(), 'var'] 
    (
        df_ftr_imp
        .groupby(['algorithm', 'tmp'])['중요도']
        .sum()
        .reset_index()
        .sort_values('중요도', ascending = False)
        .reset_index(drop = True)
        .rename(columns = {'tmp' : 'var'})
        .to_pickle(save_dir[v_model_turn - 1] + f"/df_mdl_FI_xgb_{v_tgt_var}_{baseYm}_{v_mult}_{v_model_turn}_grpby.pkl")
    )
    
    #########################################
    # 변수중요도 - LGBM
    #########################################
    
    df_ftr_imp = pd.DataFrame({
          'var'       : df_X_train.columns
        , 'algorithm' : 'lgb'
        , '중요도'    : mdl_lgb.feature_importances_/mdl_lgb.feature_importances_.sum()
    }).sort_values(['중요도'], ascending = False)
    
    df_ftr_imp.to_pickle(save_dir[v_model_turn - 1] + f"/df_mdl_FI_lgb_{v_tgt_var}_{baseYm}_{v_mult}_{v_model_turn}.pkl")
        
    # one-hot encoding된 카테고리 변수를 통합하여 변수 중요도 집계 후 저장 ex) col_가, col_나 => col
    df_ftr_imp.loc[(df_ftr_imp['var'].str.rsplit('_').str[0].isin(df_onehot_encdr['변수명'])) & (df_ftr_imp['var'].str.contains('_')), 'tmp'] = df_ftr_imp['var'].str.rsplit('_').str[0]
    df_ftr_imp.loc[df_ftr_imp['tmp'].isnull(), 'tmp'] = df_ftr_imp.loc[df_ftr_imp['tmp'].isnull(), 'var'] 
    (
        df_ftr_imp
        .groupby(['algorithm', 'tmp'])['중요도']
        .sum()
        .reset_index()
        .sort_values('중요도', ascending = False)
        .reset_index(drop = True)
        .rename(columns = {'tmp' : 'var'})
        .to_pickle(save_dir[v_model_turn - 1] + f"/df_mdl_FI_lgb_{v_tgt_var}_{baseYm}_{v_mult}_{v_model_turn}_grpby.pkl")
    )
    
    return df_prfmnc
    