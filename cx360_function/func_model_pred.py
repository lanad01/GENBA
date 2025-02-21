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

def func_model_pred(df_mart, mdl, list_use_ftr, cutoff, json_params, mdl_baseYm = None):
    # 배치작업/작업명 선언

    ####################################################################################################
    # 저장 디렉토리 확인
    ####################################################################################################
    st_dir = os.getcwd()
    df_mdl_params  = json_params['modeling_parameter'].dropna()
    setting_params = json_params['setting_parameter'].dropna()
    pred_dir       = st_dir + setting_params['pred_dir']

    ####################################################################################################
    # 예측마트의 기준년월로 예측값의 기준년월 설정
    ####################################################################################################

    if mdl_baseYm == None:
        baseYm   = datetime.strftime(datetime.now() - relativedelta(months = 1), '%Y%m')
    baseYm = mdl_baseYm
    predYm = datetime.strftime(datetime.strptime(baseYm + "01", '%Y%m%d') + relativedelta(months = 1), '%Y%m')
    baseYm0 = baseYm

    ########################################################################################################################
    # 배치작업 시작
    ########################################################################################################################
    df_cust = df_mart[['고객ID']]
    # df_mart = df_mart.drop(['고객ID'], axis = 1)
    df_X    = (
        df_mart
        .drop(columns = ["고객ID", "기준년월"])[list_use_ftr]
        .fillna(0)
    )
    
    df_pred_Y      = pd.DataFrame({'Score' : mdl.predict_proba(df_X)})
    df_pred_Y['Y'] = df_pred_Y.apply(lambda x: 1 if x >= cutoff else 0)
    df_pred_Y      = pd.concat(df_cust, df_pred_Y)
    
    df_pred_Y.to_pickle(f"{pred_dir}/df_pred_{baseYm}.pkl")