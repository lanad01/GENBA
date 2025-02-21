
########################################################################################################################
# 라이브러리 선언                                                                                   
########################################################################################################################
import pandas as pd
import sqlite3
import os
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


########################################################################################################################
# 모델 성능 측정 함수                                                                                     
########################################################################################################################
def func_mdl_prfmnc(
          df_nm   # 성능 측정대상 df
        , cutoff = None
    ):
    
    list_prfmnc = []
    # 하나의 cutoff에 대한 성능 측정
    if cutoff != None:
        i = cutoff
        tmp_Y = df_nm.copy()
        
        tmp_Y["cutOff"] = i
        
        # 예측여부 판단
        tmp_Y["Y_Pred"] = tmp_Y["Y_Prob"].apply(lambda x: 1 if x >= i / 100 else 0)
        
        # 성능 판단
        tmp_Y["N11"] = np.where((tmp_Y["Y_Real"] == 1) & (tmp_Y["Y_Pred"] == 1), 1, 0)
        tmp_Y["N10"] = np.where((tmp_Y["Y_Real"] == 1) & (tmp_Y["Y_Pred"] == 0), 1, 0)
        tmp_Y["N01"] = np.where((tmp_Y["Y_Real"] == 0) & (tmp_Y["Y_Pred"] == 1), 1, 0)
        tmp_Y["N00"] = np.where((tmp_Y["Y_Real"] == 0) & (tmp_Y["Y_Pred"] == 0), 1, 0)
        
        # 집계
        tmp_Y_2 = (
            tmp_Y
            .groupby("cutOff", as_index = False)
            .agg(
                  R1  = ("Y_Real", 'sum')
                , P1  = ("Y_Pred", 'sum')
                , N11 = ("N11"   , 'sum')
                , N10 = ("N10"   , 'sum')
                , N01 = ("N01"   , 'sum')
                , N00 = ("N00"   , 'sum')
            )   
        )
        
        # 성능 계산
        tmp_Y_2["정밀도"]  = tmp_Y_2["N11"] / (tmp_Y_2["N11"] + tmp_Y_2["N01"])
        tmp_Y_2["재현율"]  = tmp_Y_2["N11"] / (tmp_Y_2["N11"] + tmp_Y_2["N10"])
        tmp_Y_2["F1Score"] = 2 * tmp_Y_2["정밀도"] * tmp_Y_2["재현율"] / (tmp_Y_2["정밀도"] + tmp_Y_2["재현율"])
        
        list_prfmnc.append(tmp_Y_2)
        df_prfmnc = pd.concat(list_prfmnc)
        
        return df_prfmnc
    
    # 모든 cutoff에 대한 성능 측정
    for i in range(1, 100):
        
        tmp_Y = df_nm.copy()
        
        tmp_Y["cutOff"] = i
        
        # 예측여부 판단
        tmp_Y["Y_Pred"] = tmp_Y["Y_Prob"].apply(lambda x: 1 if x >= i / 100 else 0)
        
        # 성능 판단
        tmp_Y["N11"] = np.where((tmp_Y["Y_Real"] == 1) & (tmp_Y["Y_Pred"] == 1), 1, 0)
        tmp_Y["N10"] = np.where((tmp_Y["Y_Real"] == 1) & (tmp_Y["Y_Pred"] == 0), 1, 0)
        tmp_Y["N01"] = np.where((tmp_Y["Y_Real"] == 0) & (tmp_Y["Y_Pred"] == 1), 1, 0)
        tmp_Y["N00"] = np.where((tmp_Y["Y_Real"] == 0) & (tmp_Y["Y_Pred"] == 0), 1, 0)
        
        # 집계
        tmp_Y_2 = (
            tmp_Y
            .groupby("cutOff", as_index = False)
            .agg(
                  R1  = ("Y_Real", 'sum')
                , P1  = ("Y_Pred", 'sum')
                , N11 = ("N11"   , 'sum')
                , N10 = ("N10"   , 'sum')
                , N01 = ("N01"   , 'sum')
                , N00 = ("N00"   , 'sum')
            )   
        )
        
        # 성능 계산
        tmp_Y_2["정밀도"]  = tmp_Y_2["N11"] / (tmp_Y_2["N11"] + tmp_Y_2["N01"])
        tmp_Y_2["재현율"]  = tmp_Y_2["N11"] / (tmp_Y_2["N11"] + tmp_Y_2["N10"])
        tmp_Y_2["F1Score"] = 2 * tmp_Y_2["정밀도"] * tmp_Y_2["재현율"] / (tmp_Y_2["정밀도"] + tmp_Y_2["재현율"])
        
        list_prfmnc.append(tmp_Y_2)
        
    df_prfmnc = pd.concat(list_prfmnc)
             
    return df_prfmnc
  
########################################################################################################################
# lift 측정 함수                                                                                     
########################################################################################################################
def func_mdl_prfmnc_lift(
          df_nm   # 성능 측정대상 df
        , list_cutoff = [10, 20, 30, 40, 50, 60, 70, 80, 90] # 측정 cutoff
    ):
    
    list_prfmnc = []
    for cutoff in list_cutoff:
        tmp_Y = df_nm.copy()
        
        tmp_Y["cutOff"] = cutoff
        
        # 예측여부 판단
        tmp_Y["Y_Pred"] = tmp_Y["Y_Prob"].apply(lambda x: 1 if x >= cutoff / 100 else 0)
        
        # 성능 판단
        tmp_Y["N11"] = np.where((tmp_Y["Y_Real"] == 1) & (tmp_Y["Y_Pred"] == 1), 1, 0)
        tmp_Y["N10"] = np.where((tmp_Y["Y_Real"] == 1) & (tmp_Y["Y_Pred"] == 0), 1, 0)
        tmp_Y["N01"] = np.where((tmp_Y["Y_Real"] == 0) & (tmp_Y["Y_Pred"] == 1), 1, 0)
        tmp_Y["N00"] = np.where((tmp_Y["Y_Real"] == 0) & (tmp_Y["Y_Pred"] == 0), 1, 0)
        
        # 집계
        tmp_Y_2 = (
            tmp_Y
            .groupby("cutOff", as_index = False)
            .agg(
                  CNT = ("Y_Real", 'count')
                , R1  = ("Y_Real", 'sum')
                , P1  = ("Y_Pred", 'sum')
                , N11 = ("N11"   , 'sum')
                , N10 = ("N10"   , 'sum')
                , N01 = ("N01"   , 'sum')
                , N00 = ("N00"   , 'sum')
            )   
        )
        
        # prob >= cutoff일 때 1 예측건수
        v_tmp  = (tmp_Y_2["N11"] + tmp_Y_2["N01"]).values[0]
        
        # 모델 반응률
        tmp_Y_2['마케팅대상고객수'] = v_tmp
        tmp_Y_2['mdl_반응률']       = tmp_Y_2['N11'] / v_tmp
        
        # 랜덤 반응률 (복원추출해서 평균 산출)
        list_tmp_prc = []
        for i in range(100):
            tmp_rdn_avg = tmp_Y.sample(n            = v_tmp
                                     , random_state = 0
                                     , replace      = True)['Y_Real'].sum() / v_tmp
            list_tmp_prc.append(tmp_rdn_avg)
        
        # 100번 반복하여 평균치 산출하여 최종 랜덤반응률 산출
        tmp_Y_2["rnd_반응률"] = np.mean(list_tmp_prc)
        
        # 성능 계산
        tmp_Y_2["Lift"]    = tmp_Y_2['mdl_반응률'] / tmp_Y_2['rnd_반응률']
        
        list_prfmnc.append(tmp_Y_2)
        
    df_prfmnc = pd.concat(list_prfmnc)
             
    return df_prfmnc
  
