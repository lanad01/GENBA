
# 함수생성
def func_pps_missing(df_mart_1, dict_json) :   

    # 변수 리스트 생성
    df_var_list = pd.DataFrame(df_mart_1.dtypes, columns = ["변수유형"]).reset_index().rename(columns = {"index" : "변수명"})

    # 연속형 변수 리스트 생성
    df_num_list = (
        df_var_list[
             (df_var_list["변수유형"] != 'object')
             & ~(df_var_list["변수명"].isin(dict_json.get("v_exc_num", [])))
        ]
    )    
         
    # 범주형 변수 리스트 생성
    df_char_list = (
        df_var_list[
                (df_var_list["변수유형"] == 'object')
            & ~(df_var_list["변수명"].isin(dict_json.get("v_exc_char", [])))
        ]
    )
    
    # 연속형 변수 결측값에 대해 일괄적으로 0으로 대체, 범주형 변수 결측값에 대해 일괄적으로 '기타'로 대체
    df_mart_2 = df_mart_1.copy()
    df_mart_2[df_num_list["변수명"]] = df_mart_2[df_num_list["변수명"]].fillna(0)
    df_mart_2[df_char_list["변수명"]] = df_mart_2[df_char_list["변수명"]].fillna('기타')
    
    # 결측치 변수 리스트 생성
    df_missing_list = pd.DataFrame(df_mart_1.isnull().sum(), columns = ["변수개수"]).reset_index().rename(columns = {"index" : "변수명"})

    # 결측치 처리한 변수에 대한 설명을 담을 새로운 컬럼 추가
    df_missing_list["처리방법"] = ""
    
    df_missing_list = (
        df_missing_list[  
              (df_missing_list["변수개수"] != 0)
            & ~(df_missing_list["변수명"].isin(dict_json.get("v_exc_num", []))
            | df_missing_list["변수명"].isin(dict_json.get("v_exc_char", [])))
        ]
    ) 
    
    # 결측치 변수 리스트 생성
    list_missing = df_missing_list["변수명"].tolist()
    list_num = df_num_list["변수명"].tolist()
    list_char = df_char_list["변수명"].tolist()
    
    # 결측치 처리 방법 기록
    for v_missing in list_missing:
        if v_missing in list_num:
            df_missing_list.loc[df_missing_list["변수명"] == v_missing, "처리방법"] = '0'
        elif v_missing in list_char:
            df_missing_list.loc[df_missing_list["변수명"] == v_missing, "처리방법"] = '기타'
    
    return df_mart_2, df_missing_list
    