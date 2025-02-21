
# 함수생성
def func_pps_onehot(df_mart_1, dict_json):

    # 변수 리스트 생성
    df_var_list = pd.DataFrame(df_mart_1.dtypes, columns=["변수유형"]).reset_index().rename(columns={"index": "변수명"})

    # 범주형 변수 리스트 생성
    df_char_list = (
        df_var_list[
                (df_var_list["변수유형"] == 'object')
            & ~(df_var_list["변수명"].isin(dict_json.get("v_exc_char", [])))
        ]
    )
    # 특수문자 제거
    for col in df_char_list["변수명"]:
        df_mart_1[col] = df_mart_1[col].apply(lambda x:re.sub('[^\uAC00-\uD7A30-9a-zA-Z\s_]', '', x))
    
    df_mart_2 = df_mart_1.copy()

    # 원핫인코딩 적합
    oneHotEncdr = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore').fit(df_mart_1[df_char_list["변수명"]])

    # 원핫인코딩 적용
    tmp_cat = pd.DataFrame(
        oneHotEncdr.transform(df_mart_1[df_char_list["변수명"]]),
        columns=[x.replace(" ", "") for x in oneHotEncdr.get_feature_names_out(df_char_list["변수명"])]
    )

    # 데이터프레임 초기화
    df_onehot_encdr = pd.DataFrame(columns=['변수명', '변수개수', '처리방법'])

    # 속성값 개수와 처리방법 저장
    for v_onehot in df_char_list['변수명']:
        v_col_cnt = df_mart_2[v_onehot].nunique()
        df_onehot_encdr = pd.concat([df_onehot_encdr, pd.DataFrame({
            '변수명': [v_onehot],
            '변수개수': [v_col_cnt],
            '처리방법': ['원핫인코딩']
        })], ignore_index=True)

    # 원핫인코딩 정보로 대체
    df_mart_2 = df_mart_2[df_mart_2.columns[~df_mart_2.columns.isin(df_char_list["변수명"])]]
    
    df_mart_2 = pd.concat(
        [df_mart_2[df_mart_2.columns[~df_mart_2.columns.isin(dict_json.get("v_drp_char", []))]], tmp_cat],
         axis=1
    )
    
    return df_mart_2, df_onehot_encdr, oneHotEncdr
