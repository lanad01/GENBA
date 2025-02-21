from collections import Counter
# 함수생성
def func_pps_ideal(df_mart_1, dict_json): 
    
    # 최빈값 계산 함수 생성
    def calculate_mode(numbers):
        counter = Counter(numbers)
        common_values = counter.most_common(2)
        
        if len(common_values) > 1:
            mode = common_values[1][0]
        else:
            mode = common_values[0][0]
            
        return mode
    
    # 변수 리스트 생성
    df_var_list = pd.DataFrame(df_mart_1.dtypes, columns = ["변수유형"]).reset_index().rename(columns = {"index" : "변수명"})

    # 연속형 변수 리스트 생성
    df_num_list = (
        df_var_list[
             (df_var_list["변수유형"] != 'object')
             & ~(df_var_list["변수명"].isin(dict_json.get("v_exc_num", [])))
             & ~(df_var_list["변수명"].isin(dict_json.get("v_exc_ideal", [])))
        ]
    ) 
    
    list_num = df_num_list["변수명"].tolist()
    
    # 예외처리 변수 리스트 생성(건수, 평균)
    list_exc_num = [v_exc for v_exc in list_num if v_exc.endswith("건수") or v_exc.endswith("평균")]
    
    # IQR Method 적용 변수 리스트 생성
    list_ideal = list(set(list_num) - set(list_exc_num))
 
    # 이상치처리 변수 저장 데이터프레임 생성
    df_outliers = pd.DataFrame(columns=["변수명", "변수개수", "처리방법"])
    
    # 사본 생성
    df_mart_2 = df_mart_1.copy()
    
    # IQR Method를 활용한 이상치 처리
    for v_ideal in list_ideal:
        df_sorted = df_mart_2.sort_values(v_ideal, ascending=True)
        v_Q3 = np.percentile(df_sorted[v_ideal], 75)  # 변수의 75%값 계산
        v_Q1 = np.percentile(df_sorted[v_ideal], 25)  # 변수의 25%값 계산
        v_IQR = v_Q3 - v_Q1
        v_lower_bound = v_Q1 - 1.5 * v_IQR
        v_upper_bound = v_Q3 + 1.5 * v_IQR

        # 이상치 식별 (Q1 - 1.5*IQR 보다 작거나 Q3 + 1.5*IQR 보다 큰 값이거나 음수 값)
        v_outlier_lower = np.where((df_sorted[v_ideal] < v_lower_bound))[0]
        v_outlier_upper = np.where((df_sorted[v_ideal] > v_upper_bound))[0]
      # v_outlier_zero  = np.where((df_sorted[v_ideal] < 0))[0]

        # 이상치를 가지고 있는 변수의 이름, 개수, 처리 방법을 저장
        if len(v_outlier_lower) > 0:
            df_outliers = pd.concat([df_outliers, pd.DataFrame({
                "변수명" : [v_ideal],
                "변수개수": [len(v_outlier_lower)],
                "처리방법": [f"{v_lower_bound}로 대체(하한선)"]
            })], ignore_index=True)
            
        if len(v_outlier_upper) > 0:
            df_outliers = pd.concat([df_outliers, pd.DataFrame({
                "변수명": [v_ideal],
                "변수개수": [len(v_outlier_upper)],
                "처리방법": [f"{v_upper_bound}로 대체(상한선)"]
            })], ignore_index=True)
            
      # if len(v_outlier_zero) > 0:
      #     df_outliers = pd.concat([df_outliers, pd.DataFrame({
      #         "변수명": [v_ideal],
      #         "변수개수": [len(v_outlier_zero)],
      #         "처리방법": [f"{0}으로 대체(음수변환)"]
      #     })], ignore_index=True)            

        # 이상치 처리
        df_mart_2.loc[df_mart_2[v_ideal] < v_lower_bound, v_ideal] = v_lower_bound
        df_mart_2.loc[df_mart_2[v_ideal] > v_upper_bound, v_ideal] = v_upper_bound
      # df_mart_2.loc[df_mart_2[v_ideal] < 0, v_ideal] = 0
    
    # 예외변수에 대한 이상치처리(두번째 최빈값으로 대체)
    for v_exc_ideal in list_exc_num:
        
        # 두 번째 최빈값 계산
        mode = calculate_mode(df_mart_2[v_exc_ideal])
        
        # 이상치 식별 (두번째 최빈값의 10배수보다 큰 값)
        v_exc_outlier = np.where(df_mart_2[v_exc_ideal] > 10 * mode)[0]
        
        # 이상치를 가지고 있는 변수의 이름, 개수, 처리 방법을 저장
        if len(v_exc_outlier) > 0:
            df_outliers = pd.concat([df_outliers, pd.DataFrame({
                "변수명" : [v_exc_ideal],
                "변수개수": [len(v_exc_outlier)],
                "처리방법": [f"{mode}로 대체(두번째 최빈값)"]
            })], ignore_index=True)    
            
        # 두 번째 최빈값의 10배수를 초과하는 값을 두 번째 최빈값으로 대체
        df_mart_2.loc[df_mart_2[v_exc_ideal] > (10 * mode), v_exc_ideal] = mode 
        
    return df_mart_2, df_outliers

