import pandas as pd
import numpy as np
import os

def summarize_data(df, df_name):
    """
    주어진 데이터프레임(df)에 대한 EDA(탐색적 데이터 분석) 결과를 정리하여 반환하는 함수.
    - 테이블 개요, 컬럼 개요, 범주형 변수 분포, 상관계수 정보를 포함
    """

    # ✅ 1. 이상치 탐지 (IQR 방식)
    outliers_info = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        if outlier_count > 0:
            outliers_info[col] = int(outlier_count)

    # ✅ 2. 테이블 개요 (데이터프레임 요약)
    table_summary = pd.DataFrame({
        "데이터프레임명": [df_name],
        "전체 데이터 크기 (행, 열)": [df.shape],
        "전체 컬럼 수": [df.shape[1]],
        "전체 조건 컬럼 수": [df.shape[0]],
        "범주형 변수 수": [len(df.select_dtypes(include=["object", "category"]).columns)],
        "연속형 변수 수": [len(df.select_dtypes(include=[np.number]).columns)],
        "이상치가 있는 컬럼 수": [len(outliers_info)],
        "이상치 총 개수": [sum(outliers_info.values())],
        "eda주제": ["테이블 개요"]
    })

    # ✅ 3. 컬럼 개요 (기본 통계 및 결측 정보 + 인스턴스 예제)
    columns_info = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        # 유니크 값 샘플링 (20개 이상이면 10개만 출력 + '...')
        if len(unique_vals) > 20:
            sample_display = list(unique_vals[:10]) + ['...']
        else:
            sample_display = unique_vals.tolist()

        columns_info.append({
            "데이터프레임명": df_name,
            "컬럼명": col,
            "데이터 타입": df[col].dtype,
            "count": df[col].count(),
            "mean": df[col].mean() if df[col].dtype in ["int64", "float64"] else None,
            "std": df[col].std() if df[col].dtype in ["int64", "float64"] else None,
            "min": df[col].min() if df[col].dtype in ["int64", "float64"] else None,
            "25%": df[col].quantile(0.25) if df[col].dtype in ["int64", "float64"] else None,
            "50%": df[col].median() if df[col].dtype in ["int64", "float64"] else None,
            "75%": df[col].quantile(0.75) if df[col].dtype in ["int64", "float64"] else None,
            "max": df[col].max() if df[col].dtype in ["int64", "float64"] else None,
            "결측 개수": df[col].isnull().sum(),
            "결측치 비율": round(df[col].isnull().sum() / len(df) * 100, 2),
            "고유값 개수": df[col].nunique(),
            "인스턴스(예제)": sample_display,
            "eda주제": "컬럼 개요"
        })
    columns_info_df = pd.DataFrame(columns_info)

    # ✅ 4. 범주형 변수 분포 (범주형 변수의 고유값 비율)
    categorical_info = []
    categorical_cols = [x for x in df.select_dtypes(include=["object", "category"]).columns if not any(keyword in x for keyword in ['년월', 'date', 'dt', 'ym'])]
    for col in categorical_cols:
        value_counts = df[col].value_counts(normalize=True) * 100
        for val, pct in value_counts.items():
            categorical_info.append({
                "데이터프레임명": df_name,
                "컬럼명": col,
                "값": val,
                "비율(%)": round(pct, 2),
                "eda주제": "범주형 변수 분포"
            })
    categorical_info_df = pd.DataFrame(categorical_info)

    # ✅ 5. 상관계수 분석 (상위 삼각 행렬 & 0.2 이상 필터링)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    reduced_corr = corr_matrix.where(mask).stack().reset_index()
    reduced_corr.columns = ["컬럼명", "상대 컬럼명", "상관계수"]
    reduced_corr = reduced_corr[reduced_corr["상관계수"] >= 0.2]
    reduced_corr["데이터프레임명"] = df_name
    reduced_corr["eda주제"] = "상관계수 분석"

    return table_summary, columns_info_df, categorical_info_df, reduced_corr


# 다중 데이터프레임 처리 및 결과 저장
def analyze_multiple_dataframes(dataframe_list):
    result_tmp = {
        "테이블 개요": [],
        "컬럼 개요": [],
        "범주형 변수 분포": [],
        "상관계수 분석": []
    }

    for df_name in dataframe_list:
        df = globals().get(df_name)

        if df is not None:
            rslt1, rslt2, rslt3, rslt4 = summarize_data(df, df_name)

            # 결과 합치기
            result_tmp["테이블 개요"].append(rslt1)
            result_tmp["컬럼 개요"].append(rslt2)
            result_tmp["범주형 변수 분포"].append(rslt3)
            result_tmp["상관계수 분석"].append(rslt4)

    # 결과 데이터프레임 병합
    for key in result_tmp.keys():
        result_tmp[key] = pd.concat(result_tmp[key], ignore_index=True)

    # 결과 저장 (엑셀 형식)
    output_path = "output/eda_summary.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in result_tmp.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[LOG] EDA 결과 저장 완료: {output_path}")

    return result_tmp
