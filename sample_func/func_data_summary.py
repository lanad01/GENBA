import pandas as pd
import numpy as np

def summarize_data(df, df_name):
    """
    주어진 데이터프레임(df)에 대한 EDA(탐색적 데이터 분석) 결과를 정리하여 반환하는 함수.
    - "컬럼 개요"에 범주형 변수 분포 정보 포함
    - 연속형 변수의 인스턴스(예제) 제외 (토큰 수 절감 목적)
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

    # ✅ 2. 컬럼 개요 (기본 통계 및 결측 정보 + 인스턴스 예제 + 범주형 분포 추가)
    columns_info = []
    for col in df.columns:
        col_dtype = df[col].dtype

        col_info = {
            "데이터프레임명": df_name,
            "컬럼명": col,
            "데이터 타입": col_dtype,
            "count": df[col].count(),
            "mean": round(df[col].mean(), 2) if col_dtype in ["int64", "float64"] else None,
            "std": round(df[col].std(), 2) if col_dtype in ["int64", "float64"] else None,
            "min": round(df[col].min(), 2) if col_dtype in ["int64", "float64"] else None,
            "25%": round(df[col].quantile(0.25), 2) if col_dtype in ["int64", "float64"] else None,
            "75%": round(df[col].quantile(0.75), 2) if col_dtype in ["int64", "float64"] else None,
            "max": round(df[col].max(), 2) if col_dtype in ["int64", "float64"] else None,
            "결측 개수": df[col].isnull().sum(),
            "결측치 비율": round(df[col].isnull().sum() / len(df) * 100, 0),
            "고유값 개수": df[col].nunique(),
        }

        # ✅ 범주형 변수만 인스턴스(예제) 포함 (연속형 변수 제외하여 토큰 수 절감)
        if col_dtype in ["object", "category"]:
            unique_vals = df[col].dropna().unique()
            # 유니크 값 샘플링 (20개 이상이면 10개만 출력 + '...')
            if len(unique_vals) > 20:
                sample_display = list(unique_vals[:10]) + ['...']
            else:
                sample_display = unique_vals.tolist()
            col_info["인스턴스(예제)"] = sample_display

            # ✅ 범주형 변수일 경우, 분포 정보 추가 (최대 10개만 저장하여 토큰 절감)
            value_counts = df[col].value_counts(normalize=True) * 100
            value_counts = value_counts[:10]  # 최대 10개만 유지
            category_distribution = {val: round(pct, 2) for val, pct in value_counts.items()}
            col_info["범주형 분포"] = category_distribution

        columns_info.append(col_info)

    columns_info_df = pd.DataFrame(columns_info)

    return columns_info_df


# 다중 데이터프레임 처리 및 결과 저장
def analyze_multiple_dataframes(dataframe_list):
    result_tmp = {"데이터 개요": []}

    for df_name in dataframe_list:
        df = globals().get(df_name)

        if df is not None:
            rslt = summarize_data(df, df_name)
            result_tmp["데이터 개요"].append(rslt)

    # 결과 데이터프레임 병합
    for key in result_tmp.keys():
        result_tmp[key] = pd.concat(result_tmp[key], ignore_index=True)

    # 결과 저장 (엑셀 형식)
    output_path = "../output/stage1/eda_summary.xlsx"
    try :
        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, df in result_tmp.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"✅ [LOG] EDA 결과가 성공적으로 저장되었습니다: {output_path}")
    except : 
            print(f"❌ [LOG] 파일 저장 실패: {output_path}")

    return result_tmp