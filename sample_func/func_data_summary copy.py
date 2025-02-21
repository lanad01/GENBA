import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime

def summarize_data(df, df_name):
    summary = {}

    # 1. 기본 정보
    summary['전체 데이터 크기 (행, 열)'] = df.shape
    summary['전체 컬럼 수'] = df.shape[1]
    summary['전체 인스턴스 수 (레코드 수)'] = df.shape[0]

    # 2. 결측치 정보 (컬럼 단위)
    eda_category = '결측치 분석'
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({
        'eda주제': eda_category,
        '결측치 개수': missing_values.values,
        '결측치 비율(%)': missing_percentage.round(2)
    }).reset_index(names = '컬럼명')
    missing_info = missing_info[missing_info['결측치 개수'] > 0].sort_values(by='결측치 개수', ascending=False)
    summary['결측 존재 컬럼 수'] = missing_info.shape[0]

    # 3. 변수 유형 분류
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary['범주형 변수 수'] = len(categorical_cols)
    summary['연속형 변수 수'] = len(numerical_cols)

    # 4. 연속형 변수 기초 통계량
    eda_category = '연속형 변수 분석 기초 통계량 분석'
    basic_stats = df[numerical_cols].describe().transpose().reset_index(names = '컬럼명')
    basic_stats['missing'] = df[numerical_cols].isnull().sum()
    basic_stats.insert(0, "eda주제", eda_category)
    basic_stats.insert(0, "데이터프레임명", df_name)

    # 5. 이상치 탐지 (IQR 방식)
    outliers = {}
    for col in numerical_cols:
        if any(keyword in col for keyword in ['여부', '코드', '등급']):
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outlier_count > 0:
            outliers[col] = int(outlier_count)

    summary['이상치가 있는 컬럼 수'] = len(outliers)
    summary['이상치 총 개수'] = sum(outliers.values())

    # 6. 컬럼 인스턴스 정보 (제약사항 반영)
    eda_category = '컬럼 인스턴스 정보'
    instance_info = []
    for col in df.columns:
        col_type = df[col].dtype
        unique_vals = df[col].nunique()

        if col in numerical_cols and not any(keyword in col for keyword in ['여부', '코드', '등급']):
            continue

        unique_samples = df[col].dropna().unique()
        if len(unique_samples) > 20:
            sample_display = list(unique_samples[:10]) + ['...']
        else:
            sample_display = unique_samples.tolist()

        instance_info.append({
            'eda주제': eda_category, 
            '데이터프레임명': df_name,
            '컬럼명': col,
            '데이터 타입': col_type,
            '고유값 개수': unique_vals,
            '예제 인스턴스': sample_display
        })
    instance_info_df = pd.DataFrame(instance_info)

    # 7. 상관계수 분석 (연속형 변수)
    eda_category = '상관관계 분석'
    if numerical_cols:
        correlation_matrix = df[numerical_cols].corr().round(2)

        # 0.2 미만의 상관계수 제거
        correlation_matrix = correlation_matrix.where(np.abs(correlation_matrix) >= 0.2, np.nan)

        # 삼각 행렬 변환 (위쪽 삼각형만 유지)
        triu_idx = np.triu_indices_from(correlation_matrix, k=1)
        correlation_matrix.iloc[triu_idx] = np.nan

        # NaN이 아닌 값만 남긴 DataFrame 생성
        correlation_matrix = correlation_matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)

        correlation_matrix.insert(0, "eda주제", eda_category)
        correlation_matrix.insert(1, "데이터프레임명", df_name)
    else:
        correlation_matrix = "연속형 변수가 없어 상관계수를 계산할 수 없습니다."

    # 8. 범주형 변수 주요 값 비율
    eda_category = '범주형 변수 분석'
    category_distribution = []
    for col in categorical_cols:
        top_categories = df[col].value_counts(normalize=True).head(5).to_dict()
        for cat, ratio in top_categories.items():
            category_distribution.append({'eda주제': eda_category, '데이터프레임명': df_name, '컬럼명': col, '값': cat, '비율(%)': round(ratio * 100, 2)})

    category_distribution_df = pd.DataFrame(category_distribution)

    # 9. 데이터 요약 정보 DataFrame 생성
    eda_category = '요약정보'
    summary_df = pd.DataFrame({
        'eda주제': eda_category,
        '항목': list(summary.keys()),
        '내용': list(summary.values())
    })
    summary_df.insert(0, "데이터프레임명", df_name)

    # 결과 반환
    return summary_df, instance_info_df, correlation_matrix, missing_info, basic_stats, category_distribution_df

# 다중 데이터프레임 처리
def analyze_multiple_dataframes(dataframe_list):
    result_tmp = {}

    for df_name in dataframe_list:
        df = globals().get(df_name)

        if df is not None:
            rslt1, rslt2, rslt3, rslt4, rslt5, rslt6 = summarize_data(df, df_name)

            result_tmp[f"{df_name}_요약정보"] = rslt1
            result_tmp[f"{df_name}_컬럼인스턴스"] = rslt2
            result_tmp[f"{df_name}_상관계수"] = rslt3
            result_tmp[f"{df_name}_결측치정보"] = rslt4
            result_tmp[f"{df_name}_기초통계량"] = rslt5
            result_tmp[f"{df_name}_범주형분포"] = rslt6
            
    return result_tmp
