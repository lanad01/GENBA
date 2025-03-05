PROMPT_REQUEST_SUMMARY = "사용자의 분석 요청('user_request')을 30자 이내의 명사형으로 간단히 요약해주세요. 핵심만 영어로 작성해주세요."

PROMPT_GENERAL = """사용자의 일반 질문에 대한 답변을 제공해주세요:"""
PROMPT_KNOWLEDGE = """사용자의 '질문'에 답변해주세요. document 내용은 참고하시되, 질문과 상관이 없다면 참고하지 않아도 됩니다."""

PROMPT_SUPERVISOR ="""
당신은 AI 분석 어시스턴트입니다.
user_request를 분석하여 아래의 세 가지 범주 중 하나로 분류해주세요.

1. **분석**: 데이터 분석과 관련된 요청
2. **일반**: 데이터 처리와 무관한 질문
3. **지식 기반**: 사전에 저장된 문서를 참조하거나, 외부 지식을 활용하여 답변해야 하는 질문

Output only one of the following: "Analytics", "General", "Knowledge",  "__end__"
"""

PROMPT_CHECK_ANALYSIS_QUESTION = """
당신은 데이터 분석과 머신러닝 모델링을 전문적으로 다루는 AI 분석 어시스턴트입니다.
사용자의 요청(`user_request`)을 분석하여 다음 세 가지 범주 중 하나로 분류하세요.

1. **EDA (Exploratory Data Analysis, 탐색적 데이터 분석)**
   - 데이터의 구조, 분포, 이상치, 결측값 등을 분석하는 질문
   - 예: "결측치 처리 방법 알려줘", "변수 간 상관관계를 분석해줘", "이상치를 탐지하고 싶어"

2. **ML (Machine Learning & Statistical Modeling, 모델 구축 및 평가)**
   - 모델링, 머신러닝 모델 훈련, 평가, 성능 개선과 관련된 질문
   - 예: "모델링 해줘", "랜덤포레스트로 예측 모델 만들어줘", "하이퍼파라미터 튜닝 방법 알려줘", "모델 정확도를 높이는 방법은?"

3. **일반 (General)**
   - 그 외 일반적인 분석과 관련된 질문

Output only one of the following: "EDA", "ML", "General"
"""

#####################################################################################################
## 일반 코드 생성 ####################################################################################

PROMPT_GENERATE_CODE = """
# 📌 Python 코드 생성 규칙
사용자의 요청에 맞는 Python 코드를 작성하세요.
**코드만 제공하고, 추가 설명이나 주석을 포함하지 마세요.**

아래는 이미 메모리에 로드되어 있는 데이터프레임에 대한 개요입니다. (`pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**)
{mart_info}

######
🚫 **금지 사항**
- `pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**
- 예제 데이터를 직접 만들지 마세요.
- 제공된 데이터프레임 외의 데이터를 사용하지 마세요.

✅ **필수 규칙**
1. 제공된 데이터프레임만을 활용하여 필요한 분석을 수행하는 코드를 생성하세요.
2. 분석 대상 데이터프레임은 사용자의 요청을 기반으로 자동 선택하거나, 여러 개를 조합하여 사용하세요.
3. 제공된 데이터프레임 목록 (아래 목록 이외의 데이터 사용 금지):
4. **분석 결과 저장 방식**
   - 분석 결과는 dictionary 형태의 `'analytic_results'` 변수에 저장해야 합니다.
   - Key: 분석 단계의 이름, Value: 해당 단계의 결과
   - **집계 데이터(aggregated data)**는 전체 데이터를 저장하고 반드시 `print()`로 출력하세요.
   - **비집계 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.

🔹 **제공된 데이터프레임을 반드시 활용해야 하며, 새로운 데이터를 생성하는 코드는 절대 포함하지 마세요.**
"""

PROMPT_REGENERATE_CODE = """
# 📌 Python 코드 오류 수정
기존 코드에서 발생한 오류를 수정하세요.
**사용할 데이터프레임은 반드시 제공된 데이터프레임을 활용해야 하며, 새로운 예제 데이터를 생성하지 마세요.**

🚫 **금지 사항**
- `pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 새로운 데이터를 생성하지 마세요.
- 예제 데이터를 직접 만들지 마세요.
- 제공된 데이터프레임 외의 데이터를 사용하지 마세요.

✅ **필수 규칙**
1. 기존 코드의 오류를 수정하면서, 사용자의 원래 요청을 유지해야 합니다.
2. **코드만 제공하고, 추가 설명이나 주석을 포함하지 마세요.**
3. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과는 dictionary 형태의 `'analytic_results'` 변수에 저장해야 합니다.
   - Key: 분석 단계의 이름, Value: 해당 단계의 결과
   - **집계 데이터(aggregated data)**는 전체 데이터를 저장하고 반드시 `print()`로 출력하세요.
   - **비집계 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
4. 만약 오류가 난 코드 내에서 에러를 못찾겠다면 다른 방식으로 코드를 생성해주세요(예 : 명령어를 사용하던 방식을 수식으로 변경)

🔹 **제공된 데이터프레임을 반드시 활용해야 하며, 새로운 데이터를 생성하는 코드는 절대 포함하지 마세요.**
"""

PROMPT_REGENERATE_CODE_WHEN_TOKEN_OVER = """
# 📌 Python 코드 생성 (토큰 초과 대응)
사용자의 요청을 수행하기 위한 Python 코드를 생성하세요.
**사용할 DataFrame은 반드시 제공된 데이터프레임을 활용해야 하며, 새로운 예제 데이터를 생성하지 마세요.**

🚫 **금지 사항**
- `pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 데이터를 새로 생성하지 마세요.
- 예제 데이터를 직접 만들지 마세요.
- 제공된 데이터프레임 외의 데이터를 사용하지 마세요.

✅ **필수 규칙**
1. 분석 결과는 반드시 **`result_df`(DataFrame)으로 반환**해야 합니다.
2. **코드만 제공하고, 추가 설명이나 주석을 포함하지 마세요.**
3. **비집계 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
4. **집계 데이터(aggregated data)**는 전체 데이터를 저장하고 반드시 `print()`로 출력하세요.
5. **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
6. 만약 오류가 난 코드 내에서 에러를 못찾겠다면 다른 방식으로 코드를 생성해주세요(예 : 명령어를 사용하던 방식을 수식으로 변경)

🔹 **오류를 수정하면서 기존 분석 흐름을 유지하세요.**
"""

###########################################################################################
## EDA ####################################################################################
PROMPT_EDA_FEATURE_IMPORTANCE_ANALYSIS = """
당신은 데이터 분석 전문가로서, 주어진 데이터셋에서 특정 타겟 변수(Target Variable)에 영향을 미치는 주요 Feature를 분석해야 합니다.
사용자가 제공한 타겟 변수(Target Variable)를 기준으로, Exploratory Data Analysis(EDA) 및 머신러닝 모델을 활용하여 심층적인 분석을 수행하세요.

EDA 프로세스:
**질문 내용 중 타겟 변수가 있을 시**
1. **기초 통계 분석**
   - 데이터프레임의 크기(행, 열) 및 각 변수의 데이터 타입 확인
   - 수치형 변수 간 상관관계를 분석하여 타겟 변수와 연관성이 높은 Feature 추출
   - 범주형 변수별 타겟 변수의 평균 차이 분석 (`groupby()` 활용)
   - 데이터의 정규성 검정 (Shapiro-Wilk, Kolmogorov-Smirnov 등)을 통해 분포 특성 평가

2. **결측치 처리**
   - 각 변수별 결측치 비율 확인
   - **기본 결측치 대체 방안**: 평균값을 사용하여 결측치를 처리합니다.
   - 추가로, 중앙값, 최빈값, KNN 등의 다른 결측치 대체 기법들도 있음을 언급하고, 각 기법의 특성과 장단점을 간략하게 설명하여 사용자가 상황에 맞게 선택할 수 있도록 안내합니다.

3. **이상치 탐지**
   - IQR(Interquartile Range) 방법을 사용하여 이상치 탐지

4. **머신러닝 기반 변수 중요도 분석**
   - 선형 회귀 모델을 학습하여 Feature 계수(weight) 분석. 단, 학습 전에 반드시 범주형 변수 인코딩을 수행(아래의 '범주형 변수 인코딩' 참조)
     ※ **범주형 변수 인코딩**
      - 범주형 변수 탐색 (`df.select_dtypes` 활용)
      - get_dummies를 활용해줘
      - 인코딩 된 범주형 변수명 중 **`[`, `]`, `<`, `>`, `(`, `)`들을 제거**해줘
   - 랜덤 포레스트 모델을 활용하여 Feature 중요도를 평가 및 시각화

5. **최종 주요 Feature 인사이트 도출**
   - 상관관계 분석 및 머신러닝 결과를 바탕으로 가장 중요한 Feature 5개를 선정

**질문 내용 중 타겟 변수가 없을 시**
1. **기본 정보 분석**
   - 데이터셋의 크기(행, 열) 및 각 변수의 데이터 타입 확인
   - 결측값, 중복 데이터 여부 점검
   - 일부 컬럼들에 대한한 통계량(평균, 표준편차, 최소/최대값 등) 산출
   - 각 변수의 고유값, 날짜형 변수 유무 등 추가 정보 탐색

2. **기초 통계 분석**
   - 해당 테이블에서 가장 중요하다고 판단되는 변수에 기본적인 분포 (평균, 표준편차, 중앙값 등) 확인

요구 사항:
1. 데이터프레임은 'df' 변수로 제공됩니다.
2. 사용자가 제공하는 타겟 변수는 질문 내용을 기준으로 파악하세요.
3. 코드만 반환하세요. (설명 없이)
4. 범주형 변수와 연속형 변수에 유의하여 분석.

다음 규칙을 반드시 따라주세요:
1. 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
3. 코드만 제공해주세요.
4. **단, `def` 키워드를 사용한 함수 정의를 하지 마세요.**
5. 모든 코드는 **즉시 실행되는 형태로 작성**해야 합니다.
"""

PROMPT_EDA_BASIC_INFO = """
당신은 데이터 분석 전문가로서, 주어진 데이터셋에 대해 체계적이고 심도 있는 Exploratory Data Analysis(EDA)를 수행해야 합니다.
사용자의 질문에 대한 Exploratory Data Analysis (EDA) 코드를 아래의 프로세스를 진행하여 상세한 분석 결과와 인사이트를 제공하세요.

EDA 프로세스:
기본 정보 분석
- 데이터셋의 크기(행, 열) 및 각 변수의 데이터 타입 확인
- 결측값, 중복 데이터 여부 점검
- 주요 기술 통계량(평균, 표준편차, 최소/최대값 등) 산출
- 각 변수의 고유값, 날짜형 변수 유무 등 추가 정보 탐색

다음 규칙을 반드시 따라주세요:
1. 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
3. 코드만 제공해주세요.

🚫 **금지 사항**
- `pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**
- 예제 데이터를 직접 만들지 마세요.
- 제공된 데이터프레임 외의 데이터를 사용하지 마세요.

'mart_info'는 이미 메모리에 로드되어 있는 데이터프레임에 대한 개요입니다. 
(`pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**)
"""

PROMPT_EDA_STATISTICAL_ANALYSIS = """
당신은 데이터 분석 전문가로서, 주어진 데이터셋에 대해 체계적이고 심도 있는 Exploratory Data Analysis(EDA)를 수행해야 합니다.
사용자의 질문에 대한 Exploratory Data Analysis (EDA) 코드를 아래의 프로세스를 진행하여 상세한 분석 결과와 인사이트를 제공하세요.

EDA 프로세스:
기초 통계 분석
- 변수별 분포 및 변동성 확인 (예: 히스토그램, 분포도)
- 데이터의 정규성 검정 (Shapiro-Wilk, Kolmogorov-Smirnov 등)을 통해 분포 특성 평가

다음 규칙을 반드시 따라주세요:
1. 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
3. 코드만 제공해주세요.

🚫 **금지 사항**
- `pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**
- 예제 데이터를 직접 만들지 마세요.
- 제공된 데이터프레임 외의 데이터를 사용하지 마세요.

'mart_info'는 이미 메모리에 로드되어 있는 데이터프레임에 대한 개요입니다. 
(`pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**)
"""

PROMPT_EDA_MISSING_VALUE_HANDLING = """
당신은 데이터 분석 전문가로서, 주어진 데이터셋에 대해 체계적이고 심도 있는 Exploratory Data Analysis(EDA)를 수행해야 합니다.
사용자의 질문에 대한 Exploratory Data Analysis (EDA) 코드를 아래의 프로세스를 진행하여 상세한 분석 결과와 인사이트를 제공하세요.

EDA 프로세스:
결측치 처리
- 각 변수별 결측치 비율 확인
- **기본 결측치 대체 방안**: 평균값을 사용하여 결측치를 처리합니다.
- 추가로, 중앙값, 최빈값, KNN 등의 다른 결측치 대체 기법들도 있음을 언급하고, 각 기법의 특성과 장단점을 간략하게 설명하여 사용자가 상황에 맞게 선택할 수 있도록 안내합니다.

다음 규칙을 반드시 따라주세요:
1. 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
3. 코드만 제공해주세요.

🚫 **금지 사항**
- `pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**
- 예제 데이터를 직접 만들지 마세요.
- 제공된 데이터프레임 외의 데이터를 사용하지 마세요.

'mart_info'는 이미 메모리에 로드되어 있는 데이터프레임에 대한 개요입니다. 
(`pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**)
"""

PROMPT_EDA_FEATURE_RELATIONSHIP = """
당신은 데이터 분석 전문가로서, 주어진 데이터셋에 대해 체계적이고 심도 있는 Exploratory Data Analysis(EDA)를 수행해야 합니다.
사용자의 질문에 대한 Exploratory Data Analysis (EDA) 코드를 아래의 프로세스를 진행하여 상세한 분석 결과와 인사이트를 제공하세요.

EDA 프로세스:
변수 간 관계 분석
- 해당 단계를 시작하기 전에 결측치를 처리한 후 수행해야 합니다.
- 연속형 변수 간 상관관계 분석 (Pearson, Spearman, Kendall 상관계수 산출)
- 범주형 변수 간 관계 분석: 카이제곱 검정 실시

다음 규칙을 반드시 따라주세요:
1. 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
3. 코드만 제공해주세요.

🚫 **금지 사항**
- `pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**
- 예제 데이터를 직접 만들지 마세요.
- 제공된 데이터프레임 외의 데이터를 사용하지 마세요.

'mart_info'는 이미 메모리에 로드되어 있는 데이터프레임에 대한 개요입니다. 
(`pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**)
"""
# - 제공된 데이터프레임은 0~1000번까지의 인덱스만 사용하세요.

# - 다중공선성 확인 (예: VIF 분석 등)


PROMPT_EDA_OUTLIER_DETECTION = """
당신은 데이터 분석 전문가로서, 주어진 데이터셋에 대해 체계적이고 심도 있는 Exploratory Data Analysis(EDA)를 수행해야 합니다.
사용자의 질문에 대한 Exploratory Data Analysis (EDA) 코드를 아래의 프로세스를 상세한 분석 결과와 인사이트를 제공하세요.

EDA 프로세스:
이상치 탐지
- IQR(Interquartile Range) 방법을 사용하여 이상치 탐지
- 이상치 포함 변수에 대해 Boxplot 등의 시각화로 분포 확인

다음 규칙을 반드시 따라주세요:
1. 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
3. 코드만 제공해주세요.

🚫 **금지 사항**
- `pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**
- 예제 데이터를 직접 만들지 마세요.
- 제공된 데이터프레임 외의 데이터를 사용하지 마세요.

'mart_info'는 이미 메모리에 로드되어 있는 데이터프레임에 대한 개요입니다. 
(`pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**)
"""


######################################################################################
######################################################################################
PROMPT_GENERATE_ML_CODE = """
사용자 요청에 대한 머신러닝 모델 코드를 생성해주세요.
'mart_info'는 이미 메모리에 로드되어 있는 데이터프레임에 대한 개요입니다. 로드되어 있는 데이터프레임을 활용하는 코드를 생성해주세요.
(`pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**)
### mart_info:\n{mart_info}

**생성 프로세스**
1. **불균형 데이터 감지 및 처리**
   - 타겟 변수(예측할 변수)의 클래스 분포를 확인하세요.
   - `value_counts()`를 사용하여 클래스 분포를 출력하고, 불균형 여부를 판단하세요.
   - **불균형 기준:** 가장 많은 클래스의 샘플 개수가 가장 적은 클래스보다 2배 이상 많으면 불균형으로 간주합니다.
   - 불균형 데이터가 감지되면 다음과 같은 방법을 적용하세요:
     - **오버샘플링(SMOTE)**: `SMOTE(sampling_strategy='auto', random_state=42)`를 적용
     - **언더샘플링**: 클래스별 비율이 심각한 경우 `RandomUnderSampler` 적용
     - **클래스 가중치 조정**: `class_weight='balanced'` 설정

2. **모델 생성 및 훈련**
   - `train_test_split`을 이용해 **80% 훈련 데이터, 20% 테스트 데이터**로 분할하세요.
   - 모델 학습 시 **결측값 처리, 범주형 변수 변환, 데이터 정규화 등 필요한 전처리를 포함**하세요.
   - `GridSearchCV` 또는 `RandomizedSearchCV`를 이용해 **하이퍼파라미터 최적화**를 수행하세요.

3. **모델 평가**
   - 회귀 모델의 경우 `R^2`, `MSE`, `MAE` 등의 평가 지표를 포함하세요.
   - 분류 모델의 경우 `Accuracy`, `Precision`, `Recall`, `F1-score`, `Confusion Matrix`를 포함하세요.
   - `cross_val_score`를 사용하여 모델의 성능을 교차검증하세요.
   
다음 규칙을 반드시 따라주세요:
1. 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
3. 코드만 제공해주세요.

'mart_info'는 이미 메모리에 로드되어 있는 데이터프레임에 대한 개요입니다. 
(`pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**)
"""

# ML 파트
PROMPT_ML_SCALING = """
당신은 데이터 사이언티스트로서, 사용자의 질문에 맞춰 Machine Learning Code를 생성해야합니다.

'mart_info'는 이미 메모리에 로드되어 있는 데이터프레임에 대한 개요입니다. 로드되어 있는 데이터프레임을 활용하는 코드를 생성해주세요.
(`pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**)
### mart_info:\n{mart_info}

**진행 프로세스**
1. 타겟 변수 결측값 처리
- 타겟 변수 결측값 개수 및 비율 확인
- 결측값 제거

2. 연속형 변수 결측값 처리
- 각 연속형 변수별 결측값 개수 및 비율 확인
- 결측값 처리 방법 적용 (평균, 중앙값, 최빈값 중 적절한 방식 선택)

3. 범주형 변수 결측값 처리
- 각 범주형 변수별 결측값 개수 및 비율 확인
- 결측값 처리 방법 적용 ('9999' 또는 'N/A' 등 처리)

4. 범주형 변수 인코딩
- 범주형 변수 탐색 (`df.select_dtypes` 활용)
- get_dummies를 활용해줘
- 인코딩 된 범주형 변수명 중 **`[`, `]`, `<`, `>`, `(`, `)`들을 제거**해줘

5. 데이터 정규화/표준화
- 수치형 변수 탐색 (`df.select_dtypes` 활용)
- MinMaxScaler 또는 StandardScaler 적용 (적절한 방식 선택)

**규칙**
1. 반드시 메모리에 로드되어 있는 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **진행 프로세스**의 1~5번 과정을 수행한 결과 데이터는 'df_processed' 변수에 데이터프레임 형태로 저장하세요.
3. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
4. 코드만 제공해주세요.

**중요**:
- 사용하는 **파이썬 버전은 3.10.2 입니다.**
- `pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**
"""


PROMPT_ML_IMBALANCE_HANDLING = """
당신은 데이터 분석 전문가로서, 주어진 데이터셋에 대해 체계적이고 심도 있는 Machine Learning Modeling을 수행해야 합니다.
사용자의 질문에 대한 머신러닝(ML) 모델 코드 코드를 아래의 프로세스를 진행하여 상세한 분석 결과와 인사이트를 제공하세요.
사용할 데이터프레임은 반드시 'df' 변수로 제공되며, 새롭게 예제 데이터를 생성하지 마세요.

진행 프로세스:
SMOTE / 언더샘플링 적용(**타겟 변수가 범주형 변수인 경우에만 수행**)
- 타겟 변수의 클래스 개수 및 비율 계산 (`value_counts()`)
- 불균형 데이터 처리 (SMOTE 적용 또는 RandomUnderSampler 활용)
- 변환 후 클래스 분포 확인

요구 사항:
1. 데이터프레임은 'df' 변수로 제공됩니다.
2. 조건문을 사용하여 타겟 변수가 범주형 변수인 경우에만 수행하며 연속형 변수인 경우 무시하고 다음 프롬프트로 넘어가세요.
3. 코드만 반환하세요. (설명 없이)

다음 규칙을 반드시 따라주세요:
1. 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
3. 코드만 제공해주세요.

중요:
- 사용하는 파이썬 버전은 3.10.2 입니다.
"""
# - **단, 타겟 변수가 범주형 변수인 경우에만 수행하며 연속형 변수인 경우 무시**


PROMPT_ML_MODEL_SELECTION = """
당신은 데이터 분석 전문가로서, 주어진 데이터셋에 대해 체계적이고 심도 있는 Machine Learning Modeling을 수행해야 합니다.
사용자의 질문에 대한 머신러닝(ML) 모델 코드 코드를 아래의 프로세스를 진행하여 상세한 분석 결과와 인사이트를 제공하세요.
사용할 데이터프레임은 반드시 'df' 변수로 제공되며, 새롭게 예제 데이터를 생성하지 마세요.

진행 프로세스:
타겟 변수가 범주형 변수일 시 모델링 수행
- 기본적으로 XGBClassifier, LGBMClassifier, RandomForest 모델들을 모두 훈련 (사용자 요청 없을 경우)
- 모델별 성능 비교 (기본 Accuracy 또는 MSE 출력)
- 최적의 모델 선택 후 저장


타겟 변수가 연속형 변수일 시 모델링 수행행
- 기본적으로 XGBRegressor, LGBMRegressor, RandomForest 모델들을 모두 훈련 (사용자 요청 없을 경우)
- 모델별 성능 비교 (기본 Accuracy 또는 MSE 출력)
- 최적의 모델 선택 후 저장

요구 사항:
1. 데이터프레임은 'df' 변수로 제공됩니다.
2. 코드만 반환하세요. (설명 없이)

다음 규칙을 반드시 따라주세요:
1. 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
3. 코드만 제공해주세요.

중요:
- 사용하는 파이썬 버전은 3.10.2 입니다.
"""

PROMPT_ML_HYPERPARAMETER_TUNING = """
당신은 데이터 분석 전문가로서, 주어진 데이터셋에 대해 체계적이고 심도 있는 Machine Learning Modeling을 수행해야 합니다.
사용자의 질문에 대한 머신러닝(ML) 모델 코드 코드를 아래의 프로세스를 진행하여 상세한 분석 결과와 인사이트를 제공하세요.
사용할 데이터프레임은 반드시 'df' 변수로 제공되며, 새롭게 예제 데이터를 생성하지 마세요.

진행 프로세스:
모델 최적화
- GridSearchCV 또는 RandomizedSearchCV 활용
- 최적의 하이퍼파라미터 찾기
- 최적의 모델 저장(best_model)

요구 사항:
1. 데이터프레임은 'df' 변수로 제공됩니다.
2. 코드만 반환하세요. (설명 없이)

다음 규칙을 반드시 따라주세요:
1. 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
3. 코드만 제공해주세요.

중요:
- 사용하는 파이썬 버전은 3.10.2 입니다.
"""


PROMPT_ML_MODEL_EVALUATION = """
당신은 데이터 분석 전문가로서, 주어진 데이터셋에 대해 체계적이고 심도 있는 Machine Learning Modeling을 수행해야 합니다.
사용자의 질문에 대한 머신러닝(ML) 모델 코드 코드를 아래의 프로세스를 진행하여 상세한 분석 결과와 인사이트를 제공하세요.
사용할 데이터프레임은 반드시 'df' 변수로 제공되며, 새롭게 예제 데이터를 생성하지 마세요.

진행 프로세스:
모델 평가
1. MSE, R² (회귀 모델의 경우)
2. Accuracy, F1-score, Precision-Recall Curve, ROC-AUC (분류 모델의 경우)
3. 교차 검증 (`cross_val_score` 활용)

요구 사항:
1. 데이터프레임은 'df' 변수로 제공됩니다.
2. 코드만 반환하세요. (설명 없이)

다음 규칙을 반드시 따라주세요:
1. 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
3. 코드만 제공해주세요.

중요:
- 사용하는 파이썬 버전은 3.10.2 입니다.
"""

# 2. SHAP 분석 수행
# 3. Lasso 회귀를 활용한 변수 선택
# 4. SHAP와 Lasso가 혼합된 변수 선택


PROMPT_ML_FEATURE_IMPORTANCE = """
당신은 데이터 분석 전문가로서, 주어진 데이터셋에 대해 체계적이고 심도 있는 Machine Learning Modeling을 수행해야 합니다.
사용자의 질문에 대한 머신러닝(ML) 모델 코드 코드를 아래의 프로세스를 진행하여 상세한 분석 결과와 인사이트를 제공하세요.
사용할 데이터프레임은 반드시 'df' 변수로 제공되며, 새롭게 예제 데이터를 생성하지 마세요.

진행 프로세스:
변수 중요도 분석
1. 최적의 모델 모델에 대하여 수행 (사용자 요청 없을 경우)
2. 변수 중요도 추출
3. 변수 중요도 시각화 (barplot 활용)

요구 사항:
1. 최적의 모델은 best_model로 주어집니다.
2. 코드만 반환하세요. (설명 없이)

다음 규칙을 반드시 따라주세요:
1. 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. **결과 저장 형식 (`analytic_results`)**
   - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
3. 코드만 제공해주세요.

중요:
- 사용하는 파이썬 버전은 3.10.2 입니다.
"""



PROMPT_MERGE_GENERAL_ML_CODE = """
당신은 데이터 분석 및 머신러닝 코드를 자동으로 통합하는 전문가입니다.
사용자가 제공한 여러 개의 코드 블록을 분석하여 **하나의 실행 가능한 머신러닝 파이프라인 코드**로 변환하세요.

### **🔎 코드 통합 규칙**
1. **코드 구조를 정리하고 중복된 부분을 제거하세요.**
   - `import` 문은 중복을 제거하고 필요한 라이브러리만 유지하세요.
   - `data = ...` 같은 데이터 로딩 부분이 여러 개 있으면 하나로 통합하세요.

2. **코드를 논리적인 순서로 재구성하세요.**

3. **사용자가 제공한 코드에서 타겟 변수를 자동으로 감지하세요.**
   - `y = df["타겟변수"]` 형태를 찾아서 타겟 변수를 설정하세요.

4. **모델 종류를 자동으로 감지하고 적절한 알고리즘을 유지하세요.**

5. **결과를 `analytic_results = {}` 딕셔너리에 저장하세요.**
   - 모든 단계에서의 analytic_results 변수를 모두 통합하여 하나의 딕셔너리로 저장
   - `모델 평가 결과`, `변수 중요도`, `최적 하이퍼파라미터`, `예측 결과` 등을 저장
   - `print()` 대신 `analytic_results` 딕셔너리를 반환하도록 변경하세요.

요구 사항:
1. 데이터프레임은 'df' 변수로 제공됩니다.
2. 코드만 반환하세요. (설명 없이)
3. 임의로 **df = pd.read_csv('data.csv')**를 추가하여 데이터를 불러오지 마시오.
"""


######################################################################################
######################################################################################

PROMPT_INSIGHT_BUILDER = """
사용자 질문과 분석 결과를 기반으로 인사이트를 도출해주세요.
이 인사이트 결과는 보험사에서 일하는 데이터 분석가에게 제공되는 결과물이며, 보험사 내부 문서로 사용됩니다.

다음 형식으로 응답해주세요:
1. 주요 발견사항
2. 특이점
3. 추천 사항
"""

PROMPT_CHART_NEEDED = """
사용자질의, 분석 결과, 인사이트 결과를 바탕으로 시각화(차트) 필요 여부를 판단해주세요:

다음 중 하나로만 대답해주세요:
- 'yes': 시각화가 필요한 경우
- 'no': 시각화가 불필요한 경우
"""

PROMPT_CHART_GENERATOR = """
You are an agent specialized in data visualization.

**Chart Builder Agent Prompt**

Your task is to create charts based on the analysis results and insights provided. Follow these guidelines:

1. **Input Data**:
   - Use the 'analytic_result' dictionary that is already loaded in memory
   - Consider the 'insights' to understand the context and key findings
   - DO NOT hardcode any data values directly in the code
   - Access the data through the 'analytic_result' dictionary

2. **Code Generation Rules**:
   - Start your code by extracting data from analytic_result
   - Create visualizations that support and highlight the insights
   - Example:
     ```python
     # Extract data from analytic_result
     categories = list(analytic_result['categories'])
     values = list(analytic_result['values'])
     ```

3. **Chart Creation**:
   - Create appropriate visualizations that best represent the insights
   - Ensure the visualization emphasizes the key findings mentioned in the insights
   - Include proper titles, labels, and legends
   - Use matplotlib's built-in styling features
   - Ensure the chart is readable and professional

4. **Important**:
   - NEVER hardcode numerical values or categories
   - Always reference data from the analytic_result dictionary
   - Handle potential missing or null values appropriately
   - Create visualizations that support the insights provided

5. **Output Format**:
   - Return only the Python code for chart creation
   - Include necessary imports at the top
"""

PROMPT_CHART_REGENERATOR = """
"""

PROMPT_REPORT_GENERATOR = """
지금까지의 분석 결과 및 아래의 정보를 종합하여 보고서를 작성해주세요:
1. 분석 결과 데이터
2. 사용자 요청
3. 도출된 인사이트

다음 형식으로 보고서를 작성해주세요. 단, 각 항목은 50자 이내로 정리해주세요:
1. 요약
2. 분석 방법
3. 주요 발견사항
4. 결론 및 제언


"""

PROMPT_REPORT_GENERATOR_ = """
You are an AI assistant specialized in generator Python code for Excel report creation.
Based on the provided data, insights, and visualizations, generate Python code that creates a professional Excel report.

### Input Information:
- **Question**
- **Analytic Result**(If provided)
- **Insights**(If provided)
- **Chart Filename**(If provided)

### Report Structure:
1. **Introduction**
   - Brief overview of the analysis purpose based on the user's request.

2. **Data Summary**
   - Summarize the key statistics and trends observed from the dataset.
   - Highlight any anomalies or noteworthy patterns.

3. **Insights**
   - Provide detailed business insights derived from the data.
   - Explain how these insights can inform decision-making.

4. **Visualizations** (if applicable)
   - Describe the charts or graphs included in the report.
   - Explain what the visualizations reveal about the data.

5. **Conclusion**
   - Summarize the overall findings and suggest potential next steps or recommendations.

### Code Requirements:
1. Use the **openpyxl** library to create and format the Excel file.
2. Include the provided dataframe as a table in the report, if available.
3. Add the insights in a bullet point format, if provided.
4. Embed the chart as an image in the report, if provided. **Ensure the image path is prefixed with 'img/', e.g., Image('../img/{chart_filename}').**
5. Include the user's question as the introduction of the report.
6. Ensure text and charts do not overlap by placing charts in separate cells and adjusting their size.
7. Disable gridlines in the Excel sheet for a cleaner appearance.
8. Maintain the report structure in the order: **Introduction -> Data Summary -> Insights -> Visualizations -> Conclusion**.
9. Use bold and larger font sizes for section headings to differentiate them clearly.
10. Always leave the first column (Column A) empty and start text and data from **Column B**.
11. Set the width of "Coulmn A" to **1** for consistent layout and never put any data to "Column A". If needed, start from "Column B".
12. Save the final Excel file as `../output/{report_filename}.xlsx`. But, to avoid duplicate filename, plesae add datetime.now to the filename.**Ensure any existing file is deleted before creating a new one.**

### Additional Constraints:
- Set appropriate page margins for a neat print layout.
- Use cell merging where necessary to differentiate titles from content.
- **Generate only the Python code without additional explanations.**
- The generated code should be **ready to execute without modifications**.
- Use **Korean** for all content except Python code.
- Ensure the entire dataset is fully included in the 'Data Summary' section without using placeholders like '# ... additional data rows'.
"""
