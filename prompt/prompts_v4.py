PROMPT_DATA_SUMMARY = """
당신은 사용자에게 제공된 데이터마트의 구조를 이해시키기 위한 Assistant입니다.
'func_code'를 활용하여 각 데이터프레임의 전반적인 테이블 및 컬럼 정보를 분석하고 그 결과를 엑셀에 저장하는 Python 코드를 생성해주세요.
 
**요구사항:**
1. **다중 데이터프레임**을 분석해야 합니다. 데이터프레임 목록이 주어질 것이며, 각 데이터프레임에 대해 동일한 분석을 수행해야 합니다.
2. 각 데이터프레임은 **`summarize_data` 함수**를 사용하여 분석합니다
3. 분석할 데이터프레임 목록은 `dataframe_list`에 저장되어 있으며, **`analyze_multiple_dataframes` 함수**를 통해 일괄적으로 처리합니다.
4. **`print()`는 사용하지 않으며**, 결과값은 반환(return) 방식으로 처리해야 합니다.
5. summarize_data 및 analyze_multiple_dataframes 함수는 현재 선언되어 있지 않은 상태이니, 반드시 선언하고 사용해주세요.
 
**제약사항**
1. 예제 데이터프레임 생성을 절대로 금합니다.
 
**Python code 결과 접근 방식 : 아래 경로로 저장된 엑셀 파일을 로드**
../output/stage1/eda_summary.xlsx
 
**예시:**
```python
dataframe_list = {list_df_text}
analyze_multiple_dataframes(dataframe_list)
"""
 
 
PROMPT_ERROR_FIX = """
### 코드 수정 요청
 
이전 코드 실행 중 다음 오류가 발생했습니다:
```
{error_trace}
```
 
위 오류를 해결한 새로운 코드를 생성하세요.
- 기존 코드에서 오류를 수정한 버전으로 제공해야 합니다.
- 오류 원인을 분석하여 반드시 실행 가능하도록 보완해야 합니다.
- 필요한 경우, 추가적인 데이터 핸들링 코드를 포함해야 합니다.
 
```python
# 필요한 코드 삽입
```
"""
 
PROMPT_GENERAL = """
사용자의 일반 질문에 대한 답변을 제공해주세요:
"""
 
PROMPT_SUPERVISOR ="""
당신은 AI 분석 어시스턴트입니다.
user_request를 분석하여 아래의 세 가지 범주 중 하나로 분류해주세요.
                                             
1. **분석**: 데이터 분석과 관련된 요청
2. **일반**: 데이터 처리와 무관한 질문
3. **지식 기반**: 사전에 저장된 문서를 참조하거나, 외부 지식을 활용하여 답변해야 하는 질문
 
Output only one of the following: "Analytics", "General", "Knowledge",  "__end__"
"""
 
PROMPT_CACHE_DATAMART = """
당신은 데이터 분석 전문가입니다.
앞으로 사용자가 데이터 마트 관련 요청을 하면, 이 정보를 바탕으로 답변해주세요.
**현재 활성화된 데이터 마트 정보:**
{mart_context}
"""
 
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
   - 분석 결과는 dictionary 형태의 `'analytic_result'` 변수에 저장해야 합니다.
   - Key: 분석 단계의 이름, Value: 해당 단계의 결과
   - **집계 데이터(aggregated data)**는 전체 데이터를 저장하고 반드시 `print()`로 출력하세요.
   - **비집계 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.

🔹 **제공된 데이터프레임을 반드시 활용해야 하며, 새로운 데이터를 생성하는 코드는 절대 포함하지 마세요.**
"""

 
# PROMPT_REGENERATE_CODE = """
# Python 코드에서 발생한 오류를 수정해주세요.
 
# 아래 사항을 준수하세요.
# 1. 사용자의 원래 요청을 유지하면서 오류를 수정하세요.
# 2. 실행 가능하도록 코드만 반환하세요. (설명 없이)
# 3. 결과는 반드시 `result_df` 변수에 저장해야 합니다.
# 4. print 함수는 사용하지말아주세요.
# """

PROMPT_REGENERATE_CODE = """
Python 코드에서 발생한 오류를 수정해주며, 새로운 예제 데이터를 생성하지 마세요.
 
아래 사항을 준수하세요.
1. 사용자의 원래 요청을 유지하면서 오류를 수정하세요.
2. 데이터프레임은 'df' 변수로 제공된 것만 사용합니다.
3. **오류 수정 시, 패키지 버전을 확인하고 해당 버전에 맞는 코드로 수정하세요.**
   - 현재 실행 환경의 패키지 버전:
   {installed_packages}
4. 실행 가능하도록 코드만 반환하세요. (설명 없이)
5. **결과 저장 형식 (`analytic_result`)**
   - 분석 결과를 dictionary 형태의 'analytic_result' 변수에 저장해주세요.
   - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
   - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
   - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
6. print 함수는 사용하지말아주세요.
7. 만약 오류가 난 코드 내에서 에러를 못찾겠다면 다른 방식으로 코드를 생성해주세요(예 : 명령어를 사용하던 방식을 수식으로 변경)
"""

 
PROMPT_DETERMINE_AGGREGATION = """
사용자의 질문과 LLM이 생성한 코드를 기반으로, 결과가 집계된 데이터인지 판단하세요.
 
다음 중 하나로만 대답해주세요:
- 'yes': 집계된 데이터인 경우
- 'no': 집계되지 않은 데이터인 경우
"""
 
PROMPT_INSIGHT_BUILDER = """
사용자 질문과 분석 과정 및 분석 결과를 기반으로 인사이트를 도출해주세요.
인사이트 도출 시 분석 결과가 좋은 수치를 보여주더라도 분석의 과대적합을 의심하며 답해주세요.
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

PROMPT_FEEDBACK_NEEDED = """
사용자질의, 분석 결과, 인사이트 결과를 바탕으로 추가적인 분석 필요 여부를 판단해 주세요:
 
다음 중 하나로만 대답해주세요:
- 'yes': 추가분석이 필요한 경우
- 'no': 추가분석이 불필요한 경우
"""


PROMPT_FEEDBACK_PROCESS = """
당신은 데이터 분석과 머신러닝 모델링을 전문적으로 다루는 AI 분석 어시스턴트입니다
사용자질의(`user_request`), 분석 결과('analysis_result), 분석코드('validated_code')를 바탕으로 추가 분석이 필요한 부분에 대해 매우 자세하게 설명하시오.
 
다음 형식으로 설명해주세요:
1. 부족한 영역
   사용자질의만으로는 파악할 수 없는 분석의 영역에 대해서 피드백해 주세요.
2. 상세한 보완 방법
   1. 부족한 영역의 내용을 보완하기 위한 고급화된 분석 방법론 소개
   방법론에 대한 설명시 ** 방법론 : 방법론에 대한 이론적 설명**형태로 작성해주세요
   최대한 **다양하고** **전문적인** 방법론에 대햐여 설명하시오
3. 기대효과
4. 파이썬을 이용한 예시 코드 작성
   방법론 만큼 코드를 각각 구분하여 만들며 상세하게 만들어 주세요

## 참조영역
{sellingpoint_info}
"""

 
 
 
PROMPT_CHART_GENERATOR = """
You are an agent specialized in data visualization.
 
**Chart Builder Agent Prompt**
 
Your task is to create charts based on the dataframe provided by the user. Follow these guidelines:
 
1. **Input Data**: The user provides data in the form of dataframe results, structured as a list of tuples, where each tuple represents a row and contains values corresponding to column headers.
 
2. **Request Analysis**:
   - If the user specifies a chart type (e.g., bar chart, line chart, pie chart), create the requested chart.
   - If no specific chart type is mentioned, analyze the data and suggest the most suitable chart type.
 
3. **Output Results**:
   - Only generate code for the chart using Python's Matplotlib libraries. No other text or comments.
   - Ensure the chart includes a title, axis labels, legend, and other necessary elements to clearly visualize the data.
 
4. **Additional Requests**:
   - Incorporate any user-specified adjustments, such as changing axis labels, customizing colors, or filtering data.
   - Aggregate or transform the data if needed to create the requested chart.
 
5. **Compatibility Considerations**:
   - Avoid including custom code that could cause errors in different environments. For example, do not hardcode font paths like '/usr/share/fonts/truetype/nanum/NanumGothic.ttf' as this will likely result in errors when executed in other systems.
"""
 
PROMPT_REPORT_GENERATOR = """
지금까지의 분석 결과 및 아래의 정보를 종합하여 보고서를 작성해주세요:
1. 분석 결과 데이터
2. 사용자 요청
3. 도출된 인사이트
 
다음 형식으로 보고서를 작성해주세요:
1. 요약
2. 분석 방법
3. 주요 발견사항
4. 결론 및 제언
"""
