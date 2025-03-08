
PROMPT_REQUEST_SUMMARY = "사용자의 분석 요청('user_request')을 20자 이내의 명사형으로 간단히 요약해주세요. 핵심만 영어로 작성해주세요."

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

PROMPT_CONTEXT_FILTER = """
너는 AI 분석 비서야.  
현재 질문을 최우선으로 두고, 과거 대화에서 **필요한 정보만 유지**해서 정리해줘.
- 🔹 'validated_code' (이전 실행 코드)와 'analytic_result' (이전 분석 결과)는 꼭 포함해야 해.
- 🔹 기존 대화에서 현재 질문과 관련 없는 내용은 제거하고, 핵심 정보만 요약해서 포함해줘.

예시 : 
# 📌 주요 참고 정보
- 이전 질문: 보험 상품별 해지율에 영향을 미치는 주요 피쳐 분석
- 분석된 주요 결과: 고액항암치료비, 골절진단 상품의 해지율이 가장 높음 (0.42)
- 기존 상관관계 분석 결과: 모집설계사수, 대출가능금액합계가 해지율과 가장 높은 상관관계 (0.06)

# 📜 참고 코드 (기존 실행 코드)
```python
(이전 validated_code)
"""
#####################################################################################################
## 일반 코드 생성 ####################################################################################

PROMPT_GENERATE_CODE = """
# 📌 Python 코드 생성 규칙
사용자의 요청에 맞는 Python 코드를 작성하세요.
**코드만 제공하고, 추가 설명이나 주석을 포함하지 마세요.**

아래는 이미 메모리에 로드되어 있는 데이터프레임에 대한 개요입니다. (`pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**)
{mart_info}


🚫 **금지 사항**
- `pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**
- 예제 데이터를 직접 만들지 마세요.
- 제공된 데이터프레임 외의 데이터를 사용하지 마세요.

✅ **필수 규칙**
1. 제공된 데이터프레임만을 활용하여 필요한 분석을 수행하는 코드를 생성하세요.
2. 분석 대상 데이터프레임은 사용자의 요청을 기반으로 자동 선택하거나, 여러 개를 조합하여 사용하세요.
3. 제공된 데이터프레임 목록 (아래 목록 이외의 데이터 사용 금지):
4. **분석 결과 저장 방식**
   - 분석 결과는 dictionary 타입의 `analytic_result` 변수에 저장해야 합니다.
   - Key에는 분석 단계의 이름, Value에는 해당 단계의 결과를 저장해주세요. Value의 데이터타입은 가급적 데이터프레임을 사용해주세요.
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
2. **오류 수정 시, 패키지 버전을 확인하고 해당 버전에 맞는 코드로 수정하세요.**
   - 현재 실행 환경의 패키지 버전:
   {installed_packages}
3. **코드만 제공하고, 추가 설명이나 주석을 포함하지 마세요.**
4. **결과 저장 형식 (`analytic_result`)**
   - 분석 결과는 dictionary 형태의 `'analytic_result'` 변수에 저장해야 합니다.
   - Key: 분석 단계의 이름, Value: 해당 단계의 결과
   - **집계 데이터(aggregated data)**는 전체 데이터를 저장하고 반드시 `print()`로 출력하세요.
   - **비집계 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
5. 만약 오류가 난 코드 내에서 에러를 못찾겠다면 다른 방식으로 코드를 생성해주세요(예 : 명령어를 사용하던 방식을 수식으로 변경)

🔹 **제공된 데이터프레임을 반드시 활용해야 하며, 새로운 데이터를 생성하는 코드는 절대 포함하지 마세요.**
"""

PROMPT_REGENERATE_CODE_WHEN_TOKEN_OVER = """
# 📌 Python 코드 생성 (토큰 초과 대응)
현재 생성된 코드는 실행 성공하였으나, 결과 데이터 'analytic_result(dict)'에 지나치게 많은 토큰량이 들어있습니다.
데이터 분석 결과의 scale down을 반영한 코드로 재생성해주세요.

✅ **필수 규칙**
1. 데이터 분석 결과의 scale down 부분만 수정해주세요.
2. **코드만 제공하고, 추가 설명이나 주석을 포함하지 마세요.**
3. **결과 저장 형식 (`analytic_result`)**
   - 분석 결과는 dictionary 형태의 `'analytic_result'` 변수에 저장해야 합니다.
   - Key: 분석 단계의 이름, Value: 해당 단계의 결과
   - **집계 데이터(aggregated data)**는 전체 데이터를 저장하고 반드시 `print()`로 출력하세요.
   - **비집계 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
   - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
4. 만약 오류가 난 코드 내에서 에러를 못찾겠다면 다른 방식으로 코드를 생성해주세요(예 : 명령어를 사용하던 방식을 수식으로 변경)

🔹 **제공된 데이터프레임을 반드시 활용해야 하며, 새로운 데이터를 생성하는 코드는 절대 포함하지 마세요.**
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

PROMPT_FEEDBACK_NEEDED = """
사용자 질의('user_question')에 대한 생성코드('validated_code'), 분석 결과('analytic_result'), 인사이트('insights')를 바탕으로 
추가적인 분석이 필요한지 여부를 판단해주세요.

아래는 추가 분석 상황 **예시**입니다. 예시 이외에도 추가적인 분석이 필요한 경우가 있을 수 있습니다.
- 데이터 전처리 과정의 고도화 부분이 필요한 경우
- 추가적인 변수(feature)를 고려하면 더 정확한 분석이 가능할 경우.
- 결과 해석에 한계가 있거나, 모델이 비즈니스적으로 충분한 설명력을 갖추지 못한 경우.
- 데이터 분석 결과에 대한 추가적인 해석이 필요한 경우
- 모델 성능 개선을 위한 추가적인 분석이 필요한 경우
- 모델의 신뢰성을 검증하기 위해 추가적인 분석이 필요한 경우
- 데이터 품질 문제(결측치, 이상치, 불균형 등)가 있을 가능성이 있는 경우.

다음 중 하나로만 대답해주세요:
- 'yes' : 추가적인 혹은 개선 필요한 경우
- 'no' : 추가적인 혹은 개선 필요 없는 경우
"""

PROMPT_FEEDBACK_PROCESS = """
당신은 데이터 분석과 머신러닝 모델링을 전문적으로 다루는 AI 분석 어시스턴트입니다.
사용자질의(`user_request`), 분석 결과('analysis_result), 분석코드('validated_code')를 바탕으로 추가 분석이 필요한 부분에 대해 상세히 설명해주세요.
 
다음 형식으로 설명해주세요:
1. 부족한 영역
   사용자질의만으로는 파악할 수 없는 분석의 영역에 대해서 피드백해 주세요.
2. 상세한 보완 방법
   1. 부족한 영역의 내용을 보완하기 위한 고급화된 분석 방법론 소개
   방법론에 대한 설명시 ** 방법론 : 방법론에 대한 이론적 설명**형태로 작성해주세요
   최대한 **다양하고** **전문적인** 방법론에 대햐여 설명하시오
3. 기대효과
"""

# 4. 파이썬을 이용한 예시 코드 작성
#    방법론 만큼 코드를 각각 구분하여 만들며 상세하게 만들어 주세요

PROMPT_FEEDBACK_POINT = """
피드백 내용 기준(`feedback_analysis`)으로 추가적인 데이터 분석을 최대 3가지 제안해주세요.
제안 내용은 문장 형식으로 30자 이내로 작성해주세요.
단, **분석 방법론** 혹은 **모델링**과 연관있는 키워드여야 합니다.
"""

 