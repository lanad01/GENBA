from asyncio import Task
from email import generator
from itertools import tee
from altair import Key


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


# PROMPT_GENERATE_CODE = """
# 사용자 요청을 수행하기 위한 파이썬 코드를 생성해주세요:

# 다음 규칙을 반드시 따라주세요:
# 1. 결과는 반드시 result_df 변수에 저장해주세요
# 2. 데이터프레임은 'df' 변수로 제공됩니다
# 3. 코드만 제공해주세요 (설명 없이)
# 4. 예제 데이터프레임 생성을 하지말고, 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요
# 5. 단, 실행 가능한 코드인지 내부적으로 검증한 후 반환하세요.
# 6. print 함수는 사용하지말아주세요.
# """

# PROMPT_GENERATE_CODE = """
# 사용자 요청에 대한 Python 코드를 생성해주세요.
# 사용할 데이터프레임은 반드시 'df' 변수로 제공되며, 새롭게 예제 데이터를 생성하지 마세요.

# 다음 규칙을 반드시 따라주세요:
# 1. **제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.**
# 2. **결과 저장 형식 (`analytic_results`)**
#    - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
#    - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
#    - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
#    - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
#    - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
# 3. **코드만 제공하고, 추가 설명이나 주석을 포함하지 마세요.**
# """

PROMPT_GENERATE_CODE = """
# 📌 Python 코드 생성 규칙
사용자의 요청에 맞는 Python 코드를 작성하세요.  
**코드만 제공하고, 추가 설명이나 주석을 포함하지 마세요.**

아래는 이미 메모리에 로드되어 있는 활용할 데이터프레임에 대한 개요입니다. (`pd.read_csv()`, `pd.DataFrame()` 등을 사용하여 **새로운 데이터를 생성하지 마세요.**) 
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


# PROMPT_REGENERATE_CODE = """
# Python 코드에서 발생한 오류를 수정해주세요.
# 사용할 데이터프레임은 반드시 'df' 변수로 제공되며, 새롭게 예제 데이터를 생성하지 마세요.

# 아래 사항을 준수하세요.
# 1. 사용자의 원래 요청을 유지하면서 오류를 수정하세요.
# 2. **코드만 제공하고, 추가 설명이나 주석을 포함하지 마세요.**
# 3. **결과 저장 형식 (`analytic_results`)**
#    - 분석 결과를 dictionary 형태의 'analytic_results' 변수에 저장해주세요.
#    - 각 분석 단계를 Key, 해당 결과를 value로 갖는 구조여야 합니다.
#    - **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
#    - **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
#    - **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
# """

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

🔹 **오류를 수정하면서 기존 분석 흐름을 유지하세요.**
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

🔹 **제공된 데이터프레임만을 사용하여, 요청된 분석을 수행하세요.**
"""

# PROMPT_REGENERATE_CODE_WHEN_TOKEN_OVER = """
# 사용자 요청을 수행하기 위한 파이썬 코드를 생성해주세요:
# 사용할 DataFrame은 'df' 변수로 제공되며, 반드시 새로운 예제 데이터를 생성하지 마세요.

# 1. 분석 결과는 `result_df`(DataFrame)으로 반환해주세요.
# 2. 코드만 제공해주세요 (설명 없이)
# 3. **비집계성 데이터(non-aggregated data)**는 `head()` 적용 후 `round(2)` 처리한 데이터를 저장하세요.
# 4. **집계성 데이터(aggregated data)**는 전체 데이터를 저장하고, 반드시 `print()`로 출력하세요.
# 5. **모든 수치형 데이터는 `round(2)`를 적용**한 후 저장하세요.
# """

# PROMPT_DETERMINE_AGGREGATION = """
# 사용자의 질문과 LLM이 생성한 코드를 기반으로, 결과가 집계된 데이터인지 판단하세요.

# 다음 중 하나로만 대답해주세요:
# - 'yes': 집계된 데이터인 경우
# - 'no': 집계되지 않은 데이터인 경우
# """

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

# PROMPT_CHART_GENERATOR = """
# You are an agent specialized in data visualization.

# **Chart Builder Agent Prompt**

# Your task is to create charts based on the dataframe provided by the user. Follow these guidelines:

# 1. **Input Data**: The user provides data in the form of dataframe results, structured as a list of tuples, where each tuple represents a row and contains values corresponding to column headers.

# 2. **Request Analysis**:
#    - If the user specifies a chart type (e.g., bar chart, line chart, pie chart), create the requested chart.
#    - If no specific chart type is mentioned, analyze the data and suggest the most suitable chart type.

# 3. **Output Results**:
#    - Only generate code for the chart using Python's Matplotlib libraries. No other text or comments.
#    - Ensure the chart includes a title, axis labels, legend, and other necessary elements to clearly visualize the data.

# 4. **Additional Requests**:
#    - Incorporate any user-specified adjustments, such as changing axis labels, customizing colors, or filtering data.
#    - Aggregate or transform the data if needed to create the requested chart.

# 5. **Compatibility Considerations**:
#    - Avoid including custom code that could cause errors in different environments. For example, do not hardcode font paths like '/usr/share/fonts/truetype/nanum/NanumGothic.ttf' as this will likely result in errors when executed in other systems.
# """
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
