import streamlit as st
import sys
import io
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# 환경 설정
from dotenv import load_dotenv
import os
import pandas as pd
from ai_agent_v2 import DataAnayticsAssistant

# OpenAI API 키 로드
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

mart_name = "cust_intg"
df = pd.read_pickle(f'../data/{mart_name}.pkl')

# 데이터프레임 설정
llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)

# 생성형 AI가 생성한 코드를 실행하고 출력을 저장하는 함수
def execute_code_with_capture(code):
    captured_output = io.StringIO()
    original_stdout = sys.stdout  # 원래 표준 출력 저장
    sys.stdout = captured_output  # 표준 출력 변경

    analysis_results = {}  # 실행 결과 저장 변수

    try:
        exec(code, globals())  # 생성된 코드 실행
    except Exception as e:
        print(f"Error: {str(e)}")  # 에러 발생 시 출력

    sys.stdout = original_stdout  # 표준 출력을 원래 상태로 복원
    return captured_output.getvalue(), analysis_results  # 실행된 print 결과 및 analysis_results 반환

# AI 프롬프트 템플릿
PROMPT_GENERATE_CODE = """
사용자 요청에 대한 파이썬 코드를 생성해주세요:
활용할 데이터프레임은 'df' 변수로 제공됩니다.

다음 규칙을 반드시 따라주세요:
1. 예제 데이터프레임 생성을 하지말고, 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요.
2. 분석 결과를 dictionary 형태의 'analysis_results' 변수에 저장해주세요. 저장 시 아래의 **규칙**을 따라주세요.
   **규칙**
   - 집계성 데이터가 아닌 경우에는 반드시 head()를 한 뒤에 저장해주세요. 
   - 'analysis_results'는 각 분석 단계를 key, 해당 결과를 value로 갖는 구조여야 합니다.
3. 코드만 제공해주세요.
4. 집계성 데이터는 반드시 print를 찍어주세요.
"""

# Streamlit UI
st.title("AI 코드 실행 및 결과 분석")

# 사용자 요청 입력
user_request = st.text_area("수행할 분석 요청을 입력하세요", "수치형 변수 간의 상관관계를 분석하고, 강한 상관관계를 가지는 변수 쌍을 탐색해 주세요.")

if st.button("코드 생성 및 실행"):
    # AI를 통해 코드 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_GENERATE_CODE),
        ("user", f"user_request:\n{user_request}\n\n")
    ])
    chain = prompt | llm
    response = chain.invoke({"user_request": user_request})

    # 생성된 코드 표시
    ai_generated_code = response.content.split('```python')[1].split('```')[0]
    st.subheader("📝 생성된 코드")
    st.code(ai_generated_code, language="python")

    # 실행할 코드의 설명 요청
    explain_prompt = f"다음 파이썬 코드의 목적과 주요 로직을 설명해 주세요.\n\n{ai_generated_code}"
    explanation = chain.invoke({"user_request": explain_prompt}).content

    # 실행할 코드 설명 출력
    st.subheader("📌 코드 설명")
    st.write(explanation)

    # 실행한 코드의 출력 결과 저장 및 표시
    st.subheader("🔍 코드 실행 결과")
    output, analysis_results = execute_code_with_capture(ai_generated_code)

    # 실행 과정에서 print로 출력된 데이터
    if output.strip():
        st.text_area("📢 실행 과정에서 print된 출력", output, height=200)
    else:
        st.write("📌 실행 중 print된 데이터가 없습니다.")

    # analysis_results 객체 출력
    st.subheader("📊 분석 결과 데이터")
    if analysis_results:
        for key, value in analysis_results.items():
            st.write(f"**{key}**:")
            st.write(value)
    else:
        st.write("🔍 분석 결과 데이터가 없습니다.")
