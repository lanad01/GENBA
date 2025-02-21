import os
import pandas as pd
import traceback
from datetime import datetime
from typing import TypedDict, List, Literal, Annotated
from pydantic import BaseModel
from uuid import uuid4
import matplotlib.pyplot as plt
import matplotlib
import operator
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.prompts import PromptTemplate

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain.vectorstores import FAISS
from langchain.schema import Document

# ✅ 한글 폰트 설정 (Windows 환경)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ✅ AI 분석 에이전트 상태 정의
class State(TypedDict):
    messages: List[HumanMessage]
    query: str
    dataframe: pd.DataFrame
    generated_code: str
    validated_code: str
    processed_data: pd.DataFrame
    error_message: str
    data_id: str
    insights: str
    chart_decision: str
    chart_filename: str
    report_filename: str
    chart_needed : bool

# ✅ 경로 결정용 라우터
class Router(BaseModel):
    next: Literal["Generate_Code", "Execute_Sample", "Regenerate_Code", "Execute_Full", 
                  "Save_Data", "Retrieve_Data", "Insight_Builder", "Chart_Builder", "Report_Builder", "__end__"]

class AIDataFrameAssistant:
    """Python DataFrame 기반 AI 분석 에이전트 (LangGraph 기반)"""

    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
        self.active_mart = None
        self.build_graph()
        
    def set_active_mart(self, dataframe: pd.DataFrame, mart_name: str = None) -> None:
        """분석할 데이터프레임을 설정합니다.
        
        Args:
            dataframe (pd.DataFrame): 분석할 데이터프레임
            mart_name (str, optional): 데이터마트의 이름. 기본값은 None
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("입력된 데이터가 pandas DataFrame이 아닙니다.")
            
        self.active_mart = dataframe
        print(f"✅ 데이터프레임이 성공적으로 설정되었습니다.")
        print(f"📊 데이터 크기: {dataframe.shape[0]}행 x {dataframe.shape[1]}열")
        
        if mart_name:
            print(f"🏷️ 데이터마트 이름: {mart_name}")
    

    def route_after_sample(self, state: State) -> str:
        """샘플 실행 후 다음 단계를 결정하는 라우터"""
        print(f"🔄 샘플 실행 후 다음 단계를 결정하는 라우터")
        print(f"현재 상태: {state}")
        print(f"검증된 코드 존재: {'validated_code' in state}")
        
        if "error_message" in state:
            print("⚠️ 에러 발생으로 코드 재생성이 필요합니다")
            return "Regenerate_Code"
            
        if "validated_code" in state:
            print("✅ 코드 검증 완료, 전체 데이터 실행으로 진행합니다")
            return "Execute_Full"
            
        print("⚠️ 코드 검증 실패로 재생성이 필요합니다")
        return "Regenerate_Code"

    def route_after_insights(self, state: State) -> str:
        """인사이트 생성 후 다음 단계를 결정하는 라우터"""
        print(f"🔄 인사이트 라우팅 단계:")
        
        if state.get("chart_needed", False):
            print("📊 차트 생성 단계로 진행합니다")
            return "Chart_Builder"
        print("📝 보고서 생성 단계로 진행합니다")
        return "Report_Builder"

    def build_graph(self):
        """LangGraph를 활용하여 분석 흐름 구성"""
        print("\n🔨 그래프 구성 시작")
        workflow = StateGraph(State)

            # 노드 추가
        workflow.add_node("Supervisor", self.supervisor)
        workflow.add_node("Generate_Code", self.generate_python_code)
        workflow.add_node("Execute_Sample", self.execute_sample_code)
        workflow.add_node("Regenerate_Code", self.regenerate_code)
        workflow.add_node("Execute_Full", self.execute_full_data)
        workflow.add_node("Save_Data", self.save_data)
        workflow.add_node("Insight_Builder", self.generate_insights)
        workflow.add_node("Chart_Builder", self.generate_chart)
        workflow.add_node("Report_Builder", self.generate_report)

        print("✅ 노드 추가 완료")

        # 기본 흐름 정의
        workflow.add_edge(START, "Supervisor")
        workflow.add_edge("Supervisor", "Generate_Code")
        workflow.add_edge("Generate_Code", "Execute_Sample")
        workflow.add_edge("Regenerate_Code", "Execute_Sample")
        
        # 조건부 라우팅 설정
        workflow.add_conditional_edges(
            "Execute_Sample",
            self.route_after_sample,
            {
                "Execute_Full": "Execute_Full",
                "Regenerate_Code": "Regenerate_Code"
            }
        )
        
        workflow.add_edge("Execute_Full", "Save_Data")
        workflow.add_edge("Save_Data", "Insight_Builder")
        workflow.add_conditional_edges(
            "Insight_Builder",
            self.route_after_insights,
            {
                "Chart_Builder": "Chart_Builder",
                "Report_Builder": "Report_Builder"
            }
        )
        workflow.add_edge("Chart_Builder", "Report_Builder")
        workflow.add_edge("Report_Builder", END)

        print("✅ 엣지 설정 완료")
        self.graph = workflow.compile()
        print("✅ 그래프 컴파일 완료\n")
        

    def run(self, user_request: str) :
        """LangGraph 실행"""
        print(f"\n🔄 새로운 요청 처리 시작: '{user_request}'")
        initial_state = {"messages": [HumanMessage(content=user_request)]}
        return self.graph.invoke(initial_state)

    def supervisor(self, state: State) -> Command:
        """다음 단계를 결정하는 Supervisor"""
        print("\n" + "="*100)  # 구분선 추가
        print("👨‍💼 Supervisor 단계:")
        
        prompt = PromptTemplate.from_template("""
다음 사용자 요청을 분석하고 적절한 다음 단계를 결정해주세요:
요청: {user_request}

다음 단계들 중 하나를 선택하세요:
1. Generate_Code: 새로운 코드 생성이 필요한 경우
2. Insight_Builder: 인사이트 생성이 필요한 경우
3. Chart_Builder: 차트 생성이 필요한 경우
4. Report_Builder: 리포트 생성이 필요한 경우
5. __end__: 모든 처리가 완료된 경우

Output only one of the following: "Generate_Code", "Insight_Builder", "Chart_Builder", "Report_Builder", "__end__"
""")
        
        chain = prompt | self.llm.with_structured_output(Router)
        response = chain.invoke({"user_request": state['messages'][-1].content})
        print(f"🔄 다음 단계: {response.next}")
        
        return Command(goto=response.next)

    def generate_python_code(self, state):
        """사용자의 요청을 기반으로 Python 코드 생성"""
        print("\n" + "="*100)  # 구분선 추가
        print("🤖 코드 생성 단계:")
        user_request = state["messages"][-1].content
        print(f"📝 요청 내용: {user_request}")
        
        messages = [
            HumanMessage(content=f"""
다음 요청에 대한 파이썬 코드를 생성해주세요:
요청: {user_request}

현재 데이터프레임 정보:
{str(self.active_mart.head(5))}

다음 규칙을 반드시 따라주세요:
1. 결과는 반드시 result_df 변수에 저장해주세요
2. 데이터프레임은 'df' 변수로 제공됩니다
3. 코드만 제공해주세요 (설명 없이)
4. 예제 데이터프레임 생성을 하지말고, 제공된 데이터프레임에 대한 처리를 해주는 코드를 생성해주세요


**예시:**
```python
result_df = df[df['구매금액'] >= 2000]['나이'].mean()
""")
        ]
        
        response = self.llm.invoke(messages)
        print(f"✨ 생성된 코드:\n{response.content.split("```python")[1].split("```")[0].strip()}\n")
        return Command(update={"generated_code": response.content}, goto="Execute_Sample")

    
    def execute_sample_code(self, state):
        """샘플 데이터를 활용하여 Python 코드 실행"""
        print("\n" + "="*100)  # 구분선 추가
        print("🧪 샘플 코드 실행 단계:")
        sample_df = self.active_mart.sample(n=min(50, len(self.active_mart)))
        exec_globals = {"df": sample_df}
        try:
            # LLM 응답에서 코드 부분만 추출
            code_to_execute = state["generated_code"]
            if "```python" in code_to_execute:
                code_to_execute = code_to_execute.split("```python")[1].split("```")[0].strip()
            elif "```" in code_to_execute:
                code_to_execute = code_to_execute.split("```")[1].strip()
            
            # 추출된 코드 실행
            exec(code_to_execute, globals())
            print(f"성공")
            return Command(update={"validated_code": state["generated_code"]}, goto="Execute_Full")

        except Exception as e:
            print(f"실패")
            print(f"에러 타입: {type(e).__name__}")
            print(f"에러 메시지: {str(e)}")
            print(f"에러 발생 위치:")
            import traceback
            print(traceback.format_exc())
            return Command(update={"error_message": str(e)}, goto="Regenerate_Code")

    def regenerate_code(self, state):
        """코드 실행 오류 발생 시 LLM을 활용하여 코드 재생성"""
        print("\n" + "="*100)  # 구분선 추가
        print("🔄 코드 재생성 단계:")
        error_message = state["error_message"]
        original_code = state["generated_code"]
        
        messages = [
            HumanMessage(content=f"""
    다음 코드에서 발생한 오류를 수정해주세요:

    원본 코드:
    {original_code}

    발생한 오류:
    {error_message}

    다음 규칙을 반드시 따라주세요:
    1. 결과는 반드시 result_df 변수에 저장해주세요
    2. 데이터프레임은 'df' 변수로 제공됩니다
    3. 수정된 코드만 제공해주세요 (설명 없이)
    """)
        ]
        
        # response = self.llm.invoke(messages)
        fixed_code = self.llm.invoke(messages).content

        return Command(update={"generated_code": fixed_code}, goto="Execute_Sample")


    def execute_full_data(self, state):
        """전체 데이터로 Python 코드 실행"""
        print("\n" + "="*100)  # 구분선 추가
        print("📊 전체 데이터 실행 단계:")
        exec_globals = {"df": self.active_mart, "pd": pd}
        
        # 코드 블록 구문 제거
        code_to_execute = state["validated_code"]
        if "```python" in code_to_execute:
            code_to_execute = code_to_execute.split("```python")[1].split("```")[0].strip()
        elif "```" in code_to_execute:
            code_to_execute = code_to_execute.split("```")[1].strip()
        
        print(code_to_execute)
        
        # 코드 실행
        exec(code_to_execute, exec_globals)
        result = exec_globals.get("result_df", None)
        
        if result is not None:
            # 결과가 단일 값인 경우 DataFrame으로 변환
            if isinstance(result, (int, float)):
                result = pd.DataFrame({'결과': [result]})
            unique_id = self.generate_unique_id()
            return Command(update={"processed_data": result, "data_id": unique_id}, goto="Save_Data")
        return Command(goto=END)

    def save_data(self, state):
        """처리된 데이터를 저장 (ID 부여)"""
        data_id = state["data_id"]
        processed_data = state["processed_data"]
        
        # 저장 디렉토리 확인 및 생성
        os.makedirs("../output", exist_ok=True)
        processed_data.to_pickle(f"../output/data_{data_id}.pkl")
        return {"saved_data_path": f"../output/data_{data_id}.pkl"}
    
    def generate_insights(self, state):
        """저장된 데이터에서 자동 인사이트 도출 및 차트 필요 여부 결정"""
        print("\n" + "="*100)  # 구분선 추가
        print("🔄 인사이트 도출 단계:")
        print(f"[LOG] state['processed_data']: ")
        df = state["processed_data"]
        user_question = state["messages"][0].content
        
        # 1. 인사이트 생성
        insight_messages = [
            HumanMessage(content=f"""
다음 분석 결과에 대한 인사이트를 도출해주세요.
이 인사이트 결과는 보험사에서 일하는 데이터 분석가에게 제공되는 결과물이며, 보험사 내부 문서로 사용됩니다.

원본 질문: {user_question}

분석 결과:
{df.to_string()}

다음 형식으로 응답해주세요:
1. 주요 발견사항
2. 특이점
3. 추천 사항
""")
        ]
        
        insight_response = self.llm.invoke(insight_messages)
        print(f"🌀 생성된 인사이트:\n{insight_response.content}")
        
        # 2. 차트 필요 여부 결정
        chart_decision_messages = [
            HumanMessage(content=f"""
분석 결과와 인사이트를 바탕으로 시각화(차트) 필요 여부를 판단해주세요:

원본 질문: {user_question}

분석 결과:
{df.to_string()}

도출된 인사이트:
{insight_response.content}

다음 중 하나로만 대답해주세요:
- 'yes': 시각화가 필요한 경우
- 'no': 시각화가 불필요한 경우
""")
        ]
        
        chart_decision = self.llm.invoke(chart_decision_messages).content.strip().lower()
        print(f"🌀 차트 필요 여부: {chart_decision}")
        
        return Command(update={
            "insights": insight_response.content,
            "chart_needed": chart_decision == "yes"
        }, goto="Supervisor")
        

    def generate_chart(self, state):
        """차트 생성 로직"""
        print("\n" + "="*100)  # 구분선 추가
        print("📊 차트 생성 단계:")
        df = state["processed_data"]
        
        messages = [
            HumanMessage(content=f"""
**Chart Builder Agent Prompt**

You are an agent specialized in data visualization. 
Your task is to create charts based on the SQL query result data provided by the user. Follow these guidelines:

1. **Input Data**: The user provides data in the form of SQL query results, structured as a list of tuples, where each tuple represents a row and contains values corresponding to column headers.
분석 결과:
{df.to_string()}

이전 단계 인사이트:
{state.get('insights', '인사이트 없음')}

2. **Request Analysis**:
   - If the user specifies a chart type (e.g., bar chart, line chart, pie chart), create the requested chart.
   - If no specific chart type is mentioned, analyze the data and suggest the most suitable chart type.

3. **Output Results**:
   - Generate code to create the chart using Python's Matplotlib libraries.
   - Ensure the chart includes a title, axis labels, legend, and other necessary elements to clearly visualize the data.

4. **Additional Requests**:
   - Incorporate any user-specified adjustments, such as changing axis labels, customizing colors, or filtering data.
   - Aggregate or transform the data if needed to create the requested chart.

5. **Compatibility Considerations**:
   - Avoid including custom code that could cause errors in different environments. For example, do not hardcode font paths like '/usr/share/fonts/truetype/nanum/NanumGothic.ttf' as this will likely result in errors when executed in other systems.
    """)
        ]
        response = self.llm.invoke(messages)
        print(f"✅ 차트 생성 완료: {response.content}")
        return Command(update={"chart_decision": response.content}, goto="Report_Builder")
    

    def generate_report(self, state):
        """최종 보고서 생성"""
        print("\n" + "="*100)  # 구분선 추가
        print("📑 보고서 생성 단계:")
        
        messages = [
            HumanMessage(content=f"""
    지금까지의 분석 결과를 종합하여 보고서를 작성해주세요:

    1. 원본 데이터 정보:
    {state.get('processed_data', '데이터 없음').head().to_string()}

    2. 사용자 요청:
    {state['messages'][-1].content}

    3. 도출된 인사이트:
    {state.get('insights', '인사이트 없음')}

    4. 생성된 차트 정보:
    {state.get('chart_decision', '차트 없음')}

    다음 형식으로 보고서를 작성해주세요:
    1. 요약
    2. 분석 방법
    3. 주요 발견사항
    4. 결론 및 제언

    응답은 마크다운 형식으로 작성해주세요.
    """)
        ]
        
        response = self.llm.invoke(messages)
        print("✅ 보고서 생성 완료")
        return Command(update={"chart_decision": response.content}, goto=END)
    
    
    def generate_unique_id(self):
        """고유 ID 생성"""
        return datetime.now().strftime("%Y%m%d%H%M%S")
