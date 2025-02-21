import os
import pandas as pd
import numpy as np
import traceback
from IPython.display import display
from datetime import datetime
from typing import TypedDict, List, Literal, Annotated, Dict, Union
from pydantic import BaseModel
from uuid import uuid4
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from prompt.prompts import *
from utils.vector_handler import load_vectorstore

# ✅ 한글 폰트 설정 (Windows 환경)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ✅ AI 분석 에이전트 상태 정의(state에 적재된 데이터를 기반으로 이동)
class State(TypedDict):
    messages: List[HumanMessage]  # 🔹 사용자와 AI 간의 대화 메시지 목록
    query: str  # 🔹 사용자의 원본 질문 (query)
    dataframe: pd.DataFrame  # 🔹 현재 활성화된 데이터프레임 (분석 대상)
    mart_info: str  # 🔹 현재 활성화된 데이터프레임 (분석 대상)
    generated_code: str  # 🔹 LLM이 생성한 Python 코드 (분석을 수행하기 위한 코드)
    validated_code: str  # 🔹 샘플 실행을 통과한 유효한 Python 코드
    analytic_result: pd.DataFrame  # 🔹 전체 데이터를 실행하여 얻은 최종 결과 데이터프레임
    error_message: str  # 🔹 코드 실행 중 발생한 오류 메시지 (있다면 재시도할 때 활용)
    data_id: str  # 🔹 분석 결과를 저장할 때 부여되는 고유 ID (파일 저장 시 활용)
    insights: str  # 🔹 LLM이 분석 결과를 바탕으로 생성한 주요 인사이트
    chart_decision: str  # 🔹 차트 생성 여부를 판단한 결과 (yes/no)
    chart_filename: str  # 🔹 생성된 차트의 파일 경로 (없으면 None)
    report_filename: str  # 🔹 생성된 리포트 파일의 경로 (마크다운 형태로 저장)
    chart_needed: bool  # 🔹 차트가 필요한지 여부 (True: 필요함, False: 불필요)
    retry_chart: int  # 🔹 차트 생성 실패 시 재시도 횟수 (최대 3회)
    q_category: str  # 🔹 Supervisor가 판단한 질문 유형 (Analytics, General, Knowledge)
    general_response: str  # 🔹 General 질문에 대한 응답
    knowledge_response: str  # 🔹 Knowledge 질문에 대한 응답
    retry_count: int  # 🔹 코드 재생성 실패 시 재시도 횟수 (최대 3회)
    is_aggregated: bool  # 🔹 집계 데이터 여부 (True: 집계된 데이터, False: 집계되지 않은 데이터)
    chart_error: int  # 🔹 차트 생성 횟수 카운터

# ✅ 경로 결정용 라우터
class Router(BaseModel):
    next: Literal["Analytics", "General", "Knowledge", "Generate_Code", "Execute_Sample", "Regenerate_Code", "Execute_Full", 
                  "Save_Data", "Retrieve_Data", "Insight_Builder", "Chart_Builder", "Report_Builder", "__end__"]

class DataAnayticsAssistant:
    """Python DataFrame 기반 AI 분석 에이전트 (LangGraph 기반)"""

    def __init__(self, openai_api_key: str, mart_info : pd.DataFrame = None):
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.0)
        self.active_marts = None
        self.build_graph()
        self.chart_counter = 0  # 차트 카운터 초기화 추가
        self.mart_info = mart_info


    def build_graph(self):
        """LangGraph를 활용하여 분석 흐름 구성"""
        workflow = StateGraph(State)

        # 노드 추가
        workflow.add_node("Supervisor", self.supervisor)
        workflow.add_node("Analytics", self.handle_analytics)
        workflow.add_node("General", self.handle_general)
        workflow.add_node("Knowledge", self.handle_knowledge)
        workflow.add_node("Generate_Code", self.generate_python_code)
        workflow.add_node("Execute_Sample", self.execute_sample_code)
        workflow.add_node("Regenerate_Code", self.regenerate_code)
        workflow.add_node("Execute_Full", self.execute_full_data)
        workflow.add_node("Save_Data", self.save_data)
        workflow.add_node("Determine_Aggregation", self.determine_aggregation)
        workflow.add_node("Insight_Builder", self.generate_insights)
        workflow.add_node("Chart_Builder", self.generate_chart)
        workflow.add_node("Regenerate_Chart", self.regenerate_chart)
        workflow.add_node("Report_Builder", self.generate_report)

        # 기본 흐름 정의
        workflow.add_edge(START, "Supervisor")
        workflow.add_conditional_edges(
            "Supervisor",
            lambda state: state["q_category"],  # Supervisor가 결정한 경로로 이동
            {
                "Analytics": "Analytics",
                "General": "General",
                "Knowledge": "Knowledge",
            }
        )

        # ✅ 분석 (analytics) 흐름
        workflow.add_edge("Analytics", "Generate_Code")
        workflow.add_edge("Generate_Code", "Execute_Sample")

        # 조건부 라우팅 설정
        workflow.add_conditional_edges(
            "Execute_Sample",
            self.route_after_sample,
            {
                "Execute_Full": "Execute_Full",
                "Regenerate_Code": "Regenerate_Code",
                END : END
            }
        )

        # ✅ 코드 재생성 흐름
        workflow.add_conditional_edges(
            "Regenerate_Code",
            lambda state: END if state.get("retry_count", 0) >= 3 else "Execute_Sample",
            {
                "Execute_Sample": "Execute_Sample",
                END: END  # ✅ 3회 이상이면 종료
            }
        )

        workflow.add_edge("Execute_Full", "Save_Data")
        workflow.add_edge("Save_Data", "Determine_Aggregation")
        workflow.add_edge("Determine_Aggregation", "Insight_Builder")
        workflow.add_conditional_edges(
            "Insight_Builder",
            self.route_after_insights,
            {
                "Chart_Builder": "Chart_Builder",
                "Report_Builder": "Report_Builder"
            }
        )

        # 차트 생성 관련 조건부 라우팅 수정
        workflow.add_conditional_edges(
            "Chart_Builder",
            self.route_after_chart,
            {
                "Regenerate_Chart": "Regenerate_Chart",  # 실패 시 재생성
                "Report_Builder": "Report_Builder",  # 성공 또는 최대 재시도 초과
            }
        )
        
        # 차트 재생성 후 라우팅
        workflow.add_conditional_edges(
            "Regenerate_Chart",
            self.route_after_chart,
            {
                "Regenerate_Chart": "Regenerate_Chart",  # 여전히 실패 시 다시 재생성
                "Report_Builder": "Report_Builder",  # 성공 또는 최대 재시도 초과
            }
        )

        workflow.add_edge("Report_Builder", END)

        
        self.graph = workflow.compile()
        print("✅ 그래프 컴파일 완료")        
        

    def ask(self, user_request: str, data_info: Dict[str, pd.DataFrame] = None):
        """LangGraph 실행"""
        logo = """
 _____  _____  _   _  _____   _____  _____  _   _ ______   ___  
| ___ \|_   _|| \ | ||  ___| |  __ \|  ___|| \ | || ___ \ / _ \ 
| |_/ /  | |  |  \| || |__   | |  \/| |__  |  \| || |_/ // /_\ l
|  __/   | |  | . ` ||  __|  | | __ |  __| | . ` || ___ \|  _  |
| |     _| |_ | |\  || |___  | |_\ \| |___ | |\  || |_/ /| | | |
\_|     \___/ \_| \_/\____/   \____/\____/ \_| \_/\____/ \_| |_/
        """
        print(logo)
        print(f"🧐 새로운 요청 처리 시작: '{user_request}'")
        # data_info를 임시 저장
        return self.graph.invoke({"messages": [HumanMessage(content=user_request)],}, config={"recursion_limit": 15})

    def supervisor(self, state: State) -> Command:
        """다음 단계를 결정하는 Supervisor"""
        print("="*100)  # 구분선 추가
        print("👨‍💼 Supervisor 단계:")
        
        prompt = ChatPromptTemplate.from_messages([
                ("system", PROMPT_SUPERVISOR),
                ("user", " user_request:\n{user_request}\n\n")
        ])
        chain = prompt | self.llm.with_structured_output(Router)
        response = chain.invoke({"user_request": state['messages'][-1].content})
        print(f"🏃🏿‍➡️ 다음 단계: {response.next}")
        
        return Command(update={"q_category": response.next}, goto=response.next)
    
    def handle_analytics(self, state: State) -> Command:
        """분석 요청을 처리하는 노드"""
        print("\n📊 [handle_analytics] 분석 요청 처리 시작")
        
        # 데이터프레임 및 마트 정보 확인
        # if not self.mart_info:
        #     return Command(
        #         update={"error_message": "활성화된 데이터마트가 없습니다. 먼저 데이터마트를 선택해주세요."}, 
        #         goto=END
        #     )
            
        # Analytics 분기에서 data_info 활용
        return Command(update={"mart_info": self.mart_info.to_markdown() if hasattr(self, 'mart_info') else None},goto="Generate_Code")

    def handle_general(self, state: State) -> Command:
        """일반적인 질문을 처리하는 노드"""
        print("\n💬 [handle_general] 일반 질문 처리")
        prompt = ChatPromptTemplate.from_messages([
                ("system", PROMPT_GENERAL),
                ("user", " user_request:\n{user_request}\n\n")
        ])
                
        chain = prompt | self.llm
        user_request = state['messages'][0].content
        response = chain.invoke({"user_request": user_request})
        print(f"💡 일반 응답: {response.content}")
        return Command(update={"general_response": response.content}, goto=END)

    def handle_knowledge(self, state: State) -> Command:
        """지식 기반 응답을 처리하는 노드"""
        print("\n📚 [handle_knowledge] 지식 기반 질문 처리")

        # FAISS 벡터스토어 로드
        vectorstore = load_vectorstore()
        if vectorstore is None:
            print("❌ 벡터스토어를 로드할 수 없습니다. FAISS 인덱스를 확인하세요.")
            return Command(update={"knowledge_response": "관련된 정보를 찾을 수 없습니다."}, goto=END)

        # Retriever 생성
        retriever = vectorstore.as_retriever()

        # 사용자 질문 검색
        user_question = state['messages'][-1].content
        retrieved_docs = retriever.get_relevant_documents(user_question)

        if not retrieved_docs:
            response = "관련된 정보를 찾을 수 없습니다."
        else:
            # 검색된 문서 상위 3개를 컨텍스트로 활용
            context = "\n".join([doc.page_content for doc in retrieved_docs[:3]])
            prompt = ChatPromptTemplate.from_messages([
                    ("system", "다음 문서를 참고하여 사용자의 질문에 답변해주세요"),
                    ("user", "\n질문:\n{user_question}"),
                    ("user", "\ndocument:\n{context}")
            ])
            chain = prompt | self.llm
            response = chain.invoke({"user_question": user_question, "context": context})
        print(f"📖 지식 기반 응답: {response.content}")

        return Command(update={"knowledge_response": response.content}, goto=END)

    ###########################################################################################################
    ###########################################################################################################

    def generate_python_code(self, state):
        """사용자의 요청을 기반으로 Python 코드 생성"""
        print("="*100)  # 구분선 추가
        print("🤖 코드 생성 단계:")
        user_request = state["messages"][-1].content
        
        # 데이터프레임 정보 생성
        df_info = ""
        if hasattr(self, 'active_marts') and self.active_marts:
            for mart_name, df in self.active_marts.items():
                df_info += f"\n## {mart_name} 데이터프레임 ##\n"
                df_info += str(df.head())
                df_info += "\n"
        else:
            df_info = "데이터프레임 정보가 없습니다."

        prompt = ChatPromptTemplate.from_messages([
                    ("system", PROMPT_GENERATE_CODE),
                    ("user", "\nuser_request:\n{user_request}"),
                    ("user", "\ndf_info:\n{df_info}")
            ])
        chain = prompt | self.llm
        response = chain.invoke({
            "user_request": user_request,
            "df_info": df_info
        })
        print(f"🤖 생성된 코드:\n{response.content}\n")
        return Command(update={"generated_code": response.content}, goto="Execute_Sample")
    
    def execute_sample_code(self, state):
        """샘플 데이터를 활용하여 Python 코드 실행"""
        print("="*100)  # 구분선 추가
        print("🧪 샘플 실행 단계")
        # 각 마트별로 샘플 데이터 생성
        sample_marts = {}
        for mart_name, df in self.active_marts.items():
            sample_size = min(50, len(df))
            sample_marts[mart_name] = df.sample(n=sample_size)
            print(f"🧪 {mart_name}: {sample_size}개 샘플 추출")
        # print(f"🧪 샘플 코드 실행 직전 글로벌 키 확인(접근 가능 데이터프레임) \n {globals().keys()} ")

        try:
            # LLM 응답에서 코드 부분만 추출
            code_to_execute = state["generated_code"]
            if "```python" in code_to_execute:
                code_to_execute = code_to_execute.split("```python")[1].split("```")[0].strip()
            elif "```" in code_to_execute:
                code_to_execute = code_to_execute.split("```")[1].strip()
            
            # 실행 환경에 샘플 데이터프레임 추가
            exec_globals = {}

            # 기본 데이터프레임(df) 자동 할당
            if len(sample_marts) == 1:
                exec_globals["df"] = list(sample_marts.values())[0]  # 유일한 마트를 df로 할당
            elif len(sample_marts) > 1:
                exec_globals["df"] = list(sample_marts.values())[0]  # 첫 번째 마트를 기본 df로 설정

            # 모든 데이터프레임을 exec_globals에 추가
            exec_globals.update(sample_marts)

            # print(f"🔹 실행 환경에 추가된 데이터프레임 목록: {list(exec_globals.keys())}")

            # 추출된 코드 실행
            exec(code_to_execute, exec_globals)
            print(f"✅ 샘플 코드 실행 성공")
            return Command(update={
                "validated_code": state["generated_code"],  # ✅ 실행 성공 시 validated_code 업데이트
                "retry_count": 0,  # ✅ 성공했으므로 retry_count 초기화
                "error_message": None  # ✅ 성공했으므로 error_message 초기화
            }, goto="Execute_Full")

        except Exception as e:
            print(f"❌ 샘플 코드 실행 실패")
            print(f"에러 타입: {type(e).__name__}")
            print(f"에러 메시지: {str(e)}")
            print(f"에러 발생 위치:")
            print(traceback.format_exc())
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
                
            # 재시도 횟수 확인
            retry_count = state.get("retry_count", 0)
            return Command(update={"error_message": error_details, "retry_count": retry_count + 1}, goto="Regenerate_Code")


    def regenerate_code(self, state):
        """코드 실행 오류 발생 시 LLM을 활용하여 코드 재생성"""
        retry_count = state.get("retry_count", 0)

        if retry_count >= 3:  # ✅ 3회 초과 시 종료
            return Command(goto=END)
        
        print("="*100)  # 구분선 추가
        print("🔄 코드 재생성 단계")
        user_request = state["messages"][-1].content
        error_message = state["error_message"]
        original_code = state["generated_code"]

        prompt = ChatPromptTemplate.from_messages([
                    ("system", PROMPT_REGENERATE_CODE),
                    ("user", "\nuser_request:\n{user_request}"),
                    ("user", "\noriginal_code:\n{original_code}"),
                    ("user", "\nerror_message:\n{error_message}"),
            ])
        chain = prompt | self.llm
        response = chain.invoke({
            "user_request": user_request,
            "original_code": original_code,
            "error_message": error_message
        })
        print(f"🔄 재생성된 코드:\n{response.content}\n")
        return Command(update={"generated_code": response.content}, goto="Execute_Sample")


    def execute_full_data(self, state):
        """전체 데이터로 Python 코드 실행"""
        print("="*100)  # 구분선 추가
        print("🔄 전체 데이터 실행 단계")

        # 전체 데이터프레임 설정
        full_marts = self.active_marts  # 전체 데이터프레임 사용

        # 실행 환경에 전체 데이터프레임 추가
        exec_globals = {}

        # 기본 데이터프레임(df) 자동 할당
        if len(full_marts) == 1:
            exec_globals["df"] = list(full_marts.values())[0]  # 유일한 마트를 df로 할당
        elif len(full_marts) > 1:
            exec_globals["df"] = list(full_marts.values())[0]  # 첫 번째 마트를 기본 df로 설정

        # 모든 데이터프레임을 exec_globals에 추가
        exec_globals.update(full_marts)

        print(f"🔄 전체 데이터 실행 환경에 추가된 데이터프레임 목록: {list(exec_globals.keys())}")

        # LLM 생성 코드에서 ```python 블록 제거
        code_to_execute = state["validated_code"]
        if "```python" in code_to_execute:
            code_to_execute = code_to_execute.split("```python")[1].split("```")[0].strip()
        elif "```" in code_to_execute:
            code_to_execute = code_to_execute.split("```")[1].strip()

        try:
            # 코드 실행
            exec(code_to_execute, exec_globals)
            result = exec_globals.get("result_df", None)

            print(f"🔄 전체 데이터 실행 결과 (처음 5행)")
            display(result.head().round(2))

            if result is not None:
                # 결과가 단일 값인 경우 DataFrame으로 변환
                if isinstance(result, (int, float)):
                    result = pd.DataFrame({'결과': [result]})
                    
                unique_id = self.generate_unique_id()
                
                return Command(update={"analytic_result": result, "data_id": unique_id}, goto="Save_Data")

        except Exception as e:
            print(f"❌ 전체 데이터 실행 실패\n {code_to_execute}")
            print(f"에러 타입: {type(e).__name__}")
            print(f"에러 메시지: {str(e)}")
            print(f"에러 발생 위치:")
            print(traceback.format_exc())
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            return Command(update={"error_message": error_details}, goto="Regenerate_Code")


    def save_data(self, state):
        """처리된 데이터를 저장 (ID 부여)"""
        print("="*100)  # 구분선 추가
        print("📂 처리 데이터 저장 단계")
        # data_id가 없는 경우 생성
        data_id = state.get("data_id", self.generate_unique_id())
        analytic_result = state["analytic_result"]


        

        
        # 저장 디렉토리 확인 및 생성
        os.makedirs("../output", exist_ok=True)
        analytic_result.to_pickle(f"../output/data_{data_id}.pkl")
        print(f"📂 처리된 데이터 저장 경로: ../output/data_{data_id}.pkl")
        return Command(update={"data_id": data_id}, goto="Determine_Aggregation")
    
        
    def determine_aggregation(self, state):
            """LLM 생성 코드가 집계 데이터를 반환하는지 확인"""
            print("="*100)
            print("⚖️ 데이터 집계 여부 판단 단계")
            df_result = state["analytic_result"]
            
            # ✅ 자동 Raw 데이터 판별 기준 (행 50개 이상 AND 컬럼 50개 이상)
            if df_result.shape[0] >= 50 and df_result.shape[1] >= 50:
                print("⚖️ 데이터가 Raw 데이터로 분류됨 (행과 컬럼 개수 기준)")
                return Command(update={"is_aggregated": False}, goto="Insight_Builder")
            
            # ✅ LLM 코드 분석을 통한 집계 판단
            prompt = ChatPromptTemplate.from_messages([
                ("system", PROMPT_DETERMINE_AGGREGATION),
                ("user", "질문:{user_question}"),
                ("user", "생성된 코드:{generated_code}")
            ])

            chain = prompt | self.llm
            response = chain.invoke({
                "user_question": state["messages"][-1].content,
                "generated_code": state["generated_code"]
            }).content.lower()
            
            is_aggregated = "yes" in response
            if is_aggregated:
                print(f"⚖️ 집계 판단 결과: {is_aggregated} 데이터가 집계 데이터로 분류됨 ")
            else:
                print(f"⚖️ 집계 판단 결과: {is_aggregated} 데이터가 Raw 데이터로 분류됨 ")
            
            return Command(update={"is_aggregated": is_aggregated}, goto="Insight_Builder")


    def generate_insights(self, state):
        """저장된 데이터에서 자동 인사이트 도출 및 차트 필요 여부 결정"""
        print("="*100)  # 구분선 추가
        print("💡 인사이트 도출 단계")
        df_result = state["analytic_result"]
        user_question = state["messages"][0].content

        # ✅ 집계 데이터면 전체 데이터 전달
        print(f"💡 집계 데이터 여부: {state.get('is_aggregated', False)}")
        df_string = df_result.round(2).to_string() if state.get("is_aggregated", False) else df_result.head().round(2).to_string()

        ############################################################
        # 1. 인사이트 생성
        ############################################################
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_INSIGHT_BUILDER),
            ("user", "user_question:\n{user_question}\n\n"),
            ("user", "analysis_result:\n{analysis_result}\n\n")
        ])

        chain = prompt | self.llm
        insight_response = chain.invoke({
            "user_question": user_question,
            "analysis_result": df_string
        })

        print(f"💡 생성된 인사이트:\n{insight_response.content}")
        
        ############################################################
        # 2. 차트 필요 여부 결정
        ############################################################
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_CHART_NEEDED),
            ("user", "user_question:\n{user_question}\n\n"),
            ("user", "analysis_result:\n{analysis_result}\n\n"),
            ("user", "insight:\n{insight}\n\n")
        ])
        
        chart_decision_messages = prompt | self.llm
        chart_needed = chart_decision_messages.invoke({
            "user_question": user_question,
            "analysis_result": df_string,
            "insight": insight_response.content
        }).content.strip().lower()
        print(f"💡 차트 필요 여부: {chart_needed}")
        
        # 차트 필요 여부에 따라 다음 단계 결정
        next_step = "Chart_Builder" if chart_needed == "yes" else "Report_Builder"
        
        return Command(update={
            "insights": insight_response.content,
            "chart_needed": chart_needed == "yes"
        }, goto=next_step)  # Supervisor 대신 적절한 다음 단계로 이동
        

    def generate_chart(self, state):
        """차트 생성 로직 (최대 3회 재시도)"""
        print("="*100)  # 구분선 추가
        print("📊 차트 생성 단계")
        df_result = state["analytic_result"]
        df_string = df_result.to_string() if state.get("is_aggregated", False) else df_result.head().round(2).to_string()

        retry_cnt = state.get("retry_chart", 0)  # 🔹 차트 생성 재시도 횟수
        # ✅ 3회 초과 실패 시 차트 없이 Report_Builder로 이동
        if retry_cnt >= 3:
            print("⚠️ 차트 생성 3회 실패. 차트 없이 리포트 생성으로 이동합니다.")
            return Command(update={"chart_filename": None, "retry_chart": 0}, goto="Report_Builder")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_CHART_GENERATOR),
            ("user", """analysis_result:\n{analysis_result}"""),
            ("user", """previous_insights:\n{previous_insights}""")
        ])

        chain = prompt | self.llm
        chart_code = chain.invoke({
            "analysis_result": df_string,
            "previous_insights": state.get('insights', '인사이트 없음')
        }).content

        # ✅ 차트 코드 블록이 있는 경우 코드 추출
        if "```python" in chart_code:
            extracted_code = chart_code.split("```python")[-1].split("```")[0].strip()
        else:
            print("📊 [generate_chart] 유효한 Python 코드 블록이 없습니다. 재시도합니다.")
            return Command(update={"retry_chart": retry_cnt + 1}, goto="Chart_Builder")

        # 🔹 기존에 LLM이 생성한 코드에서 `plt.show()` 제거
        extracted_code = extracted_code.replace("plt.show()", "").strip()

        # 🔹 차트 저장 디렉토리 생성
        os.makedirs("../img", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"../img/chart_{timestamp}.png"

        # 🔹 `plt.savefig()`를 먼저 실행한 후 `plt.show()` 추가
        extracted_code += f"\nplt.savefig('{filename}', dpi=300)\nplt.show()"

        # ✅ 디버깅용 출력 (생성된 코드 확인)
        print(f"📊 실행할 차트 코드:\n{extracted_code}")

        # 🔹 차트 코드 실행
        try:
            exec(extracted_code, globals())  # 🔹 차트 코드 실행
            print(f"✅ 차트 생성 성공: {filename}")
            plt.close()
            # 성공 시 chart_filename 업데이트하고 retry_chart 초기화
            return Command(
                update={
                    "chart_filename": filename,
                    "retry_chart": 0,
                    "chart_error": None
                },
                goto="Report_Builder"  # 성공 시 바로 Report_Builder로
            )

        except Exception as e:
            print(f"❌ 차트 코드 실행 중 오류 발생: {e}")
            plt.close()
            error_info = {
                "error_message": str(e),
                "previous_code": extracted_code,
                "traceback": traceback.format_exc()
            }
            # 실패 시 Regenerate_Chart로
            return Command(
                update={
                    "chart_filename": None,
                    "retry_chart": state.get("retry_chart", 0) + 1,
                    "chart_error": error_info
                },
                goto="Regenerate_Chart"
            )

    def regenerate_chart(self, state):
        """차트 생성 실패 시 에러를 기반으로 차트 재생성"""
        print("="*100)
        print("🔄 차트 재생성 단계")
        
        df_result = state["analytic_result"]
        df_string = df_result.to_string() if state.get("is_aggregated", False) else df_result.head().round(2).to_string()
        previous_error = state.get("chart_error", {})
        retry_cnt = state.get("retry_chart", 0)

        if retry_cnt >= 3:
            print("⚠️ 차트 재생성 3회 실패. 차트 없이 리포트 생성으로 이동합니다.")
            return Command(update={
                "chart_filename": None,
                "retry_chart": 0,
                "chart_error": None
            }, goto="Report_Builder")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Python 코드에서 발생한 오류를 수정해주세요."),  # 차트 재생성 전용 프롬프트
            ("user", """
    이전 차트 생성 시도에서 다음과 같은 에러가 발생했습니다:
    에러 메시지: {error_message}

    이전 코드:
    {previous_code}

    전체 에러 내용:
    {error_traceback}

    분석할 데이터:
    {analysis_result}

    인사이트:
    {insights}

    위의 에러를 해결한 새로운 차트 코드를 생성해주세요.
            """)
        ])

        chain = prompt | self.llm
        chart_code = chain.invoke({
            "error_message": previous_error.get("error_message", "알 수 없는 에러"),
            "previous_code": previous_error.get("previous_code", "이전 코드 없음"),
            "error_traceback": previous_error.get("traceback", "트레이스백 없음"),
            "analysis_result": df_string,
            "insights": state.get('insights', '인사이트 없음')
        }).content

        if "```python" in chart_code:
            extracted_code = chart_code.split("```python")[-1].split("```")[0].strip()
        else:
            print("📊 [regenerate_chart] 유효한 Python 코드 블록이 없습니다. 재시도합니다.")
            error_info = {
                "error_message": "유효한 Python 코드 블록이 없습니다",
                "previous_code": chart_code
            }
            return Command(update={
                "retry_chart": retry_cnt + 1,
                "chart_error": error_info
            }, goto="Regenerate_Chart")

        extracted_code = extracted_code.replace("plt.show()", "").strip()
        
        os.makedirs("../img", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"../img/chart_{timestamp}.png"
        
        extracted_code += f"\nplt.savefig('{filename}', dpi=300)\nplt.show()"
        
        print(f"📊 실행할 차트 코드:\n{extracted_code}")

        try:
            exec(extracted_code, globals())
            print(f"✅ 차트 재생성 성공: {filename}")
            plt.close()
            return Command(update={
                "chart_filename": filename,
                "retry_chart": 0,
                "chart_error": None
            }, goto="Report_Builder")

        except Exception as e:
            print(f"❌ 차트 재생성 중 오류 발생: {e}")
            plt.close()
            error_info = {
                "error_message": str(e),
                "previous_code": extracted_code,
                "traceback": traceback.format_exc()
            }
            return Command(update={
                "chart_filename": None,
                "retry_chart": retry_cnt + 1,
                "chart_error": error_info
            }, goto="Regenerate_Chart")


    def generate_report(self, state):
        """최종 보고서 생성"""
        print("="*100)  # 구분선 추가
        print("📑 보고서 생성 단계")
        df_result = state["analytic_result"]
        df_string = df_result.to_string() if state.get("is_aggregated", False) else df_result.head().round(2).to_string()
        insights = state.get('insights', '인사이트 없음')
        user_request = state['messages'][-1].content

        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_REPORT_GENERATOR),
            ("user", "1. 분석 결과 데이터\n{analysis_result}\n\n"),
            ("user", "2. 사용자 요청\n{user_request}\n\n"),
            ("user", "3. 도출된 인사이트\n{insights}\n\n"),
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "user_request": user_request,
            "analysis_result": df_string,
            "insights": insights,
        })
        print("✅ 보고서 생성 완료")
        print(f"{response.content}")
        return Command(update={"report_filename": response.content}, goto=END)
    
    
    def set_active_mart(self, dataframe: Union[pd.DataFrame, Dict[str, pd.DataFrame]], mart_name: Union[str, List[str], None] = None, ) -> None:
        """분석할 데이터프레임을 설정
        Args:
            dataframe (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): 단일 데이터프레임 또는 데이터프레임 딕셔너리
            mart_name (Union[str, List[str]], optional): 데이터마트의 이름 또는 이름 리스트. 기본값은 None
        """
        if isinstance(dataframe, pd.DataFrame):
            # 단일 데이터프레임 설정 (이름이 제공되면 key로 사용)
            self.active_marts = {mart_name if mart_name else "default_mart": dataframe}
        elif isinstance(dataframe, dict):
            # 다중 데이터프레임 설정
            self.active_marts = dataframe
        else:
            raise TypeError("입력된 데이터가 pandas DataFrame 또는 DataFrame 딕셔너리가 아닙니다.")

        # 📊 데이터프레임 개수 및 정보 출력
        print(f"✅ {len(self.active_marts)}개의 데이터프레임이 성공적으로 설정되었습니다.")
        for name, df in self.active_marts.items():
            print(f"📊 데이터마트 이름: {name}")
            print(f"🔹 데이터 크기: {df.shape[0]}행 x {df.shape[1]}열")


    def route_after_sample(self, state: State):
        """샘플 실행 후 다음 단계를 결정하는 라우터"""
        print("➡️ [route_after_sample] 샘플 실행 후 경로 결정")
        retry_count = state.get("retry_count", 0)
        is_validated = "validated_code" in state

        if is_validated:
            print("➡️ [route_after_sample]  전체 데이터 실행 진행")
            return "Execute_Full"
        else :
            if retry_count >= 3:
                print("⚠️ 샘플 코드 실행 3회 실패 → 프로세스 종료")
                return END
            else :
                print(f"⚠️ 샘플 코드 실행 실패 → 코드 재생성 필요 | 재시도 횟수: {retry_count}")
                return "Regenerate_Code"


    def route_after_insights(self, state: State) -> str:
        """인사이트 생성 후 다음 단계를 결정하는 라우터"""
        print("="*100)  # 구분선 추가
        print(f"➡️ 인사이트 라우팅 단계")
        
        if state.get("chart_needed", False):
            print("📊 차트 생성 단계로 진행합니다")
            return "Chart_Builder"
        print("📝 보고서 생성 단계로 진행합니다")
        return "Report_Builder"
    
    def route_after_chart(self, state: State) -> str:
        """차트 생성 후 다음 단계를 결정하는 라우터"""
        retry_cnt = state.get("retry_chart", 0)
        
        # 차트 생성 성공 (파일명이 있는 경우)
        if state.get("chart_filename"):
            print("✅ 차트 생성 성공 → 리포트 생성 단계로 진행")
            return "Report_Builder"
        
        # 최대 재시도 횟수 초과
        if retry_cnt >= 3:
            print("⚠️ 차트 생성 3회 실패 → 차트 없이 리포트 생성으로 진행")
            return "Report_Builder"
        
        # 재시도 필요
        print(f"⚠️ 차트 생성 실패 → 재생성 시도 ({retry_cnt + 1}/3)")
        return "Regenerate_Chart"

    
    def generate_unique_id(self):
        """고유 ID 생성"""
        return datetime.now().strftime("%Y%m%d%H%M%S")
