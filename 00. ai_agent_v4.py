import os, sys
import io
import pickle
import pandas as pd
import numpy as np
import traceback
from IPython.display import display
from datetime import datetime
from typing import TypedDict, List, Literal, Annotated, Dict, Union
from pydantic import BaseModel
import tiktoken
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
from typing import Annotated  # ✅ Python 기본 모듈에서 가져오기
from fuzzywuzzy import process


from prompt.prompts_v4 import *
from common_txt import logo
from utils.vector_handler import load_vectorstore

# ✅ 한글 폰트 설정 (Windows 환경)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

PROCESSED_DATA_PATH = "../output/stage1/processed_data_info.xlsx"
MAX_RETRIES = 3

eda_prompt_mapping = {
    "기본 정보 분석": PROMPT_EDA_BASIC_INFO,
    "기초 통계 분석": PROMPT_EDA_STATISTICAL_ANALYSIS,
    "결측치 처리": PROMPT_EDA_MISSING_VALUE_HANDLING,
    "변수 간 관계 분석": PROMPT_EDA_FEATURE_RELATIONSHIP,
    "이상치 탐지": PROMPT_EDA_OUTLIER_DETECTION
}


# ✅ AI 분석 에이전트 상태 정의(state에 적재된 데이터를 기반으로 이동)
class State(TypedDict):
    messages: List[HumanMessage]  # 🔹 사용자와 AI 간의 대화 메시지 목록
    query: str  # 🔹 사용자의 원본 질문 (query)
    dataframe: pd.DataFrame  # 🔹 현재 활성화된 데이터프레임 (분석 대상)
    mart_info: str  # 🔹 현재 활성화된 데이터프레임 (분석 대상)
    generated_code: Annotated[str, "last"]  # 🔹 LLM이 생성한 Python 코드 (분석을 수행하기 위한 코드)
    validated_code: str  # 🔹 샘플 실행을 통과한 유효한 Python 코드
    analytic_result: Dict  # 🔹 전체 데이터를 실행하여 얻은 최종 결과 딕셔너리
    execution_output: str  # 🔹 코드 실행 중 생성된 출력 텍스트
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
    chart_error: int  # 🔹 차트 생성 횟수 카운터
    eda_question: str  # 🔹 EDA 코드 생성 결과
    from_full_execution: bool  # 🔹 코드 재생성 시 초기 실행 여부
    eda_stage: int  # 🔹 EDA 단계 카운터
    request_summary: str  # 🔹 분석 요청을 한글로 요약한 내용
    regenerated_code: str  # 🔹 재생성된 코드
    analysis_type: str  # 🔹 분석 유형 (EDA, ML, General)

# ✅ 경로 결정용 라우터
class Router(BaseModel):
    next: Literal["Analytics", "General", "Knowledge", "Generate_Code", "Execute_Sample", "Regenerate_Code", "Execute_Full", 
                  "Save_Data", "Insight_Builder", "Chart_Builder", "Report_Builder", "__end__"]

class DataAnayticsAssistant:
    """Python DataFrame 기반 AI 분석 에이전트 (LangGraph 기반)"""

    def __init__(self, openai_api_key: str, mart_info : pd.DataFrame = None):
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.0)
        self.active_marts = None
        self.mart_info = None
        self.retry_count = 0

        # 마트 정보 초기 로드
        try:
            self.mart_info_df = pd.read_excel(PROCESSED_DATA_PATH, sheet_name=None)
            print(f"🔹 현재 접근 가능 마트 목록: {list(self.mart_info_df.keys())}")
        except Exception as e:
            print(f"⚠️ 마트 정보 로드 실패: {e}")
            self.mart_info_df = {}
            
        self.build_graph()


    def build_graph(self):
        """LangGraph를 활용하여 분석 흐름 구성"""

        workflow = StateGraph(State)

        # 기존 노드 추가
        workflow.add_node("Supervisor", self.supervisor)
        workflow.add_node("Analytics", self.handle_analytics)
        workflow.add_node("General", self.handle_general)
        workflow.add_node("Knowledge", self.handle_knowledge)
        workflow.add_node("Check_Analysis_Question", self.check_analysis_question)  # ✅ 추가
        workflow.add_node("Eda_Generate_Code", self.generate_eda_code)  # ✅ 추가
        workflow.add_node("ML_Generate_Code", self.generate_ml_code)  # ✅ 추가
        workflow.add_node("Generate_Code", self.generate_python_code)
        workflow.add_node("Execute_Sample", self.execute_sample_code)
        workflow.add_node("Regenerate_Code", self.regenerate_code)
        workflow.add_node("Execute_Full", self.execute_full_data)
        workflow.add_node("Save_Data", self.save_data)
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

        # ✅ Analytics → Check_EDA_Question 추가
        workflow.add_edge("Analytics", "Check_Analysis_Question")

        
        workflow.add_conditional_edges(
            "Check_Analysis_Question",
            lambda state: (
                "Eda_Generate_Code" if state.get("analysis_type") == "EDA" else
                "ML_Generate_Code" if state.get("analysis_type") == "ML" else
                "Generate_Code"
            ),
            {
                "Eda_Generate_Code": "Eda_Generate_Code",
                "ML_Generate_Code": "ML_Generate_Code",
                "Generate_Code": "Generate_Code",
            }
        )
        
        # ✅ 기존 흐름 유지
        workflow.add_edge("Eda_Generate_Code", "Execute_Sample")
        workflow.add_edge("ML_Generate_Code", "Execute_Sample")
        workflow.add_edge("Generate_Code", "Execute_Sample")

        # ✅ 샘플 실행 후 조건부 라우팅 설정
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
            self.route_after_regenerate,  # 새로운 라우터 함수 사용
            {
                "Execute_Sample": "Execute_Sample",
                "Execute_Full": "Execute_Full",
                END : END  # ✅ 3회 이상이면 종료
            }
        )

        workflow.add_conditional_edges(
            "Execute_Full",
            self.route_after_full_execution,
            {
                "Save_Data": "Save_Data",
                "Regenerate_Code": "Regenerate_Code",
                END : END
            }
        )

        workflow.add_edge("Save_Data", "Insight_Builder")
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

        # workflow.add_edge("Report_Builder", END)
        # Report_Builder에서 다음 EDA 단계 실행
        workflow.add_conditional_edges(
            "Report_Builder",
            route_after_report,  # ✅ 별도 함수에서 state를 받아 처리하도록 변경
            {
                "Eda_Generate_Code": "Eda_Generate_Code",
                "END": END,
            }
        )


        self.graph = workflow.compile()
        print("✅ 그래프 컴파일 완료")        
        

    def ask(self, user_request: str, data_info: Dict[str, pd.DataFrame] = None):
        """LangGraph 실행"""
        print("*"*100)
        print(logo)
        print("*"*100)
        print(f"🧐 새로운 요청 처리 시작: '{user_request}'")
        # data_info를 임시 저장
        return self.graph.invoke({"messages": [HumanMessage(content=user_request)],}, config={"recursion_limit": 150})

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
        
        return Command(update={"q_category": response.next, "eda_stage": 0}, goto=response.next)
    
    def handle_analytics(self, state: State) -> Command:
        """분석 요청을 처리하는 노드"""
        print("👨‍💼 [handle_analytics] 분석 요청 처리 시작")
        
        # 사용자 요청을 30자 이내 한글로 변환
        prompt = ChatPromptTemplate.from_messages([
            ("system", "사용자의 분석 요청을 10자 이내의 명사형으로 간단히 요약해주세요. 핵심만 영어로 작성해주세요."),
            ("user", "{request}")
        ])
        
        chain = prompt | self.llm
        request_summary = chain.invoke({
            "request": state['messages'][-1].content
        }).content.strip()
        
        print(f"🔍 변환된 분석 요청: {request_summary}")
        
        return Command(
            update={"request_summary": request_summary},
            goto="Check_Analysis_Question"
        )


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
    def check_analysis_question(self, state: State) -> Command:
        """사용자의 질문이 EDA, ML, 일반 질문인지 판단하는 노드"""
        print("=" * 100)
        print("🔍 분석 유형 판단 단계 (EDA vs ML vs 일반)")

        user_question = state["messages"][0].content

        # ✅ LLM 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_CHECK_ANALYSIS_QUESTION),
            ("user", "user_question:\n{user_question}\n\n")
        ])

        chain = prompt | self.llm
        analysis_decision = chain.invoke({"user_question": user_question}).content.strip().upper()

        # ✅ 분석 유형을 state에 저장
        state_update = {"analysis_type": analysis_decision if analysis_decision in ["EDA", "ML"] else "General"}

        # ✅ 분석 유형에 따라 적절한 노드로 이동
        next_node = (
            "Eda_Generate_Code" if state_update["analysis_type"] == "EDA" else
            "ML_Generate_Code" if state_update["analysis_type"] == "ML" else
            "Generate_Code"
        )
        print(f"🔍 분석 유형: {state_update['analysis_type']}, 다음 단계: {next_node}")

        return Command(update=state_update, goto=next_node)
        
    
    def generate_eda_code(self, state: State) -> Command:

        """사용자의 요청을 기반으로 Python 코드 생성"""
        print("="*100)  # 구분선 추가
        print("🤖 코드 생성 단계:")

        # 사용자의 질문을 가져오기
        user_request = state["messages"][-1].content.lower()
        
        # 사용자 질문을 기반으로 실행할 EDA 단계 선택
        selected_categories = self.map_to_eda_category(user_request)
        selected_categories = list(dict.fromkeys(selected_categories))
        print(f"🔍 선택된 카테고리: {selected_categories}")

        # 선택된 프롬프트만 실행
        if "전체" in selected_categories:
            selected_prompts = [PROMPT_EDA_FEATURE_IMPORTANCE_ANALYSIS]
        else:
            selected_prompts = [eda_prompt_mapping[category] for category in selected_categories if category in eda_prompt_mapping]
        print(f"🔍 선택된 프롬프트: {selected_prompts}")

        current_stage = state.get("eda_stage", 0)
        print(f"현재회차: {current_stage}, 수행회차: {len(selected_prompts)}")
        
        if current_stage >= len(selected_prompts):
            print("📌 모든 EDA 단계를 완료했습니다.")
            return Command(goto=END)
        
        # 현재 실행할 EDA 단계 출력
        print(f"📌 실행 중: {selected_prompts[current_stage]}")
        prompt_text = selected_prompts[current_stage]

        user_request = state["messages"][-1].content
        
        # 데이터프레임 정보 생성
        mart_info = ""
        if hasattr(self, 'active_marts') and self.active_marts:
            for mart_name, df in self.active_marts.items():
                mart_info += f"\n## {mart_name} 데이터프레임 ##\n"
                mart_info += str(df)
                mart_info += "\n"
        else:
            mart_info = "데이터프레임 정보가 없습니다."

        prompt = ChatPromptTemplate.from_messages([
                    ("system", prompt_text),
                    ("user", "\nuser_request:\n{user_request}"),
                    ("user", "\mart_info:\n{mart_info}")
            ])
        chain = prompt | self.llm
        response = chain.invoke({
            "user_request": user_request,
            "mart_info": mart_info
        })
        print(f"🤖 생성된 코드:\n{response.content}\n")

        return Command(update={
            "generated_code": response.content,
            "eda_stage": current_stage + 1,
            "regenerated_code": None,  # 초기화
            "validated_code": None     # 초기화
        }, goto="Execute_Sample")


    def generate_ml_code(self, state):
        """사용자의 요청을 기반으로 Python 코드 생성"""
        print("="*100)  # 구분선 추가
        print("🤖 코드 생성 단계:")
        user_request = state["messages"][-1].content

        # 🔹 실행할 ML 프로세스 목록 (고정된 순서로 진행)
        ml_process_steps = [
            "PROMPT_ML_SCALING",
            "PROMPT_ML_IMBALANCE_HANDLING",
            "PROMPT_ML_MODEL_SELECTION",
            "PROMPT_ML_HYPERPARAMETER_TUNING",
            "PROMPT_ML_MODEL_EVALUATION",
            "PROMPT_ML_FEATURE_IMPORTANCE"
        ]
        
        # 데이터프레임 정보 생성
        mart_info = ""
        if hasattr(self, 'active_marts') and self.active_marts:
            for mart_name, df in self.active_marts.items():
                mart_info += f"\n## {mart_name} 데이터프레임 ##\n"
                mart_info += str(df)
                mart_info += "\n"
        else:
            mart_info = "데이터프레임 정보가 없습니다."

        # 🔹 실행할 단계별 프롬프트 적용
        generated_code_list = []
        for prompt in ml_process_steps:
            prompt_text = globals().get(prompt, None)  # 문자열을 실제 프롬프트 변수로 변환
            if not prompt_text:
                print(f"⚠ {prompt} 프롬프트를 찾을 수 없습니다. 스킵합니다.")
                continue  # 해당 프롬프트가 없으면 건너뜀
            prompt_chain = ChatPromptTemplate.from_messages([
                ("system", prompt_text),
                ("user", "user_request:\n{user_request}"),
                ("user", "mart_info:\n{mart_info}")
            ])
            chain = prompt_chain | self.llm
            response = chain.invoke({
                "user_request": user_request,
                "mart_info": state.get("mart_info", "")
            })
            generated_code_list.append(response.content)

        # 🔹 Python 코드 블록만 추출
        extracted_code_blocks = []
        for code in generated_code_list:
            if "```python" in code and "```" in code:
                extracted_code = code.split("```python")[1].split("```")[0].strip()
                extracted_code_blocks.append(extracted_code)

        if not extracted_code_blocks:
            print("⚠ 코드 블록을 찾을 수 없습니다.")
            return Command(update={"generated_code": "# 오류: 코드 블록이 제공되지 않았습니다."})

        tmp_code = "\n\n".join(extracted_code_blocks).strip()

        # 🔹 코드가 비어있으면 실행 중지
        if not tmp_code:
            print("⚠ 생성된 코드가 없습니다.")
            return Command(update={"generated_code": "# 오류: 생성된 코드가 없습니다."})

        # 🔹 프롬프트 실행 (LLM을 이용하여 코드 통합)
        prompt_chain = ChatPromptTemplate.from_messages([
            ("system", PROMPT_MERGE_GENERAL_ML_CODE.replace("{", "{{").replace("}", "}}")),  # ✅ 중괄호 이스케이프 처리
            ("user", "사용자가 제공한 코드 블록:\n{merged_code}")
        ])
        chain = prompt_chain | self.llm
        response = chain.invoke({"merged_code": tmp_code})  # ✅ KeyError 해결
        # 🔹 통합된 최종 ML 코드 저장
        final_code = response.content.strip()

        print(f"🤖 생성된 ML 코드:\n{final_code[:500]}...\n")  # 일부만 출력
        return Command(update={
            "generated_code": final_code,
            "regenerated_code": None,  # 초기화
            "validated_code": None     # 초기화
        }, goto="Execute_Sample")
    ###########################################################################################################

    def generate_python_code(self, state):
        """사용자의 요청을 기반으로 Python 코드 생성"""
        print("="*100)  # 구분선 추가
        print("🤖 코드 생성 단계:")
        user_request = state["messages"][-1].content
        
        # 데이터프레임 정보 생성
        mart_info = ""
        if hasattr(self, 'active_marts') and self.active_marts:
            for mart_name, df in self.active_marts.items():
                mart_info += f"\n## {mart_name} 데이터프레임 ##\n"
                mart_info += str(df)
                mart_info += "\n"
        else:
            mart_info = "데이터프레임 정보가 없습니다."

        prompt = ChatPromptTemplate.from_messages([
                    ("system", PROMPT_GENERATE_CODE),
                    ("user", "\nuser_request:\n{user_request}"),
                    ("user", "\mart_info:\n{mart_info}")
            ])
        chain = prompt | self.llm
        response = chain.invoke({
            "user_request": user_request,
            "mart_info": mart_info
        })
        print(f"🤖 생성된 코드:\n{response.content}\n")
        return Command(update={
            "generated_code": response.content,
            "regenerated_code": None,  # 초기화
            "validated_code": None     # 초기화
        }, goto="Execute_Sample")
    
    def execute_sample_code(self, state):
        """샘플 데이터를 활용하여 Python 코드 실행"""
        print("="*100)  # 구분선 추가
        print("🧪 샘플 실행 단계")
        
        try:
            # 각 마트별로 샘플 데이터 생성
            sample_marts = {}
            for mart_name, df in self.active_marts.items():
                sample_size = min(50, len(df))
                sample_marts[mart_name] = df.sample(n=sample_size)
                print(f"🧪 {mart_name}: {sample_size}개 샘플 추출")
        except Exception as e:
            print(f"❌ 활성화된 마트가 없습니다. 먼저 마트를 활성화해주세요. {e}")
            return Command(update={"error_message": "❌ 활성화된 마트가 없습니다. 먼저 마트를 활성화해주세요."}, goto='__end__')
        
        # print(f"🧪 샘플 코드 실행 직전 글로벌 키 확인(접근 가능 데이터프레임) \n {globals().keys()} ")

        try:
            # 재생성된 코드가 있으면 그것을 사용, 없으면 초기 생성 코드 사용
            code_to_execute = self._extract_code_from_llm_response(
                state.get("regenerated_code") or state["generated_code"]
            )
            
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
            self._execute_code_with_capture(code_to_execute, exec_globals, is_sample=True)
            # exec(code_to_execute, exec_globals)
            
            print(f"✅ 샘플 코드 실행 성공")
            self.retry_count = 0  # 성공 시 카운터 초기화
            return Command(update={
                "error_message": None
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
            self.retry_count += 1
            return Command(update={"error_message": error_details}, goto="Regenerate_Code")


    def regenerate_code(self, state):
        """코드 실행 오류 발생 시 LLM을 활용하여 코드 재생성"""

        from_full_execution = state.get("from_full_execution", False)  # 플래그 확인
        print(f"재생성 번호: {self.retry_count}")
        print(f"retry_count:{self.retry_count}, MAX_RETRIES:{MAX_RETRIES}")
        if self.retry_count >= MAX_RETRIES:  # ✅ 3회 초과 시 종료
            return Command(goto=END)
        print(f"🔄 재생성 단계 진입 {from_full_execution}")
        
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
        next_step = "Execute_Full" if from_full_execution else "Execute_Sample"
        return Command(update={
            "regenerated_code": response.content,  # 재생성된 코드 저장
            "validated_code": None,  # validated_code 초기화
            "from_full_execution": from_full_execution
        }, goto=next_step)
        # return Command(update={"generated_code": response.content}, goto="Execute_Sample")


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
        # code_to_execute = state["validated_code"]
        # if "```python" in code_to_execute:
        #     code_to_execute = code_to_execute.split("```python")[1].split("```")[0].strip()
        # elif "```" in code_to_execute:
        #     code_to_execute = code_to_execute.split("```")[1].strip()

        # LLM 생성 코드에서 ```python 블록 제거
        code_to_execute = self._extract_code_from_llm_response(
            state.get("regenerated_code") or state["generated_code"]
        )

        try:
            # 전체 코드 실행
            output, analytic_results = self._execute_code_with_capture(code_to_execute, exec_globals, is_sample=False)
            token_count = self._calculate_tokens(str(analytic_results))
            
            # ✅ 토큰 제한 설정 (예: 5000 토큰 초과 시 차단)
            TOKEN_LIMIT = 10000
            print(f"🔄 결과 데이터 토큰 수: {token_count}")
            
            if token_count > TOKEN_LIMIT:
                print(f"⚠️ 토큰 수 초과: {token_count} > {TOKEN_LIMIT}")
                self.retry_count += 1
                return Command(update={
                    "error_message": f"결과 데이터 analytic_results의 적정 토큰 수를 초과하였습니다. analytic_results에 Raw 데이터 혹은 불필요한 반복 적재를 피해주세요: {token_count} > {TOKEN_LIMIT}",
                    "from_full_execution": True,  # 플래그 추가
                    "from_token_limit": True
                }, goto="Regenerate_Code")
            
            print(f"🔄 전체 데이터 실행 성공")
            print(f'🔄 analytic_results\n {analytic_results}')
            # print(f'🔄 : output\n {output}')

            # 분석 결과가 있는 경우
            if analytic_results is not None:
                unique_id = self.generate_unique_id()
                # 전체 실행 성공 시 validated_code 설정
                current_code = state.get("regenerated_code") or state["generated_code"]
                return Command(update={
                    "analytic_result": analytic_results,
                    "execution_output": output,
                    "data_id": unique_id,
                    "validated_code": current_code  # 성공한 코드를 validated_code로 저장
                }, goto="Save_Data")
            # 분석 결과가 없는 경우
            else:
                print("⚠️ 분석 결과가 없습니다.")
                self.retry_count += 1
                return Command(update={
                    "error_message": "분석 결과가 없습니다.",
                    "execution_output": output,
                    "from_full_execution": True  # 플래그 추가
                }, goto="Regenerate_Code")

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
            self.retry_count += 1
            return Command(update={
                "error_message": error_details,
                "from_full_execution": True  # 플래그 추가
            }, goto="Regenerate_Code")


    def save_data(self, state):
        """처리된 데이터를 저장 (ID 부여)"""
        print("="*100)  # 구분선 추가
        print("📂 처리 데이터 저장 단계")
        # data_id가 없는 경우 생성
        data_id = state.get("data_id", self.generate_unique_id())
        analytic_result = state["analytic_result"]
        execution_output = state["execution_output"]
        # 분석 결과와 실행 출력을 함께 저장
        save_data = {
            'analytic_result': analytic_result,
            'execution_output': execution_output
        }

        # 저장 디렉토리 확인 및 생성
        os.makedirs("../output", exist_ok=True)
        
        # pickle로 저장
        with open(f"../output/data_{data_id}.pkl", 'wb') as f:
            pickle.dump(save_data, f)

        # 저장 디렉토리 확인 및 생성
        os.makedirs("../output", exist_ok=True)
        with open(f"../output/data_{data_id}.pkl", 'wb') as f:
            pickle.dump(save_data, f)

        print(f"📂 처리된 데이터 저장 경로: ../output/data_{data_id}.pkl")
        return Command(update={"data_id": data_id}, goto="Insight_Builder")
    
    def generate_insights(self, state):
        """저장된 데이터에서 자동 인사이트 도출 및 차트 필요 여부 결정"""
        print("="*100)  # 구분선 추가
        print("💡 인사이트 도출 단계")
        dict_result = state["analytic_result"]
        user_question = state["messages"][0].content

        # ✅ 집계 데이터면 전체 데이터 전달
        string_of_result = str(dict_result)

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
            "analysis_result": string_of_result
        })

        print(f"💡 생성된 인사이트\n{insight_response.content}")
        
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
            "analysis_result": string_of_result,
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

        if self.retry_count >= MAX_RETRIES:
            print("⚠️ 차트 생성 3회 실패. 차트 없이 리포트 생성으로 이동합니다.")
            self.retry_count = 0  # 카운터 초기화
            return Command(update={"chart_filename": None}, goto="Report_Builder")
        
        try:
            analytic_result = state.get("analytic_result", {})
            string_of_result = str(analytic_result)
            insights = state.get('insights', '인사이트 없음')
            prompt = ChatPromptTemplate.from_messages([
                ("system", PROMPT_CHART_GENERATOR),
                ("user", """
Here are the analytic_result and insights to visualize:

Analysis Results Summary:
{analytic_result}

Key Insights:
{insights}

Please create an appropriate visualization that supports these insights.
Do not hardcode any values - use the analytic_result dictionary directly.
Consider the data type of the columns when creating the chart.
                """)            
            ])

            chain = prompt | self.llm
            chart_code = chain.invoke({
                "analytic_result": string_of_result,
                "insights": insights
            }).content
            
            print(f"💡 생성된 차트 코드\n{chart_code}")
            
            
            # ✅ 차트 코드 블록이 있는 경우 코드 추출
            extracted_code = self._extract_code_from_llm_response(chart_code)
            
            # 🔹 기존에 LLM이 생성한 코드에서 `plt.show()` 제거
            extracted_code = extracted_code.replace("plt.show()", "").strip()

            # 🔹 차트 저장 디렉토리 생성
            os.makedirs("../img", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"../img/chart_{timestamp}.png"

            # 🔹 `plt.savefig()`를 먼저 실행한 후 `plt.show()` 추가
            extracted_code += f"\nplt.savefig('{filename}', dpi=300)\nplt.show()"

            # ✅ 디버깅용 출력 (생성된 코드 확인)
            # print(f"📊 생성된 차트 코드\n{extracted_code}")
            
            # 실행 환경 설정
            exec_globals = {
                'plt': plt,
                'np': np,
                'analytic_result': analytic_result,
            }
            
            # 코드 실행
            exec(extracted_code, exec_globals)
            plt.close()

            print(f"✅ 차트 생성 성공: {filename}")
            self.retry_count = 0  # 성공 시 카운터 초기화
            return Command(update={
                "chart_filename": filename,
                "chart_error": None
            }, goto="Report_Builder")

        except Exception as e:
            print(f"❌ 차트 코드 실행 중 오류 발생: {e}")
            plt.close()
            error_info = {
                "error_message": str(e),
                "previous_code": extracted_code,
                "traceback": traceback.format_exc()
            }
            # ❌ 실패 시 Regenerate_Chart로
            self.retry_count += 1
            return Command(
                update={
                    "chart_filename": None,
                    "chart_error": error_info
                },
                goto="Regenerate_Chart"
            )

    def regenerate_chart(self, state):
        """차트 생성 실패 시 에러를 기반으로 차트 재생성"""
        print("="*100)
        print("🔄 차트 재생성 단계")
        
        dict_result = state["analytic_result"]
        string_of_result = str(dict_result)
        previous_error = state.get("chart_error", {})
        retry_cnt = state.get("retry_chart", 0)

        if retry_cnt >= MAX_RETRIES:
            print("⚠️ 차트 재생성 3회 실패. 차트 없이 리포트 생성으로 이동합니다.")
            self.retry_count = 0  # 카운터 초기화
            return Command(update={
                "chart_filename": None,
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
    {analytic_result}

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
            "analytic_result": string_of_result,
            "insights": state.get('insights', '인사이트 없음')
        }).content

        extracted_code = self._extract_code_from_llm_response(chart_code)
        if not extracted_code:
            print("📊 [regenerate_chart] 유효한 Python 코드 블록이 없습니다. 재시도합니다.")
            error_info = {
                "error_message": "유효한 Python 코드 블록이 없습니다",
                "previous_code": chart_code
            }
            self.retry_count += 1
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
            self.retry_count = 0  # 성공 시 카운터 초기화
            return Command(update={
                "chart_filename": filename,
                "chart_error": None
            }, goto="Report_Builder")

        except Exception as e:
            print(f"❌ 차트 재생성 중 오류 발생: {e}")
            plt.close()
            error_info = {"error_message": str(e),"previous_code": extracted_code,"traceback": traceback.format_exc()}
            self.retry_count += 1
            return Command(update={
                "chart_filename": None,
                "chart_error": error_info
            }, goto="Regenerate_Chart")


    def generate_report(self, state):
        """최종 보고서 생성"""
        print("="*100)  # 구분선 추가
        print("📑 보고서 생성 단계")
        dict_result = state["analytic_result"]
        string_of_result = str(dict_result)
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
            "analysis_result": string_of_result,
            "insights": insights,
        })
        print("✅ 보고서 생성 완료")
        print(f"{response.content}")
        return Command(update={"report_filename": response.content}, goto=END)
    
    
    # def set_active_mart(self, data_mart: Union[pd.DataFrame, Dict[str, pd.DataFrame]], mart_name: Union[str, List[str], None] = None) -> None:
    #     """분석할 데이터프레임과 마트 정보를 설정"""
    #     if isinstance(data_mart, pd.DataFrame):
    #         # 단일 데이터프레임 설정
    #         mart_key = mart_name if mart_name else "default_mart"
    #         self.active_marts = {mart_key: data_mart}
    #         # 해당 마트의 정보 설정
    #         self.mart_info = self.mart_info_df.get(mart_key, pd.DataFrame()).to_markdown() if mart_key in self.mart_info_df else None
    #     elif isinstance(data_mart, dict):
    #         # 다중 데이터프레임 설정
    #         self.active_marts = data_mart
    #         # 모든 활성화된 마트의 정보를 결합
    #         mart_info_list = []
    #         for mart_key in self.active_marts.keys():
    #             if mart_key in self.mart_info_df:
    #                 mart_info_list.append(f"## {mart_key} 마트 정보\n{self.mart_info_df[mart_key].to_markdown()}")
    #         self.mart_info = "\n\n".join(mart_info_list) if mart_info_list else None
    #     else:
    #         raise TypeError("입력된 데이터가 pandas DataFrame 또는 DataFrame 딕셔너리가 아닙니다.")

    #     # 데이터프레임 개수 및 정보 출력
    #     print(f"🔹 {len(self.active_marts)}개의 데이터프레임이 성공적으로 설정되었습니다.")
    #     for name, df in self.active_marts.items():
    #         print(f"🔹 데이터마트 이름: {name}")
    #         print(f"🔹 데이터 크기: {df.shape[0]}행 x {df.shape[1]}열")
    #         if self.mart_info and name in self.mart_info_df:
    #             print(f"🔹 마트 정보 로드됨")
    def set_active_mart(self, data_mart: Union[pd.DataFrame, Dict[str, pd.DataFrame]], mart_name: Union[str, List[str], None] = None) -> None:
        """분석할 데이터프레임과 마트 정보를 설정"""
        if isinstance(data_mart, pd.DataFrame):
            # 단일 데이터프레임 설정
            mart_key = mart_name if mart_name else "default_mart"
            self.active_marts = {mart_key: data_mart}
            # 해당 마트의 정보 설정
            self.mart_info = self.mart_info_df.get(mart_key, pd.DataFrame()).to_markdown() if mart_key in self.mart_info_df else None
        elif isinstance(data_mart, dict):
            # 다중 데이터프레임 설정
            self.active_marts = data_mart
            # 모든 활성화된 마트의 정보를 결합
            mart_info_list = []
            for mart_key in self.active_marts.keys():
                if mart_key in self.mart_info_df:
                    mart_info_list.append(f"## {mart_key} 마트 정보\n{self.mart_info_df[mart_key].to_markdown()}")
            self.mart_info = "\n\n".join(mart_info_list) if mart_info_list else None
        else:
            raise TypeError("입력된 데이터가 pandas DataFrame 또는 DataFrame 딕셔너리가 아닙니다.")

        # 데이터프레임 개수 및 정보 출력
        print(f"🔹 {len(self.active_marts)}개의 데이터프레임이 성공적으로 설정되었습니다.")
        for name, df in self.active_marts.items():
            print(f"🔹 데이터마트 이름: {name}")
            print(f"🔹 데이터 크기: {df.shape[0]}행 x {df.shape[1]}열")
            if self.mart_info and name in self.mart_info_df:
                print(f"🔹 마트 정보 로드됨")



    def route_after_sample(self, state: State):
        """샘플 실행 후 다음 단계를 결정하는 라우터"""
        print("➡️ [route_after_sample] 샘플 실행 후 경로 결정")
        
        if not self.active_marts or self.active_marts is None:
            print("➡️ [route_after_sample] 활성화된 마트가 없습니다. 먼저 마트를 활성화해주세요.")
            return END
        
        if not state.get("error_message"):  # 에러가 없으면
            print("➡️ [route_after_sample] 전체 데이터 실행 진행")
            return "Execute_Full"
        else:
            if self.retry_count >= MAX_RETRIES:
                print("⚠️ 샘플 코드 실행 3회 실패 → 프로세스 종료")
                # self.retry_count = 0
                return END
            print(f"⚠️ 샘플 코드 실행 실패 → 코드 재생성 필요 | 재시도 횟수: {self.retry_count}")
            return "Regenerate_Code"


    def route_after_insights(self, state: State) -> str:
        """인사이트 생성 후 다음 단계를 결정하는 라우터"""
        print("="*100)  # 구분선 추가
        print(f"➡️ [route_after_insights] 인사이트 생성 후 경로 결정(차트 or 보고서)")
        
        if state.get("chart_needed", False):
            print("➡️ [route_after_insights] 차트 생성 단계로 진행합니다")
            return "Chart_Builder"
        print("➡️ [route_after_insights] 보고서 생성 단계로 진행합니다")
        return "Report_Builder"
    
    def route_after_chart(self, state: State) -> str:
        """차트 생성 후 다음 단계를 결정하는 라우터"""
        print(f"➡️ [route_after_chart] 차트 생성 후 경로 결정(차트 재생성 or 보고서)")

        if state.get("chart_filename"):
            print("➡️ [route_after_chart] 차트 생성 성공 → 리포트 생성 단계로 진행")
            return "Report_Builder"
        
        if self.retry_count >= MAX_RETRIES:
            print("⚠️ 차트 생성 3회 실패 → 차트 없이 리포트 생성으로 진행")
            self.retry_count = 0  # 카운터 초기화
            return "Report_Builder"
        
        print(f"➡️ 차트 생성 실패 → 재생성 시도 (Regenerate_Chart) ({self.retry_count + 1}/3)")
        return "Regenerate_Chart"


    def route_after_regenerate(self, state: State) -> str:
        """코드 재생성 후 다음 단계를 결정하는 라우터"""
        from_full_execution = state.get("from_full_execution", False)
        if self.retry_count >= MAX_RETRIES:
            print("⚠️ 코드 재생성 3회 실패 → 프로세스 종료")
            return END
        
        if from_full_execution:
            print("➡️ [route_after_regenerate] 전체 데이터 실행으로 진행")
            return "Execute_Full"
        else:
            print("➡️ [route_after_regenerate] 샘플 실행으로 진행")
            return "Execute_Sample"
        

    def generate_unique_id(self):
        """고유 ID 생성"""
        return datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 생성형 AI가 생성한 코드를 전체 데이터 기준으로 실행하고 출력을 저장하는 함수
    def _execute_code_with_capture(self, code, exec_globals, is_sample=False):
        
        # 표준 출력을 가로채기 위해 StringIO 사용
        captured_output = io.StringIO()
        original_stdout = sys.stdout  # 원래 표준 출력 저장

        # ✅ 실행 전, exec_globals 초기화 (이전 값 유지 방지)
        safe_locals = {}

        try:
            sys.stdout = captured_output  # 표준 출력 변경
            exec(code, exec_globals, safe_locals)  # **제한된 네임스페이스에서 실행**
            sys.stdout = original_stdout # 표준 출력을 원래대로 복원
            
            print(f"🔄 [_execute_code_with_capture] 코드 실행 결과 객체 : {safe_locals.keys()}")

            # 분석 결과 초기화
            results = None
            analytic_results = None
            
            # 전체 데이터 실행 시 분석 결과 추출
            if not is_sample:
                if "result_df" in safe_locals:
                    results = safe_locals["result_df"]
                elif "analytic_results" in safe_locals:
                    results = safe_locals["analytic_results"]
                
                # 결과 타입에 따른 표준화 처리
                if results is not None:
                    if isinstance(results, pd.DataFrame):
                        # DataFrame을 dictionary로 변환
                        analytic_results = results.to_dict('records') if not results.empty else {}
                    elif isinstance(results, dict):
                        # Dictionary는 그대로 사용
                        analytic_results = results
                    elif isinstance(results, list):
                        # List는 그대로 사용
                        analytic_results = results
                    else:
                        # 기타 타입은 dictionary로 변환
                        analytic_results = {"result": str(results)}
            
            # 출력 및 분석 결과 반환
            return captured_output.getvalue(), analytic_results
            
        except Exception as e:
            captured_output.write(f"Error: {str(e)}\n")  # 에러 메시지 출력
            sys.stdout = original_stdout
            raise e


    def map_to_eda_category(self, user_request):
        """
        사용자의 질문을 분석하여 적절한 EDA 단계를 선택
        - 1차적으로 사전 정의된 동의어 매핑을 사용
        - 2차적으로 LLM을 활용하여 의미를 추론
        """
        # 사전 정의된 동의어 매핑
        eda_synonyms = {
            "기본 정보 분석": ["기본 정보", "데이터 정보", "데이터셋 개요"],
            "기초 통계 분석": ["기초 통계", "통계 분석", "데이터 분포"],
            "결측치 처리": ["결측치", "Null 값", "누락된 값", "NaN", "빠진 데이터", "소실된 값"],
            "변수 간 관계 분석": ["변수 관계", "상관관계 분석", "특성 관계"],
            "이상치 탐지": ["이상치", "이상값", "Outlier", "비정상 데이터", "이상한 데이터"]
        }
        
        # 1️⃣ 사전 정의된 키워드 매칭
        selected_keywords = process.extract(user_request, sum(eda_synonyms.values(), []), limit=3)
        matched_categories = []
        for keyword, score in selected_keywords:
            if score > 80:
                for category, synonyms in eda_synonyms.items():
                    if keyword in synonyms:
                        matched_categories.append(category)
        
        # 2️⃣ 만약 유사어 매칭이 안되면 LLM을 활용하여 분석
        if not matched_categories:
            prompt = f"""
            사용자가 EDA 분석을 요청했습니다.  
            아래 문장에서 사용자가 원하는 분석 단계를 정확히 판별하세요.  

            EDA 분석 단계:
            1. 기본 정보 분석: 데이터 크기, 타입, 결측값, 중복 여부 확인
            2. 기초 통계 분석: 변수의 분포, 변동성, 정규성 검정
            3. 결측치 처리: 결측값 비율 확인, 보간 방법 적용 (예: Null 값, NaN, 누락된 값, 빠진 데이터)
            4. 변수 간 관계 분석: 상관관계, 다중공선성, 범주형 변수 관계 분석
            5. 이상치 탐지: 이상값 검출, 박스플롯 시각화 (예: Outlier, 비정상 데이터)

            예제 1:
            입력: "결측값 분석해줘"
            출력: "결측치 처리"

            예제 2:
            입력: "이상값 확인 부탁해"
            출력: "이상치 탐지"

            사용자의 요청이 다음과 같을 때, 가장 적절한 EDA 단계를 반환하세요.  
            요청: "{user_request}"  
            만약 적절한 분석 단계가 없다면 빈 값을 반환하세요.
            """
            
            llm_response = self.llm.invoke(prompt)  # LLM을 활용하여 의미 분석
            llm_response = llm_response.content
            print(f"📌 LLM 추론 결과: {llm_response}")
            
            # LLM이 추천하는 카테고리를 기존 매핑과 비교하여 판단
            for category in eda_synonyms.keys():
                if category in llm_response:
                    matched_categories.append(category)

            # 적절한 EDA 단계가 없을 경우 전체 EDA 프로세스를 실행
            if not matched_categories:
                # matched_categories = list(eda_synonyms.keys())
                matched_categories = ["전체"]
            print(f"🔍 매칭된 카테고리: {matched_categories}")
        
        return matched_categories
    
    def _calculate_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 계산하는 메소드
        
        Args:
            text (str): 토큰 수를 계산할 텍스트
            
        Returns:
            int: 토큰 수
        """
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"⚠️ 토큰 계산 중 오류 발생: {str(e)}")
            return 0
        

    def _extract_code_from_llm_response(self, response: str) -> str:
        """LLM 응답에서 코드 블록을 추출하는 메소드
        
        Args:
            response (str): LLM이 생성한 응답 텍스트
            
        Returns:
            str: 추출된 코드 (코드 블록이 없는 경우 원본 텍스트를 정리하여 반환)
        """
        try:
            if "```python" in response:
                return response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                return response.split("```")[1].strip()
            return response.strip()
        except Exception as e:
            print(f"⚠️ 코드 추출 중 오류 발생: {str(e)}")
            return response.strip()

    def route_after_full_execution(self, state: State) -> str:
        """전체 데이터 실행 후 다음 단계를 결정하는 라우터
        
        Returns:
            str: 다음 실행할 노드의 이름
        """
        print("➡️ [route_after_full_execution] 전체 데이터 실행 후 경로 결정")
        
        if state.get("validated_code"):  # validated_code가 있으면 성공
            print("➡️ [route_after_full_execution] 데이터 저장 단계로 진행")
            return "Save_Data"
        
        if self.retry_count >= MAX_RETRIES:
            print("⚠️ 전체 데이터 실행 3회 실패 → 프로세스 종료")
            return END
        
        print(f"⚠️ 전체 데이터 실행 실패 → 코드 재생성 필요 | 재시도 횟수: {self.retry_count}")
        return "Regenerate_Code"

    def route_after_report(state):
        """Report_Builder 이후 EDA 단계를 추가 실행할지 결정"""
        selected_categories = state.get("selected_categories", [])  # ✅ 기본값을 빈 리스트로 설정
        current_stage = state.get("eda_stage", 0)

        # ✅ 모든 EDA 단계를 완료했다면 END로 이동
        if current_stage >= len(selected_categories):
            return "END"

        return "Eda_Generate_Code"