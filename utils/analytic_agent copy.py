##########################################################################################
# Program Description
##########################################################################################
##########################################################################################
# 라이브러리
##########################################################################################
# Built-in Packages
import os, sys
import io
import pickle
import pandas as pd
import numpy as np
import traceback
import ast
import pkg_resources
from datetime import datetime
import threading
import time

# Thire Party Packages
from typing import TypedDict, List, Literal, Annotated, Dict, Union
from pydantic import BaseModel, Field
import tiktoken
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from fuzzywuzzy import process

# User Defined Packages
from prompt.prompts import *
from utils.vector_handler import load_vectorstore

##########################################################################################
# 상수 및 변수 선언부
##########################################################################################
# ✅ 한글 폰트 설정 (Windows 환경)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

VECTOR_DB_BASE_PATH = "./vectordb/analysis"
PROCESSED_DATA_PATH = "../output/stage1/processed_data_info.xlsx"
MAX_RETRIES = 3
TOKEN_LIMIT = 10000 # ✅ 토큰 제한 설정 (예: 5000 토큰 초과 시 차단)
RECURSION_LIMIT = 100

##########################################################################################
# 구현 코드
##########################################################################################
# ✅ AI 분석 에이전트 상태 정의(state에 적재된 데이터를 기반으로 이동)
class State(TypedDict):
    messages: List[BaseMessage]  # 🔹 사용자와 AI 간의 모든 대화 메시지 목록 (HumanMessage, AIMessage 등)
    mart_info: str  # 🔹 현재 활성화된 데이터프레임 (분석 대상)
    generated_code: str  # 🔹 초기 생성된 코드
    q_category: str  # 🔹 Supervisor가 판단한 질문 유형 (Analytics, General, Knowledge)
    content: str  # 🔹 일반/지식 질문에 대한 응답
    retry_count: int  # 🔹 코드 재생성 실패 시 재시도 횟수 (최대 3회)
    regenerated_code: str  # 🔹 재생성된 코드
    validated_code: str  # 전체 실행까지 통과한 코드
    analytic_result: Dict  # 🔹 전체 데이터를 실행하여 얻은 최종 결과 딕셔너리
    error_message: dict  # 🔹 코드 실행 중 발생한 오류 메시지 (있다면 재시도할 때 활용)
    data_id: str  # 🔹 분석 결과를 저장할 때 부여되는 고유 ID (파일 저장 시 활용)
    insights: str  # 🔹 LLM이 분석 결과를 바탕으로 생성한 주요 인사이트
    report: str  # 🔹 생성된 리포트
    chart_needed: bool  # 🔹 차트가 필요한지 여부 (True: 필요함, False: 불필요)
    chart_filename: str  # 🔹 생성된 차트의 파일 경로 (없으면 None)
    chart_error: int  # 🔹 차트 에러 메시지
    from_full_execution: bool  # 🔹 코드 재생성 시 초기 실행 여부
    from_token_limit: bool  # 🔹 토큰 제한 초과 시 초기 실행 여부
    request_summary: str  # 🔹 분석 요청을 한글로 요약한 내용
    installed_packages: Dict[str, str]  # 🔹 패키지 이름 및 버전 저장
    feedback: str  # 🔹 피드백 내용
    feedback_point: list  # 🔹 피드백 포인트
    start_from_analytics: bool  # 🔹 분석 시작 여부

class Feedback(BaseModel):
    feedback_point: list[str]  # 리스트 항목의 타입을 명시적으로 str로 지정

# 피드백 필요 여부를 위한 구조화된 출력 모델
class YesNo(BaseModel):
    decision: Literal["yes", "no"] = Field(description="여부 (yes 또는 no)")

# 질문 분류 라우터
class Question_Classifier(BaseModel):
    next: Literal["Analytics", "General", "Knowledge"]

class DataAnayticsAssistant:
    """Python DataFrame 기반 AI 분석 에이전트 (LangGraph 기반)"""
    ###############################################################################################
    # ✅ 초기화
    ###############################################################################################
    def __init__(self, openai_api_key: str):
        print("="*100)
        print("🔹 분석 에이전트 초기화")
        print("="*100)
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.0)
        self.active_marts = None
        self.mart_info = None
        self.retry_count = 0

        # 질의 관련 변수 초기화
        self.original_query = None  # 원본 사용자 질의
        self.context = None  # 이전 대화 기록 및 문맥 정보
        self.context_query = None  # 문맥이 포함된 최종 질의
        
        # 마트 정보 초기 로드
        try:
            self.mart_info_df = pd.read_excel(PROCESSED_DATA_PATH, sheet_name=None)
            print(f"🔹 현재 접근 가능 마트 목록: {list(self.mart_info_df.keys())}")
        except Exception as e:
            print(f"⚠️ 마트 정보 로드 실패: {e}")
            self.mart_info_df = {}
            
        self.build_graph()

    ###############################################################################################
    # ✅ 그래프 구성
    ###############################################################################################
    def build_graph(self):
        """LangGraph를 활용하여 분석 흐름 구성"""
        workflow = StateGraph(State)

        # 노드 선언
        workflow.add_node("Context", self.handle_context)
        workflow.add_node("Supervisor", self.supervisor)
        workflow.add_node("Analytics", self.handle_analytics)
        workflow.add_node("General", self.handle_general)
        workflow.add_node("Knowledge", self.handle_knowledge)
        workflow.add_node("Generate_Code", self.generate_python_code)
        workflow.add_node("Execute_Sample", self.execute_sample_code)
        workflow.add_node("Regenerate_Code", self.regenerate_code)
        workflow.add_node("Execute_Full", self.execute_full_data)
        workflow.add_node("Save_Data", self.save_data)
        workflow.add_node("Insight_Builder", self.generate_insights)
        workflow.add_node("Chart_Builder", self.generate_chart)
        workflow.add_node("Regenerate_Chart", self.regenerate_chart)
        workflow.add_node("Report_Builder", self.generate_report)
        workflow.add_node("After_Feedback", self.after_feedback)

        # 기본 흐름 정의
        workflow.add_edge(START, "Context")
        workflow.add_edge("Context", "Supervisor")
        workflow.add_conditional_edges(
            "Supervisor",
            lambda state: state["q_category"],  # Supervisor가 결정한 경로로 이동
            {
                "Analytics": "Analytics",
                "General": "General",
                "Knowledge": "Knowledge",
            }
        )
        
        workflow.add_edge("Analytics", "Generate_Code")
        workflow.add_conditional_edges(
            "Generate_Code",
            self.route_after_generate_code,
            {
                "Execute_Sample": "Execute_Sample",
                END: END
            }
        )
        workflow.add_conditional_edges( # ✅ 샘플 실행 후 조건부 라우팅 설정
            "Execute_Sample",
            self.route_after_sample,
            {
                "Execute_Full": "Execute_Full",
                "Regenerate_Code": "Regenerate_Code",
                END : END
            }
        )
        workflow.add_conditional_edges( # ✅ 코드 재생성 흐름
            "Regenerate_Code",
            self.route_after_regenerate,  # 새로운 라우터 함수 사용
            {
                "Execute_Sample": "Execute_Sample",
                "Execute_Full": "Execute_Full",
                END: END  # ✅ 3회 이상이면 종료
            }
        )
        workflow.add_conditional_edges( # ✅ 전체 데이터 실행 후 조건부 라우팅 설정
            "Execute_Full",
            self.route_after_full_execution,
            {
                "Save_Data": "Save_Data",
                "Regenerate_Code": "Regenerate_Code",
                END : END
            }
        )
        workflow.add_edge("Save_Data", "Insight_Builder")
        workflow.add_conditional_edges( # ✅ 인사이트 생성 후 조건부 라우팅 설정
            "Insight_Builder",
            self.route_after_insights,
            {
                "Chart_Builder": "Chart_Builder",
                "Report_Builder": "Report_Builder"
            }
        )
        workflow.add_conditional_edges( # ✅ 차트 생성 후 조건부 라우팅 설정
            "Chart_Builder",
            self.route_after_chart,
            {
                "Regenerate_Chart": "Regenerate_Chart",  # 실패 시 재생성
                "Report_Builder": "Report_Builder",  # 성공 또는 최대 재시도 초과
            }
        )
        workflow.add_conditional_edges( # ✅ 차트 재생성 후 조건부 라우팅 설정
            "Regenerate_Chart",
            self.route_after_chart,
            {
                "Regenerate_Chart": "Regenerate_Chart",  # 여전히 실패 시 다시 재생성
                "Report_Builder": "Report_Builder",  # 성공 또는 최대 재시도 초과
            }
        )
        workflow.add_conditional_edges(
            "Report_Builder",
            self.route_after_report,  # 새로운 라우터 함수 추가
            {
                "After_Feedback": "After_Feedback",
                END: END
            }
        )
        workflow.add_edge("After_Feedback", END)
        self.graph = workflow.compile()
        print("✅ 그래프 컴파일 완료")        
        
    ###############################################################################################
    # ✅ 실행
    ###############################################################################################
    def ask(self, query: str, context: list, start_from_analytics=False, feedback_point=None):
        """LangGraph 실행"""

        # 컨텍스트 저장
        self.context = context
        # print(f"🔍 컨텍스트:\n{self.context}")

        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content=query)],  # 원본 쿼리만 전달
        }

        # 개선 요청일 경우
        if start_from_analytics:

            # 개선 요청 사항을 원본 질문으로 설정
            self.original_query = feedback_point

            # 초기 상태에 질문 분류를 'Analytics'로 설정 및 플래그 추가
            initial_state.update({
                "q_category": "Analytics",
                "start_from_analytics": True 
            })

            # 문맥이 담긴 질의 생성
            self.context_query = f"""
# 개선 요청 사항
{self.original_query}

{self.context}
        """
        # 일반 질문일 경우
        else:
            self.original_query = query
        
        # 그래프 실행
        result = self.graph.invoke(initial_state, config={"recursion_limit": RECURSION_LIMIT})
        
        return result

    ###############################################################################################
    # ✅ 노드 구현
    ###############################################################################################
    #########################################################
    # ✅ Context Windows 노드
    # -> Context_Filter
    #########################################################
    def handle_context(self, state: State) -> Command:
        """컨텍스트를 정리하고, 현재 질문을 최우선으로 강조하는 개선된 노드"""
        print("🗃️ 컨텍스트 필터링 단계")

        # Analytics부터 시작하는 경우 Supervisor로 바로 이동(개선 요청)
        if state.get("start_from_analytics", False):
            print("🗃️ [handle_context] 개선 요청 처리이므로 바로 Supervisor로 이동")
            return Command(goto="Supervisor")
        
        # 기존 대화 이력이 없는 경우 원본 질의를 그대로 사용
        if not self.context:
            print("🗃️ [handle_context] 이전 대화 기록 없음 -> Supervisor로 이동")
            self.context_query = self.original_query  # 문맥이 없으면 원본 질의를 그대로 사용
            return Command(goto="Supervisor")
        # 기존 대화 이력이 있는 경우 컨텍스트 필터링
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", PROMPT_CONTEXT_FILTER),
                ("user", "##### 현재 질문\n{user_request}"),
                ("user", "##### 최근 대화 기록\n{context}"),
                ("user", "##### 정리된 문맥"),
            ])

            chain = prompt | self.llm
            context_str = "\n".join([f"\n사용자: {chat['query']}\n어시스턴트: {chat['response']}" for chat in self.context])
            
            # 컨텍스트 필터링
            filtered_context = chain.invoke({
                "user_request": self.original_query,
                "context": context_str
            }).content.strip()

            print(f"🗃️ [handle_context] 필터링 후 대화 기록 :\n{filtered_context}")
            
            # 🔹 최종 Context 구성 (현재 질문을 최상단으로)
            self.context_query = f"""
# 🤔 현재 질문 (최우선)
{self.original_query}

{filtered_context}
            """
            return Command(goto="Supervisor")

    #########################################################
    # ✅ Supervisor 노드
    # -> General / Knowledge / Analytics 
    #########################################################
    def supervisor(self, state: State) -> Command:
        """다음 단계를 결정하는 Supervisor"""
        print("="*100)  # 구분선 추가
        print("👨‍💼 Supervisor 단계:")

        # Request Summary 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_REQUEST_SUMMARY),
            ("user", "{user_request}")
        ])
        
        chain = prompt | self.llm
        request_summary = chain.invoke({
            "user_request": self.original_query # request summary는 원본 질의로 생성
        }).content.strip()
        
        print(f"👨‍💼 요약된 질의 내용: {request_summary}")
        
        # 문맥이 포함된 최종 질의 사용
        user_request = self.context_query
        
        # Analytics부터 시작하는 경우 바로 Analytics로 이동
        if state.get("start_from_analytics", False):
            print("👨‍💼 개선 요청 -> Analytics 단계로 바로 이동")
            return Command(
                update={
                    "q_category": 'Analytics', 
                    "request_summary": request_summary,
                }, 
                goto="Analytics"
            )

        # 질문 유형 결정
        prompt = ChatPromptTemplate.from_messages([
                ("system", PROMPT_SUPERVISOR),
                ("user", " user_request:\n{user_request}\n\n")
        ])
        chain = prompt | self.llm.with_structured_output(Question_Classifier)
        response = chain.invoke({"user_request": user_request})
        print(f"👨‍💼 다음 단계(Analytics or General or Knowledge): {response.next}")
        return Command(
            update={
                "q_category": response.next, 
                "request_summary": request_summary,
            }, 
            goto=response.next
        )
    
    #########################################################
    # ✅ General 노드
    #########################################################
    def handle_general(self, state: State) -> Command:
        """일반적인 질문을 처리하는 노드"""
        print("\n💬 [handle_general] 일반 질문 처리")
        user_request = self.original_query
        prompt = ChatPromptTemplate.from_messages([
                ("system", PROMPT_GENERAL),
                ("user", " user_request:\n{user_request}\n\n")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"user_request": user_request})
        print(f"💬 일반 응답: {response.content}")
        return Command(update={"content": response.content}, goto=END)

    #########################################################
    # ✅ Knowledge 노드
    #########################################################
    def handle_knowledge(self, state: State) -> Command:
        """지식 기반 응답을 처리하는 노드"""
        print("\n📚 [handle_knowledge] 지식 기반 질문 처리")
        
        user_request = self.context_query or self.original_query

        # 분석 어시스턴트 벡터스토어 로드(미리 문맥 등록이 필요)
        vectorstore = load_vectorstore('./vectordb/analysis')
        if vectorstore is None:
            print("⚠️ 벡터스토어 연결 실패: FAISS 인덱스를 찾을 수 없습니다. LLM으로만 응답합니다.")
            # 일반 LLM 응답 생성
            prompt = ChatPromptTemplate.from_messages([
                    ("system", PROMPT_GENERAL),
                    ("user", "{user_question}")
            ])
            chain = prompt | self.llm
            response = chain.invoke({"user_question": user_request})
            return Command(update={"content": response.content}, goto=END)

        # Retriever 생성
        retriever = vectorstore.as_retriever()

        # 사용자 질문 검색
        retrieved_docs = retriever.get_relevant_documents(user_request)

        if not retrieved_docs:
            response = "관련된 정보를 찾을 수 없습니다."
        else:
            # 검색된 문서 상위 3개를 컨텍스트로 활용
            context = "\n".join([doc.page_content for doc in retrieved_docs[:3]])
            prompt = ChatPromptTemplate.from_messages([
                    ("system", PROMPT_KNOWLEDGE),
                    ("user", "\n질문:\n{user_question}"),
                    ("user", "\ndocument:\n{context}")
            ])
            chain = prompt | self.llm
            response = chain.invoke({"user_question": user_request, "context": context})
        print(f"📖 지식 기반 응답: {response.content}")
        return Command(update={"content": response.content}, goto=END)
    
    #########################################################
    # ✅ Analytics 노드
    # -> Generate_Code
    #########################################################
    def handle_analytics(self, state: State) -> Command:
        """분석 요청을 처리하는 노드"""
        print("👨‍💼 [handle_analytics] 분석 요청 처리 시작")
        return Command(goto="Generate_Code")

    #########################################################
    # ✅ Generate_Code 노드
    # -> Execute_Sample / END
    #########################################################
    def generate_python_code(self, state)-> Command:
        """
        사용자의 요청을 기반으로 Python 코드 생성
        IF 활성화된 마트가 없음 -> END 노드로 이동
        ELSE 데이터프레임 정보 생성 및 코드 생성 -> Execute_Sample 노드로 이동
        """
        print("="*100)
        print("🤖 코드 생성 단계:")
        
        # 활성화된 마트가 있는지 확인
        if not self.active_marts:
            print("❌ 활성화된 마트가 없습니다. 먼저 마트를 활성화해주세요.")
            return Command(
                update={
                    "error_message": self._normalize_error_message("❌ 활성화된 마트가 없습니다. 먼저 마트를 활성화해주세요.")
                }, 
                goto=END
            )
        
        user_request = self.context_query
        
        # 마트 정보 가져오기 (중괄호 이스케이프 적용)
        mart_info = self._get_mart_info()

        # 기존 프롬프트 사용
        prompt_text = PROMPT_GENERATE_CODE.format(mart_info=mart_info)
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("user", "\nuser_request:\n{user_request}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({
            "user_request": user_request,
        })
            
        print(f"🤖 생성된 코드:\n{response.content}\n")
        return Command(update={
            "generated_code": response.content,
            "regenerated_code": None,  # 초기화
            "validated_code": None     # 초기화
        }, goto="Execute_Sample")
    
    #########################################################
    # ✅ Execute_Sample 노드
    # -> Execute_Full / Regenerate_Code / END
    #########################################################
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
        
        # 재생성된 코드가 있으면 그것을 사용, 없으면 초기 생성 코드 사용
        code_to_execute = self._extract_code_from_llm_response(
            state.get("regenerated_code") or state["generated_code"]
        )
            
        # ✅ 사용된 패키지 자동 추출
        used_packages = self._extract_imported_packages(code_to_execute)
        installed_versions = self._get_installed_versions(used_packages)

        print(f"🧪 사용된 패키지 목록: {used_packages} | 패키지 버전 정보: {installed_versions}")
        try:
            # 실행 환경에 샘플 데이터프레임 추가
            exec_globals = {}

            # 모든 데이터프레임을 exec_globals에 추가
            exec_globals.update(sample_marts)

            print(f"🔹 실행 환경에 추가된 데이터프레임 목록: {list(exec_globals.keys())}")

            # 추출된 코드 실행
            self._execute_code_with_capture(code_to_execute, exec_globals, is_sample=True)
            
            print(f"✅ 샘플 코드 실행 성공")
            self.retry_count = 0  # 성공 시 카운터 초기화
            return Command(update={
                "error_message": None,
                "installed_packages": installed_versions
            }, goto="Execute_Full")

        except Exception as e:
            print(f"❌ 샘플 코드 실행 실패")
            print(traceback.format_exc())
            error_details = {
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "traceback": traceback.format_exc(),
                "installed_packages": installed_versions
            }
            self.retry_count += 1
            if self.retry_count >= MAX_RETRIES:
                print("⚠️ 샘플 코드 실행 3회 실패 → 프로세스 종료")
                return Command(update={
                    "error_message": self._normalize_error_message(error_details), 
                    "installed_packages": installed_versions
                }, goto=END)
            return Command(update={
                "error_message": self._normalize_error_message(error_details), 
                "installed_packages": installed_versions
            }, goto="Regenerate_Code")

    #########################################################
    # ✅ Regenerate_Code 노드
    # -> Execute_Full / Execute_Sample / END
    #########################################################
    def regenerate_code(self, state):
        """코드 실행 오류 발생 시 LLM을 활용하여 코드 재생성"""
        from_full_execution = state.get("from_full_execution", False)  # 플래그 확인

        if self.retry_count >= MAX_RETRIES:  # ✅ 3회 초과 시 종료
            return Command(goto=END)
        
        print("="*100)  # 구분선 추가
        print("⚒️ 코드 재생성 단계")
        user_request = self.context_query or self.original_query
        error_message = state["error_message"]
        original_code = state["generated_code"]
        installed_packages = state.get("installed_packages", {})  # 설치된 패키지 정보 가져오기

        # 토큰 초과 시 코드 재생성
        if state.get("from_token_limit", False):
            print(f"⚒️ 토큰 초과 시의 코드 재생성 진행")
            prompt = ChatPromptTemplate.from_messages([
                    ("system", PROMPT_REGENERATE_CODE_WHEN_TOKEN_OVER),
                    ("user", "\nuser_request:\n{user_request}"),
            ])
        # 일반 코드 재생성
        else:
            prompt = ChatPromptTemplate.from_messages([
                        ("system", PROMPT_REGENERATE_CODE),
                        ("user", "\nuser_request:\n{user_request}"),
                        ("user", "\noriginal_code:\n{original_code}"),
                        ("user", "\nerror_message:\n{error_message}"),
                        ("user", "\ninstalled_packages:\n{installed_packages}")
                ])
        
        chain = prompt | self.llm
        
        # 코드 재생성
        response = chain.invoke({
            "user_request": user_request,
            "original_code": original_code,
            "error_message": error_message,       
            "installed_packages": installed_packages  # 패키지 정보 전달
        })
        print(f"⚒️ 재생성된 코드:\n{response.content}\n")
        next_step = "Execute_Full" if from_full_execution else "Execute_Sample"
        
        return Command(update={
            "regenerated_code": response.content,  # 재생성된 코드 저장
            "validated_code": None,  # validated_code 초기화
            "from_full_execution": from_full_execution
        }, goto=next_step)

    #########################################################
    # ✅ Execute_Full 노드
    # -> Save_Data / Regenerate_Code 
    #########################################################
    def execute_full_data(self, state):
        """전체 데이터로 Python 코드 실행 (300초 제한)"""
        print("="*100)  # 구분선 추가
        print("🔄 전체 데이터 실행 단계 (최대 300초 제한)")

        # 전체 데이터프레임 설정
        full_marts = self.active_marts  # 전체 데이터프레임 사용

        # 실행 환경에 전체 데이터프레임 추가
        exec_globals = {}

        # 모든 데이터프레임을 exec_globals에 추가
        exec_globals.update(full_marts)

        print(f"🔄 전체 데이터 실행 환경에 추가된 데이터프레임 목록: {list(exec_globals.keys())}")

        # LLM 생성 코드에서 ```python 블록 제거
        code_to_execute = self._extract_code_from_llm_response(
            state.get("regenerated_code") or state["generated_code"]
        )
        # ✅ 사용된 패키지 자동 추출
        used_packages = self._extract_imported_packages(code_to_execute)
        installed_versions = self._get_installed_versions(used_packages)

        print(f"🔄 사용된 패키지 목록: {used_packages} | 패키지 버전 정보: {installed_versions}")

        try:
            # 타임아웃 처리를 위한 래퍼 함수 (스레드 기반 타임아웃 구현)
            def execute_with_timeout(code, globals_dict, timeout=50):
                
                result = {"output": None, "analytic_result": None, "error": None, "timeout": False}
                
                def target():
                    try:
                        output, analytic_result = self._execute_code_with_capture(code, globals_dict, is_sample=False)
                        result["output"] = output
                        result["analytic_result"] = analytic_result
                    except Exception as e:
                        result["error"] = e
                
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout)
                
                if thread.is_alive():
                    # 타임아웃 발생
                    result["timeout"] = True
                    print(f"⚠️ 실행 시간이 {timeout}초를 초과하여 중단되었습니다.")
                    # 스레드는 daemon=True로 설정되어 있어 메인 스레드가 종료되면 자동으로 종료됨
                
                return result
            
            # 코드 실행 (타임아웃 300초)
            execution_result = execute_with_timeout(code_to_execute, exec_globals, timeout=50)
            
            # 타임아웃 발생
            if execution_result["timeout"]:
                return Command(update={
                    "analytic_result": None,  # 결과 미저장
                    "validated_code": state.get("regenerated_code") or state["generated_code"],  # 실행된 코드 유지
                    "error_message": self._normalize_error_message("⚠️ 실행 시간이 300초를 초과하여 중단되었습니다."),
                    "from_full_execution": True,  # 전체 실행 플래그 추가
                    "installed_packages": installed_versions
                }, goto="Save_Data")  # 다음 단계로 이동
            
            # 실행 중 오류 발생
            if execution_result["error"]:
                e = execution_result["error"]
                print(f"❌ 전체 데이터 실행 실패\n {code_to_execute}")
                print(f"에러 타입: {type(e).__name__}")
                print(f"에러 메시지: {str(e)}")
                print(f"에러 발생 위치:")
                print(traceback.format_exc())
                error_details = {
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "traceback": traceback.format_exc(),
                    "installed_packages": installed_versions
                }
                self.retry_count += 1
                return Command(update={
                    "error_message": self._normalize_error_message(error_details),
                    "from_full_execution": True,  # 플래그 추가
                    "installed_packages": installed_versions
                }, goto="Regenerate_Code")
            
            # 정상 실행 완료
            output = execution_result["output"]
            analytic_result = execution_result["analytic_result"]
            
            token_count = self._calculate_tokens(str(analytic_result))
            
            print(f"🔄 결과 데이터 토큰 수: {token_count}")
            
            if token_count > TOKEN_LIMIT:
                print(f"⚠️ 토큰 수 초과: {token_count} > {TOKEN_LIMIT}")
                self.retry_count += 1
                return Command(update={
                    "error_message": self._normalize_error_message(f"결과 데이터 analytic_result의 적정 토큰 수를 초과하였습니다. analytic_result에 Raw 데이터 혹은 불필요한 반복 적재를 피해주세요: {token_count} > {TOKEN_LIMIT}"),
                    "from_full_execution": True,  # 플래그 추가
                    "from_token_limit": True,
                    "installed_packages": installed_versions
                }, goto="Regenerate_Code")
            
            print(f"🔄 전체 데이터 실행 성공")
            # print(f'🔄 analytic_result\n {analytic_result}')

            # 분석 결과가 있는 경우
            if analytic_result is not None:
                unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
                # 전체 실행 성공 시 validated_code 설정
                current_code = state.get("regenerated_code") or state["generated_code"]
                return Command(update={
                    "analytic_result": analytic_result,
                    "data_id": unique_id,
                    "validated_code": current_code,  # 성공한 코드를 validated_code로 저장
                    "installed_packages": installed_versions
                }, goto="Save_Data")
            # 분석 결과가 없는 경우
            else:
                print("⚠️ 분석 결과가 없습니다.")
                self.retry_count += 1
                return Command(update={
                    "error_message": self._normalize_error_message("분석 결과가 없습니다."),
                    "from_full_execution": True,  # 플래그 추가
                    "installed_packages": installed_versions
                }, goto="Regenerate_Code")

        except Exception as e:
            print(f"❌ 전체 데이터 실행 실패\n {code_to_execute}")
            print(f"에러 타입: {type(e).__name__}")
            print(f"에러 메시지: {str(e)}")
            print(f"에러 발생 위치:")
            print(traceback.format_exc())
            error_details = {
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "traceback": traceback.format_exc(),
                "installed_packages": installed_versions
            }
            self.retry_count += 1
            return Command(update={
                "error_message": self._normalize_error_message(error_details),
                "from_full_execution": True,  # 플래그 추가
                "installed_packages": installed_versions
            }, goto="Regenerate_Code")

    #########################################################
    # ✅ Save_Data 노드
    # -> Insight_Builder
    #########################################################
    def save_data(self, state):
        """처리된 데이터를 저장 (ID 부여)"""
        print("="*100)  # 구분선 추가
        print("📂 처리 데이터 저장 단계")
        # data_id가 없는 경우 생성
        data_id = state.get("data_id", datetime.now().strftime("%Y%m%d%H%M%S"))
        analytic_result = state["analytic_result"]
        # 분석 결과와 실행 출력을 함께 저장
        save_data = {
            'analytic_result': analytic_result,
        }

        # 저장 디렉토리 확인 및 생성
        os.makedirs("../output", exist_ok=True)
        with open(f"../output/data_{data_id}.pkl", 'wb') as f:
            pickle.dump(save_data, f)

        print(f"📂 처리된 데이터 저장 경로: ../output/data_{data_id}.pkl")
        return Command(update={"data_id": data_id}, goto="Insight_Builder")
    
    #########################################################
    # ✅ Insight_Builder 노드
    # -> Chart_Builder / Report_Builder
    #########################################################
    def generate_insights(self, state):
        """저장된 데이터에서 자동 인사이트 도출 및 차트 필요 여부 결정"""
        print("="*100)  # 구분선 추가
        print("💡 인사이트 도출 단계")
        dict_result = state["analytic_result"]
        user_question = self.original_query

        # ✅ 집계 데이터면 전체 데이터 전달
        string_of_result = str(dict_result)

        ##############################
        # 1. 인사이트 생성
        ##############################
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_INSIGHT_BUILDER),
            ("user", "user_question:\n{user_question}\n\n"),
            ("user", "analytic_result:\n{analytic_result}\n\n")
        ])

        chain = prompt | self.llm
        insight_response = chain.invoke({
            "user_question": user_question,
            "analytic_result": string_of_result
        })

        print(f"💡 생성된 인사이트\n{insight_response.content}")
        
        ##############################
        # 2. 차트 필요 여부 결정
        ##############################
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_CHART_NEEDED),
            ("user", "user_question:\n{user_question}\n\n"),
            ("user", "analytic_result:\n{analytic_result}\n\n"),
            ("user", "insights:\n{insights}\n\n")
        ])
        
        # 차트 활용 여부 'yes' 또는 'no' 반환
        chart_decision_messages = prompt | self.llm.with_structured_output(YesNo)
        chart_needed = chart_decision_messages.invoke({
            "user_question": user_question,
            "analytic_result": string_of_result,
            "insights": insight_response.content
        }).decision
        print(f"💡 차트 필요 여부 (yes/no): {chart_needed}")
        
        # 차트 필요 여부에 따라 다음 단계 결정
        next_step = "Chart_Builder" if chart_needed == "yes" else "Report_Builder"
        
        return Command(update={
            "insights": insight_response.content,
            "chart_needed": chart_needed == "yes"
        }, goto=next_step)  # Supervisor 대신 적절한 다음 단계로 이동
        
    #########################################################
    # ✅ Chart_Builder 노드
    # -> Report_Builder / Regenerate_Chart
    #########################################################
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
                'sns': sns,
                'pd': pd,
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
            error_details = {
                "error_msg": str(e),
                "executed_code": extracted_code,
                "traceback": traceback.format_exc()
            }
            # ❌ 실패 시 Regenerate_Chart로 이동
            self.retry_count += 1
            return Command(
                update={
                    "chart_filename": None,
                    "chart_error": self._normalize_error_message(error_details)
                },
                goto="Regenerate_Chart"
            )

    #########################################################
    # ✅ Regenerate_Chart 노드
    # -> Report_Builder / Regenerate_Chart
    #########################################################
    def regenerate_chart(self, state):
        """차트 생성 실패 시 에러를 기반으로 차트 재생성"""
        print("="*100)
        print("🔄 차트 재생성 단계")
        
        dict_result = state["analytic_result"]
        string_of_result = str(dict_result)
        insights = state.get('insights', '인사이트 없음')
        previous_error = state.get("chart_error", {})

        if self.retry_count >= MAX_RETRIES:
            print("⚠️ 차트 재생성 3회 실패. 차트 없이 리포트 생성으로 이동합니다.")
            self.retry_count = 0  # 카운터 초기화
            return Command(update={
                "chart_filename": None,
                "chart_error": self._normalize_error_message("차트 재생성 3회 실패")
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
            "insights": insights
        }).content

        extracted_code = self._extract_code_from_llm_response(chart_code)

        # ✅ 유효한 Python 코드 블록이 없는 경우 재시도
        if not extracted_code:
            print("📊 [regenerate_chart] 유효한 Python 코드 블록이 없습니다. 재시도합니다.")
            self.retry_count += 1
            return Command(update={
                "retry_count": self.retry_count + 1,
                "chart_error": self._normalize_error_message("유효한 Python 코드 블록이 없습니다")
            }, goto="Regenerate_Chart")

        # ✅ 차트 저장 디렉토리 생성
        os.makedirs("../img", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"../img/chart_{timestamp}.png"

        # 🔹 `plt.show()` 제거
        extracted_code = extracted_code.replace("plt.show()", "").strip()

        # 🔹 `plt.savefig()` 추가
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
            error_details = {
                "error_msg": str(e),
                "executed_code": extracted_code,
                "traceback": traceback.format_exc()
            }
            self.retry_count += 1
            return Command(update={
                "chart_filename": None,
                "chart_error": self._normalize_error_message(error_details)
            }, goto="Regenerate_Chart")

    #########################################################
    # ✅ Report_Builder 노드
    # -> After_Feedback
    #########################################################
    def generate_report(self, state):
        """최종 보고서 생성"""
        print("="*100)  # 구분선 추가
        print("📑 보고서 생성 단계")
        dict_result = state["analytic_result"]
        string_of_result = str(dict_result)
        insights = state.get('insights', '인사이트 없음')
        user_request = self.original_query
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_REPORT_GENERATOR),
            ("user", "1. 분석 결과 데이터\n{analytic_result}\n\n"),
            ("user", "2. 사용자 요청\n{user_request}\n\n"),
            ("user", "3. 도출된 인사이트\n{insights}\n\n"),
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "user_request": user_request,
            "analytic_result": string_of_result,
            "insights": insights,
        })
        print("✅ 보고서 생성 완료")
        print(f"{response.content}")

        # start_from_analytics가 True이면 바로 END로 이동
        if state.get("start_from_analytics", False):
            return Command(update={
                "report": response.content, 
                "error_message": None # 오류 메시지 초기화  
            }, goto=END)
        else :
            # 일반 분석인 경우 After_Feedback으로 이동
            return Command(update={
                "report": response.content, 
                "error_message": None # 오류 메시지 초기화  
            }, goto='After_Feedback')
    
    #########################################################
    # ✅ After_Feedback 노드
    # -> END
    #########################################################
    def after_feedback(self, state):
        print("="*100)  # 구분선 추가
        print("💡 피드백 단계")
        dict_result = state["analytic_result"]
        string_of_result = str(dict_result)
        user_question = self.original_query
        validated_code = state["validated_code"]
        ############################################################
        # 피드백 필요 여부 결정
        ############################################################
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_FEEDBACK_NEEDED),
            ("user", "user_question:\n{user_question}\n\n"),
            ("user", "analysis_result:\n{analysis_result}\n\n"),
            ("user", "validated_code:\n{validated_code}\n\n")
        ])
        
        feedback_decision = self.llm.with_structured_output(YesNo)
        feedback_needed = (prompt | feedback_decision).invoke({
            "user_question": user_question,
            "analysis_result": string_of_result,
            "validated_code": validated_code
        }).decision

        print(f"💡 피드백 필요 여부: {feedback_needed}")
        
        # 피드백 필요한 경우 피드백 처리
        if feedback_needed == 'yes':            
            prompt = ChatPromptTemplate.from_messages([
                ("system", PROMPT_FEEDBACK_PROCESS),
                ("user", "user_question:\n{user_question}\n\n"),
                ("user", "analysis_result:\n{analysis_result}\n\n"),
                ("user", "validated_code:\n{validated_code}\n\n")
            ])
            
            feedback_analysis_messages = prompt | self.llm
            feedback_analysis = feedback_analysis_messages.invoke({
                "user_question": user_question,
                "analysis_result": string_of_result,
                "validated_code": validated_code,
            }).content.strip().lower()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", PROMPT_FEEDBACK_POINT),
                ("user", "feedback_analysis:\n{feedback_analysis}\n\n")
            ])
            
            feedback_point_messages = prompt | self.llm.with_structured_output(Feedback)
            feedback_point = feedback_point_messages.invoke({
                "user_question": user_question,
                "feedback_analysis": feedback_analysis
            }).feedback_point

            print(f"💡 피드백 내용: {feedback_analysis}")
            print("✅ 피드백 완료")

            return Command(update={"feedback": feedback_analysis, 'feedback_point': feedback_point}, goto=END)
        else :
            return Command(goto=END)
    

    ##################################################################################################################
    # 라우터 모음
    ##################################################################################################################
    def route_after_generate_code(self, state: State):
        """코드 생성 후 다음 단계를 결정하는 라우터"""
        print("➡️ [route_after_generate_code] 코드 생성 후 경로 결정")

        if state.get("generated_code"):
            print("➡️ [route_after_generate_code] 샘플 실행 진행")
            return "Execute_Sample"
        else:
            print("➡️ [route_after_generate_code] 마트 활성화 필요 -> [프로세스 종료]")
            return END


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
                self.retry_count = 0
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

    def route_after_report(self, state: State) -> str:
        """리포트 생성 후 다음 단계를 결정하는 라우터
        
        Returns:
            str: 다음 실행할 노드의 이름 (After_Feedback 또는 END)
        """
        print("="*100)  # 구분선 추가
        print("➡️ [route_after_report] 리포트 생성 후 경로 결정")
        
        # start_from_analytics가 True이면 바로 END로 이동
        if state.get("start_from_analytics", False):
            print("➡️ [route_after_report] 개선 요청 처리이므로 바로 종료")
            return END
        
        print("➡️ [route_after_report] 피드백 단계로 진행")
        return "After_Feedback"

    ##################################################################################################################
    # 함수 모음
    ##################################################################################################################
    def set_active_mart(self, data_mart: Union[pd.DataFrame, Dict[str, pd.DataFrame]], mart_name: Union[str, List[str], None] = None) -> None:
        """분석할 데이터프레임과 마트 정보를 설정"""
        if isinstance(data_mart, pd.DataFrame):
            # 단일 데이터프레임 설정
            mart_key = mart_name if mart_name else "default_mart"
            self.active_marts = {mart_key: data_mart}
        elif isinstance(data_mart, dict):
            # 다중 데이터프레임 설정
            self.active_marts = data_mart
        else:
            raise TypeError("입력된 데이터가 pandas DataFrame 또는 DataFrame 딕셔너리가 아닙니다.")

        # 마트 정보 설정 (엑셀 파일의 sheet에서 가져옴)
        mart_info_list = []
        for mart_key in self.active_marts.keys():
            if mart_key in self.mart_info_df:
                mart_info_list.append(f"## {mart_key} 마트 정보\n{self.mart_info_df[mart_key].to_markdown()}")
        
        self.mart_info = "\n\n".join(mart_info_list) if mart_info_list else None

        # 데이터프레임 개수 및 정보 출력
        print(f"🔹 {len(self.active_marts)}개의 데이터프레임이 성공적으로 설정되었습니다.")
        for name, df in self.active_marts.items():
            print(f"🔹 데이터마트 이름: {name}")
            print(f"🔹 데이터 크기: {df.shape[0]}행 x {df.shape[1]}열")
            if self.mart_info and name in self.mart_info_df:
                print(f"🔹 마트 정보 로드됨")
    
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
            analytic_result = None
            
            # 전체 데이터 실행 시 분석 결과 추출
            if not is_sample:
                if "result_df" in safe_locals:
                    results = safe_locals["result_df"]
                elif "analytic_result" in safe_locals:
                    results = safe_locals["analytic_result"]
                
                # 결과 타입에 따른 표준화 처리
                if results is not None:
                    if isinstance(results, pd.DataFrame):
                        # DataFrame을 dictionary로 변환
                        analytic_result = results.to_dict('records') if not results.empty else {}
                    elif isinstance(results, dict):
                        # Dictionary는 그대로 사용
                        analytic_result = results
                    elif isinstance(results, list):
                        # List는 그대로 사용
                        analytic_result = results
                    else:
                        # 기타 타입은 dictionary로 변환
                        analytic_result = {"result": str(results)}
            
            # 출력 및 분석 결과 반환
            return captured_output.getvalue(), analytic_result
            
        except Exception as e:
            captured_output.write(f"Error: {str(e)}\n")  # 에러 메시지 출력
            sys.stdout = original_stdout
            raise e

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
        
    def _extract_code_from_llm_response(self, code_block: str) -> str:
        """LLM 응답에서 코드 블록을 추출하는 메소드
        
        Args:
            response (str): LLM이 생성한 응답 텍스트
            
        Returns:
            str: 추출된 코드 (코드 블록이 없는 경우 원본 텍스트를 정리하여 반환)
        """
        try:
            if "```python" in code_block:
                return code_block.split("```python")[1].split("```")[0].strip()
            elif "```" in code_block:
                return code_block.split("```")[1].strip()
            return code_block.strip()
        except Exception as e:
            print(f"⚠️ 코드 추출 중 오류 발생: {str(e)}")
            return code_block.strip()

   
    def _get_mart_info(self) -> str:
        """데이터프레임의 마트 정보를 생성하는 메서드
        
        Returns:
            str: 마트 정보 문자열 (마트가 없는 경우 기본 메시지 반환)
        """
        mart_info = ""
        if hasattr(self, 'active_marts') and self.active_marts:
            for mart_name in self.active_marts.keys():
                if mart_name in self.mart_info_df:
                    mart_info += f"\n- 데이터프레임 : {mart_name}의 컬럼 및 인스턴스 정보 ##\n"
                    mart_info += self.mart_info_df[mart_name].to_markdown().replace("{", "{{").replace("}", "}}")  # 이스케이프 적용
                    mart_info += "\n"
                else:
                    mart_info += f"\n## {mart_name} 마트 정보 없음 ##\n"
        else:
            mart_info = "데이터프레임 정보가 없습니다."
        
        return mart_info


    def _extract_imported_packages(self, code):
        """LLM이 생성한 코드에서 import된 패키지 이름만 추출"""
        tree = ast.parse(code)
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])  # 최상위 패키지만 추출
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module.split('.')[0])  # from X import Y 형식 처리

        return list(imports)

    def _get_installed_versions(self,used_packages):
        """사용된 패키지의 버전만 가져오기"""
        return {
            pkg: pkg_resources.get_distribution(pkg).version
            for pkg in used_packages if pkg in [p.key for p in pkg_resources.working_set]
        }
        
    def _normalize_error_message(self, error_message: Union[str, Dict]) -> Dict:
        """error_message를 표준화된 딕셔너리 형태로 변환하는 유틸리티 메서드"""
        if isinstance(error_message, dict):
            return error_message
    
    # 문자열인 경우 표준 형식으로 변환
        return {
                "error_type": "GeneralError",
                "error_msg": str(error_message),
                "traceback": "",
                "installed_packages": {}
            }