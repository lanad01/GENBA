##########################################################################################
# Program Description
##########################################################################################
# 1. 프로그램 설명
##########################################################################################

##########################################################################################
# 라이브러리
##########################################################################################
import os, sys
import io
import pickle
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from typing import TypedDict, List, Literal, Annotated, Dict, Union
from pydantic import BaseModel
import tiktoken
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from fuzzywuzzy import process

from prompt.prompts import *
from common_txt import logo
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
eda_prompt_mapping = {
    "기본 정보 분석": PROMPT_EDA_BASIC_INFO,
    "기초 통계 분석": PROMPT_EDA_STATISTICAL_ANALYSIS,
    "결측치 처리": PROMPT_EDA_MISSING_VALUE_HANDLING,
    "변수 간 관계 분석": PROMPT_EDA_FEATURE_RELATIONSHIP,
    "이상치 탐지": PROMPT_EDA_OUTLIER_DETECTION
}

##########################################################################################
# 구현 코드
##########################################################################################
# ✅ AI 분석 에이전트 상태 정의(state에 적재된 데이터를 기반으로 이동)
class State(TypedDict):
    messages: List[HumanMessage]  # 🔹 사용자와 AI 간의 대화 메시지 목록    
    context_history: List[Dict]  # 이전 대화 기록
    filtered_context: str  # 필터링된 컨텍스트
    final_query: str  # 최종 처리된 질의
    mart_info: str  # 🔹 현재 활성화된 데이터프레임 (분석 대상)
    generated_code: str  # 🔹 초기 생성된 코드
    q_category: str  # 🔹 Supervisor가 판단한 질문 유형 (Analytics, General, Knowledge)
    general_response: str  # 🔹 General 질문에 대한 응답
    knowledge_response: str  # 🔹 Knowledge 질문에 대한 응답
    retry_count: int  # 🔹 코드 재생성 실패 시 재시도 횟수 (최대 3회)
    regenerated_code: str  # 🔹 재생성된 코드
    validated_code: str  # 전체 실행까지 통과한 코드
    analytic_result: Dict  # 🔹 전체 데이터를 실행하여 얻은 최종 결과 딕셔너리
    execution_output: str  # 🔹 코드 실행 중 생성된 출력 텍스트
    error_message: str  # 🔹 코드 실행 중 발생한 오류 메시지 (있다면 재시도할 때 활용)
    data_id: str  # 🔹 분석 결과를 저장할 때 부여되는 고유 ID (파일 저장 시 활용)
    insights: str  # 🔹 LLM이 분석 결과를 바탕으로 생성한 주요 인사이트
    report: str  # 🔹 생성된 리포트
    chart_needed: bool  # 🔹 차트가 필요한지 여부 (True: 필요함, False: 불필요)
    chart_filename: str  # 🔹 생성된 차트의 파일 경로 (없으면 None)
    chart_error: int  # 🔹 차트 생성 횟수 카운터
    from_full_execution: bool  # 🔹 코드 재생성 시 초기 실행 여부
    from_token_limit: bool  # 🔹 토큰 제한 초과 시 초기 실행 여부
    request_summary: str  # 🔹 분석 요청을 한글로 요약한 내용
    analysis_type: str  # 🔹 분석 유형 (EDA, ML, General)
    eda_question: str  # 🔹 EDA 코드 생성 결과
    eda_stage: int  # 🔹 EDA 단계 카운터

# ✅ 경로 결정용 라우터
class Router(BaseModel):
    next: Literal["Analytics", "General", "Knowledge", "Generate_Code", "Execute_Sample", "Regenerate_Code", "Execute_Full", 
                  "Save_Data", "Insight_Builder", "Chart_Builder", "Regenerate_Chart", "Report_Builder", "__end__"]

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

        # 질의 분류(문맥 + 사용자질의 or 사용자 질의)
        self.original_query = None
        self.context = None
        
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
        workflow.add_node("Check_Analysis_Question", self.classify_analysis_question)  # ✅ 추가
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
        
         # ✅ Analytics → Check_EDA_Question 추가
        workflow.add_edge("Analytics", "Check_Analysis_Question")

        # ✅ Check_Analysis_Question → 분석 코드 생성 노드 조건부 라우팅 설정
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

        # ✅ 코드 생성 노드 조건부 라우팅 설정
        # workflow.add_edge("Eda_Generate_Code", "Execute_Sample")
        workflow.add_conditional_edges(
            "Eda_Generate_Code",
            self.route_after_generate_code,
            {
                "Execute_Sample": "Execute_Sample",
                END : END,
            }
        )
        # workflow.add_edge("ML_Generate_Code", "Execute_Sample")
        workflow.add_conditional_edges(
            "ML_Generate_Code",
            self.route_after_generate_code,
            {
                "Execute_Sample": "Execute_Sample",
                END : END,
            }
        )
        workflow.add_conditional_edges(
            "Generate_Code",
            self.route_after_generate_code,
            {
                "Execute_Sample": "Execute_Sample",
                END : END,
            }
        )

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
                END: END  # ✅ 3회 이상이면 종료
            }
        )

        # ✅ 전체 데이터 실행 후 조건부 라우팅 설정
        workflow.add_conditional_edges(
            "Execute_Full",
            self.route_after_full_execution,
            {
                "Save_Data": "Save_Data",
                "Regenerate_Code": "Regenerate_Code",
                END : END
            }
        )

        # ✅ 데이터 저장 후 인사이트 생성 노드로 이동
        workflow.add_edge("Save_Data", "Insight_Builder")

        # ✅ 인사이트 생성 후 조건부 라우팅 설정
        workflow.add_conditional_edges(
            "Insight_Builder",
            self.route_after_insights,
            {
                "Chart_Builder": "Chart_Builder",
                "Report_Builder": "Report_Builder"
            }
        )

        # ✅ 차트 생성 후 조건부 라우팅 설정
        workflow.add_conditional_edges(
            "Chart_Builder",
            self.route_after_chart,
            {
                "Regenerate_Chart": "Regenerate_Chart",  # 실패 시 재생성
                "Report_Builder": "Report_Builder",  # 성공 또는 최대 재시도 초과
            }
        )
        
        # ✅ 차트 재생성 후 조건부 라우팅 설정
        workflow.add_conditional_edges(
            "Regenerate_Chart",
            self.route_after_chart,
            {
                "Regenerate_Chart": "Regenerate_Chart",  # 여전히 실패 시 다시 재생성
                "Report_Builder": "Report_Builder",  # 성공 또는 최대 재시도 초과
            }
        )

        # ✅ 리포트 생성 후 종료
        workflow.add_conditional_edges(
            "Report_Builder",
            self.route_after_report,  # ✅ 별도 함수에서 state를 받아 처리하도록 변경
            {
                "Eda_Generate_Code": "Eda_Generate_Code",
                END : END,
            }
        )
        self.graph = workflow.compile()
        print("✅ 그래프 컴파일 완료")        
        
    ###############################################################################################
    # ✅ 실행
    ###############################################################################################
    def ask(self, query: str, context: list):
        """LangGraph 실행"""
        # 쿼리 상태 저장
        self.original_query = query
        self.context = context
        
        return self.graph.invoke({
            "messages": [HumanMessage(content=query)],  # 원본 쿼리만 전달
        }, config={"recursion_limit": RECURSION_LIMIT})

    ###############################################################################################
    # ✅ 노드 구현
    ###############################################################################################
    #########################################################
    # ✅ Context Windows 노드
    # -> Context_Filter
    #########################################################
    def handle_context(self, state: State) -> Command:
        """관련 있는 대화만 필터링하는 노드"""
        print("🔍 컨텍스트 필터링 단계")
        
        user_request = self.original_query
        context = self.context
        
        if not context:
            print("🔍 이전 대화 기록 없음")
            self.context_query = user_request  # context가 없으면 원본 쿼리만 사용
            return Command(
                update={"filtered_context": None},
                goto="Supervisor"
            )
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 AI 비서입니다. 
사용자의 최근 대화 기록 중, 현재 질문과 관련 있는 대화만 남기고 나머지는 제거하세요.
- 단, 연관된 질문일 경우 반드시 코드(`validated_code`) 및 분석 결과(`analytic_results`)를 함께 유지합니다.
            """),
            ("user", "### 현재 질문\n{user_request}"),
            ("user", "### 최근 대화 기록\n{context}"),
            ("user", "### 필터링된 문맥 (관련 있는 문맥만 유지)"),
        ])

        chain = prompt | self.llm
        filtered_context = chain.invoke({
            "user_request": user_request,
            "context": "\n".join([f"\n사용자: {chat['query']}\n어시스턴트: {chat['response']}" for chat in self.context])
        }).content.strip()

        print(f"🔍 필터링 후 대화 :\n{filtered_context}")
        
        # context_query 설정 (필터링된 컨텍스트가 있는 경우 포함)
        if filtered_context:
                self.context_query = f"""
# 📝 이전 대화 내역
{filtered_context}

# 🤔 현재 질문
{self.original_query}
        """
        else:
            self.context_query = self.original_query
        
        return Command(
            update={"filtered_context": filtered_context},
            goto="Supervisor"
        )

    #########################################################
    # ✅ Supervisor 노드
    # -> General / Knowledge / Analytics 
    #########################################################
    def supervisor(self, state: State) -> Command:
        """다음 단계를 결정하는 Supervisor"""
        print("="*100)  # 구분선 추가
        print("👨‍💼 Supervisor 단계:")
        user_request = self.context_query or self.original_query # 문맥 + 질의

        # Request Summary 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_REQUEST_SUMMARY),
            ("user", "{user_request}")
        ])
        
        chain = prompt | self.llm
        request_summary = chain.invoke({
            "user_request": user_request
        }).content.strip()
        
        print(f"👨‍💼 요약된 질의 내용: {request_summary}")
        
        # 질문 유형 결정
        prompt = ChatPromptTemplate.from_messages([
                ("system", PROMPT_SUPERVISOR),
                ("user", " user_request:\n{user_request}\n\n")
        ])
        chain = prompt | self.llm.with_structured_output(Router)
        response = chain.invoke({"user_request": user_request})
        print(f"👨‍💼 다음 단계(Analytics or General or Knowledge): {response.next}")
        return Command(
            update={
                "q_category": response.next, 
                "request_summary": request_summary,
                "eda_stage": 0
            }, 
            goto=response.next
        )
    
    #########################################################
    # ✅ General 노드
    #########################################################
    def handle_general(self, state: State) -> Command:
        """일반적인 질문을 처리하는 노드"""
        print("\n💬 [handle_general] 일반 질문 처리")
        prompt = ChatPromptTemplate.from_messages([
                ("system", PROMPT_GENERAL),
                ("user", " user_request:\n{user_request}\n\n")
        ])
                
        chain = prompt | self.llm
        # user_request = state['messages'][0].content
        user_request = self.original_query
        response = chain.invoke({"user_request": user_request})
        print(f"💬 일반 응답: {response.content}")
        return Command(update={"general_response": response.content}, goto=END)

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
            return Command(update={"knowledge_response": response.content}, goto=END)

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
        return Command(update={"knowledge_response": response.content}, goto=END)
    
    #########################################################
    # ✅ Analytics 노드
    # -> Check_Analysis_Question
    #########################################################
    def handle_analytics(self, state: State) -> Command:
        """분석 요청을 처리하는 노드"""
        print("👨‍💼 [handle_analytics] 분석 요청 처리 시작")
        return Command(goto="Check_Analysis_Question")

    #########################################################
    # ✅ classify_analysis_question 노드
    # -> Eda_Generate_Code / ML_Generate_Code / Generate_Code
    #########################################################
    def classify_analysis_question(self, state: State) -> Command:
        """사용자의 질문이 EDA, ML, 일반 질문인지 판단 및 분류하는 노드"""
        print("=" * 100)
        print("👨‍💼 분석 유형 판단 단계 (EDA vs ML vs 일반)")

        user_question = self.context_query or self.original_query

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
        print(f"👨‍💼 분석 유형: {state_update['analysis_type']}, 다음 단계: {next_node}")

        return Command(update=state_update, goto=next_node)

    #########################################################
    # ✅ Eda_Generate_Code 노드
    # -> Execute_Sample / END
    #########################################################
    def generate_eda_code(self, state: State) -> Command:
        """사용자의 요청을 기반으로 Python 코드 생성"""
        print("="*100)  # 구분선 추가
        print("🤖 코드 생성 단계:")

        # 활성화된 마트가 있는지 확인
        if not self.active_marts:
            print("❌ 활성화된 마트가 없습니다. 먼저 마트를 활성화해주세요.")
            return Command(
                update={"error_message": "❌ 활성화된 마트가 없습니다. 먼저 마트를 활성화해주세요."}, 
                goto='__end__'
            )

        user_request = self.context_query or self.original_query
        
        # 사용자 질문을 기반으로 실행할 EDA 단계 선택
        selected_categories = self.map_to_eda_category(user_request)
        selected_categories = list(dict.fromkeys(selected_categories))
        # print(f"🤖 선택된 카테고리: {selected_categories}")

        # 선택된 프롬프트만 실행
        if "전체" in selected_categories:
            selected_prompts = [PROMPT_EDA_FEATURE_IMPORTANCE_ANALYSIS]
        elif "기타" in selected_categories:
            selected_prompts = [PROMPT_GENERATE_CODE]
        else:
            selected_prompts = [eda_prompt_mapping[category] for category in selected_categories if category in eda_prompt_mapping]
        # print(f"🤖 선택된 프롬프트: {selected_prompts}")

        current_stage = state.get("eda_stage", 0)
        print(f"현재회차: {current_stage}, 수행회차: {len(selected_prompts)}")
        
        if current_stage >= len(selected_prompts):
            print("🤖 모든 EDA 단계를 완료했습니다.")
            return Command(goto=END)
        
        # 현재 실행할 EDA 단계 출력
        # print(f"🤖 실행 중: {selected_prompts[current_stage]}")
        prompt_text = selected_prompts[current_stage]

        # 마트 정보 가져오기 (중괄호 이스케이프 적용)
        mart_info = self._get_mart_info()

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

    #########################################################
    # ✅ ML_Generate_Code 노드
    # -> Execute_Sample / END
    #########################################################
    def generate_ml_code(self, state):
        """사용자의 요청을 기반으로 Python 코드 생성"""
        print("="*100)  # 구분선 추가
        print("🤖 코드 생성 단계:")

        # 활성화된 마트가 있는지 확인
        if not self.active_marts:
            print("❌ 활성화된 마트가 없습니다. 먼저 마트를 활성화해주세요.")
            return Command(
                update={"error_message": "❌ 활성화된 마트가 없습니다. 먼저 마트를 활성화해주세요."}, 
                goto='__end__'
            )

        user_request = self.context_query or self.original_query

        # 🔹 실행할 ML 프로세스 목록 (고정된 순서로 진행)
        ml_process_steps = [
            "PROMPT_ML_SCALING",
            "PROMPT_ML_IMBALANCE_HANDLING",
            "PROMPT_ML_MODEL_SELECTION",
            "PROMPT_ML_HYPERPARAMETER_TUNING",
            "PROMPT_ML_MODEL_EVALUATION",
            "PROMPT_ML_FEATURE_IMPORTANCE"
        ]
        
        # 마트 정보 가져오기 (중괄호 이스케이프 적용)
        mart_info = self._get_mart_info()

        # 🔹 실행할 단계별 프롬프트 적용
        generated_code_list = []
        for prompt in ml_process_steps:
            prompt_text = globals().get(prompt, None)  # 문자열을 실제 프롬프트 변수로 변환
            if not prompt_text:
                print(f"⚠ {prompt} 프롬프트를 찾을 수 없습니다. 스킵합니다.")
                continue  # 해당 프롬프트가 없으면 건너뜀
            prompt_chain = ChatPromptTemplate.from_messages([
                ("system", prompt_text.format(mart_info=mart_info)),
                ("user", "user_request:\n{user_request}"),
            ])
            chain = prompt_chain | self.llm
            response = chain.invoke({
                "user_request": user_request,
            })
            generated_code_list.append(response.content)

        # 🔹 Python 코드 블록만 추출
        list_code = []
        for code in generated_code_list:
            extracted_code = self._extract_code_from_llm_response(code)
            list_code.append(extracted_code)

        if not list_code:
            print("⚠ 코드 블록을 찾을 수 없습니다.")
            return Command(update={
                "generated_code": "# 오류: 코드 블록이 제공되지 않았습니다.",
                }, goto =  "__end__")

        tmp_code = "\n\n".join(list_code).strip()

        # 🔹 코드가 비어있으면 실행 중지
        if not tmp_code:
            print("⚠ 생성된 코드가 없습니다.")
            return Command(
                update={"generated_code": "# 오류: 생성된 코드가 없습니다."}, 
                goto=END
            )

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

    #########################################################
    # ✅ Generate_Code 노드
    # -> Execute_Sample / END
    #########################################################
    def generate_python_code(self, state):
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
                update={"error_message": "❌ 활성화된 마트가 없습니다. 먼저 마트를 활성화해주세요."}, 
                goto='__end__'
            )
        
        user_request = self.context_query or self.original_query
        
        # 마트 정보 가져오기 (중괄호 이스케이프 적용)
        mart_info = self._get_mart_info()

        # 프롬프트 생성 (format 활용)
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
            
            print(f"✅ 샘플 코드 실행 성공")
            self.retry_count = 0  # 성공 시 카운터 초기화
            return Command(update={
                "error_message": None
            }, goto="Execute_Full")

        except Exception as e:
            print(f"❌ 샘플 코드 실행 실패")
            print(f"에러 타입: {type(e).__name__}")
            print(f"에러 메시지: {str(e)}")
            print(traceback.format_exc())
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            self.retry_count += 1
            if self.retry_count >= MAX_RETRIES:
                print("⚠️ 샘플 코드 실행 3회 실패 → 프로세스 종료")
                return Command(update={"error_message": error_details}, goto="__end__")
            return Command(update={"error_message": error_details}, goto="Regenerate_Code")

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
                ])
        
        chain = prompt | self.llm
        
        # 코드 재생성
        response = chain.invoke({
            "user_request": user_request,
            "original_code": original_code,
            "error_message": error_message
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
        code_to_execute = self._extract_code_from_llm_response(
            state.get("regenerated_code") or state["generated_code"]
        )
        try:
            # 전체 코드 실행
            output, analytic_results = self._execute_code_with_capture(code_to_execute, exec_globals, is_sample=False)
            token_count = self._calculate_tokens(str(analytic_results))
            
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
                unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
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
        execution_output = state["execution_output"]
        # 분석 결과와 실행 출력을 함께 저장
        save_data = {
            'analytic_result': analytic_result,
            'execution_output': execution_output
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
        user_question = state["messages"][0].content

        # ✅ 집계 데이터면 전체 데이터 전달
        string_of_result = str(dict_result)

        ############################################################
        # 1. 인사이트 생성
        ############################################################
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
        
        ############################################################
        # 2. 차트 필요 여부 결정
        ############################################################
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_CHART_NEEDED),
            ("user", "user_question:\n{user_question}\n\n"),
            ("user", "analytic_result:\n{analytic_result}\n\n"),
            ("user", "insight:\n{insight}\n\n")
        ])
        
        # 차트 활용 여부 'yes' 또는 'no' 반환
        chart_decision_messages = prompt | self.llm
        chart_needed = chart_decision_messages.invoke({
            "user_question": user_question,
            "analytic_result": string_of_result,
            "insight": insight_response.content
        }).content.strip().lower()
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
            "insights": insights
        }).content

        extracted_code = self._extract_code_from_llm_response(chart_code)

        # ✅ 유효한 Python 코드 블록이 없는 경우 재시도
        if not extracted_code:
            print("📊 [regenerate_chart] 유효한 Python 코드 블록이 없습니다. 재시도합니다.")
            error_info = {
                "error_message": "유효한 Python 코드 블록이 없습니다",
                "previous_code": chart_code
            }
            self.retry_count += 1
            return Command(update={
                "retry_count": self.retry_count + 1,
                "chart_error": error_info
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
            error_info = {"error_message": str(e),"previous_code": extracted_code,"traceback": traceback.format_exc()}
            self.retry_count += 1
            return Command(update={
                "chart_filename": None,
                "chart_error": error_info
            }, goto="Regenerate_Chart")

    #########################################################
    # ✅ Report_Builder 노드
    # -> END
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
        return Command(update={
            "report": response.content, 
            "error_message": None
        }, goto=END)
    

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
        """Report_Builder 이후 EDA 단계를 추가 실행할지 결정"""
        selected_categories = state.get("selected_categories", [])  # ✅ 기본값을 빈 리스트로 설정
        eda_stage = state.get("eda_stage", 0)

        # ✅ 모든 EDA 단계를 완료했다면 END로 이동
        if eda_stage >= len(selected_categories):
            return END

        return "Eda_Generate_Code"

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
            1. 만약 EDA 전반적인 내용을 묻는다면 "전체"라는 값을 반환하세요.
            2. 만약 EDA 전반적인 내용이 아니며 특정 단계에 해당되는 내용이 아니라면 "기타"라는 값을 반환하세요.
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
                if "전체" in llm_response:
                    matched_categories = ["전체"]
                elif "기타" in llm_response:
                    matched_categories = ["기타"]
            print(f"🔍 매칭된 카테고리: {matched_categories}")
        
        return matched_categories
    

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
