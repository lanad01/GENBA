import os, sys
from datetime import datetime

from IPython.display import display, Image
import pandas as pd
from typing import TypedDict, List, Literal, Annotated
from pydantic import BaseModel
from uuid import uuid4
import matplotlib.pyplot as plt
import matplotlib
import operator

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain.vectorstores import FAISS
from langchain.schema import Document

from utils.with_postgre import PostgresDB
from utils.get_suggestions import get_suggestions
from prompt.prompt_agency import PromptAgency

schema_name = "biz"
num_db_return = 20
recursion_limit = 10
MAX_RETRIES = 3 
# ✅ 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 사용 시
matplotlib.rcParams['axes.unicode_minus'] = False

class State(TypedDict):
    messages: List[HumanMessage]
    documents: List[str]
    query : str
    dataframe : pd.DataFrame
    chart_decision: str
    chart_filename: str
    insights: str
    report_filename: str
    sql_attempts : int


class Router(BaseModel):
    next: Literal['SQL_Builder', 'Chart_Builder', 'Insight_Builder', 'Replier', 'Report_Builder', 'General_Query_Handler', '__end__']


class SQLQuery(BaseModel):
    query: str

class MartAssistant:
    def __init__(self, vectorstore: FAISS, openai_api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
        self.db = PostgresDB()
        self.prompts = PromptAgency()  # PromptAgency 인스턴스 생성
        self.retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5}) if vectorstore else None
        self.chart_counter = 1  # 차트 파일 명 생성보드 큰수 설정
        self.build_graph()


    def retrieve_documents(self, state: State) -> State:
        query = state["messages"][-1].content
        if self.retriever:
            docs = self.retriever.invoke(query)
            documents = [
                Document(page_content=doc.page_content, metadata={"source": doc.metadata.get("source", "Unknown")})
                for doc in docs
            ]
        print(f"✅ [retrieve_documents] 참조 문서 건수: {len(documents)}")
        return {**state, "documents": documents}


    def supervisor(self, state: State) -> Command:

        dataframe_generated = "dataframe" in state and not state["dataframe"].empty
        insights_generated = "insights" in state and state["insights"]
        print(f"😀 [Supervisor] 데이터 프레임 생성 여부: {dataframe_generated}")
        print(f"😀 [Supervisor] 인사이트 생성 여부: {insights_generated}")

        if not dataframe_generated:
            prompt = self.prompts.get_supervisor_prompt()
            chain = prompt | self.llm.with_structured_output(Router)
            response = chain.invoke({"messages": state['messages'][0].content})
            print(f"😀 [Supervisor] 데이터 분석 요청 여부 판단: {response.next}")
            if response.next == 'General_Query_Handler': # 일반 질의 처리기로 이동
                return Command(goto='General_Query_Handler')
            return Command(goto=response.next)
        
        elif dataframe_generated :
            if not insights_generated:
                print(f"😀 [Supervisor] 데이터 생성 완료, 인사이트 생성으로 이동.")
                return Command(goto="Insight_Builder")
        

    def general_query_handler(self, state: State) -> Command:
        suggestions = get_suggestions()
        response = f"이 질문은 데이터 분석과 관련이 없는 것 같습니다. 😊\n {suggestions}"
        return Command(update={"messages": [AIMessage(content=response)]}, goto="__end__")


    def sql_builder(self, state: State) -> Command:
        question = state["messages"][-1].content
        print(f"🚀 [SQL_Builder] 들어온 질문: {question}")

        state.setdefault("documents", [])
        document_texts = [doc.page_content if isinstance(doc, Document) else str(doc) for doc in state["documents"]]
        prompt = self.prompts.get_sql_builder_prompt(schema_name=schema_name)

        query_chain = prompt | self.llm.with_structured_output(SQLQuery)
        response = query_chain.invoke({"messages": state["messages"][0].content, "documents": document_texts})
        query = response.query
        print(f"🚀 [SQL_Builder] 생성된 쿼리:\n {query}")

        success, result = self.db.limited_run(query, num=40)
        if success:
            print(f"🚀 [SQL_Builder] 쿼리 성공")
            headers = [col.name for col in self.db.cursor.description]
            df = pd.DataFrame(result, columns=headers)
            
            if df.empty:
                print(f"🚀 [SQL_Builder] 쿼리는 성공했으나 결과가 없습니다.")
                return Command(update={
                    "messages": [AIMessage(content=f"**쿼리 성공 (빈 결과)**\n```sql\n{query}```\n\n쿼리는 성공했지만 결과가 없습니다. 쿼리를 수정합니다.")],
                    "sql_attempts": state.get("sql_attempts", 0) + 1  # 시도 횟수 증가
                }, goto="sql_rebuilder")
            
            display(df.head())
            return Command(update={
                "dataframe": df,
                "query": query
            }, goto=END)
        else:
            print(f"🚀 [SQL_Builder] 쿼리 실패")
            return Command(
                update={
                    "messages": [AIMessage(content=f"**쿼리 실패**\n```sql\n{query}```\n\n**쿼리 실행 결과**\n{result}")]
                },
                goto="sql_rebuilder"
            )
            
                
    def sql_rebuilder(self, state: State) -> Command:
        retry_count = state.get("sql_attempts", 0)

        if retry_count >= 3:
            print("🛑 [SQL_Rebuilder] 3회 이상 쿼리 수정 실패. 프로세스를 종료합니다.")
            return Command(update={
                "messages": [AIMessage(content=f"3회 이상 쿼리 수정에 실패했습니다. 분석을 종료합니다.\n {state["messages"][-1].content}")]
            }, goto="__end__")

        last_message = state["messages"][-1].content

        if '**쿼리 실행 결과**\n' in last_message:
            query = last_message.split('```sql\n')[1].split('```')[0]
            error_message = last_message.split('**쿼리 실행 결과**\n')[1]
        elif '**쿼리 성공 (빈 결과)**' in last_message:
            query = last_message.split('```sql\n')[1].split('```')[0]
            error_message = "쿼리는 성공했지만 결과가 없습니다. 조건이나 필터링을 수정해야 합니다."
        else:
            print("🛑 [SQL_Rebuilder] 쿼리 및 오류 메시지를 분석할 수 없습니다.")
            return Command(update={
                "messages": [AIMessage(content="쿼리 및 오류 메시지를 분석할 수 없습니다. 수동 검토가 필요합니다.")]
            }, goto="__end__")

        prompt = self.prompts.get_sql_rebuilder_prompt()
        query_chain = prompt | self.llm.with_structured_output(SQLQuery)

        response = query_chain.invoke({"sql_query": query, "error_message": error_message})
        new_query = response.query

        print(f"🗻 [SQL_Rebuilder] 수정된 쿼리: {new_query}")

        success, result = self.db.limited_run(new_query, num=40)
        if success:
            headers = [col.name for col in self.db.cursor.description]
            df = pd.DataFrame(result, columns=headers)
            if df.empty:
                print(f"🗻 [SQL_Rebuilder] 수정된 쿼리는 성공했으나 결과가 없습니다.")
                return Command(update={
                    "messages": [AIMessage(content=f"**수정된 쿼리 성공 (빈 결과)**\n```sql\n{new_query}```\n\n수정된 쿼리는 성공했지만 결과가 없습니다. 분석을 종료합니다.")],
                }, goto="__end__")
            
            return Command(update={
                "messages": [AIMessage(content=f"**수정된 쿼리**\n```sql\n{new_query}```\n\n**쿼리 실행 결과가 아래 표에 표시됩니다.**")],
                "dataframe": df,
                "query": new_query
            }, goto="__end__")
        else:
            print(f"🗻 [SQL_Rebuilder] 쿼리 수정 실패, 재시도합니다.")
            return Command(update={
                "messages": [AIMessage(content=f"**수정된 쿼리 실패**\n```sql\n{new_query}```\n\n**쿼리 실행 결과**\n{result}")],
                "sql_attempts": retry_count + 1
            }, goto="sql_rebuilder")



    def insight_builder(self, state: State) -> Command:
        prompt = self.prompts.get_insight_builder_prompt()
        dataframe_text = state.get("dataframe", "No dataframe generated.") 

        insight_chain = prompt | self.llm
        insight = insight_chain.invoke({"question": state["messages"][0].content, "dataframe": dataframe_text}).content
        print(f"🌀 [Insight_Builder] 생성된 인사이트:\n {insight}")

        prompt = self.prompts.get_chart_decision_prompt()
        decision_chain = prompt | self.llm
        decision = decision_chain.invoke({"question": state["messages"][0].content, "dataframe": dataframe_text, "insights" : insight}).content.strip().lower()
        print(f"🌀 [Insight_Builder] 차트 필요여부 판단: {decision}")
        
        return Command(update={
            "insights": insight,
            "chart_decision": decision
        }, goto="Chart_Builder")


    def chart_builder(self, state: State) -> Command:
        
        decision = state.get("chart_decision", "").strip().lower()
        if 'yes' in decision:
            print("🌀 [Chart_Builder] 차트 생성 진행")
        else:
            print("🌀 [Chart_Builder] 차트 생성 건너뜀")
            return Command(update={"chart_filename": None}, goto="Report_Builder")
        
        prompt = self.prompts.get_chart_builder_prompt()
        dataframe_text = state.get("dataframe", "No dataframe generated.") 
        insights_text = state.get("insights", "No insights generated.")

        chart_chain = prompt | self.llm
        chart_code = chart_chain.invoke({"question": state["messages"][0].content, "dataframe": dataframe_text, "insights" : insights_text}).content
        print(f"📈 [Chart_Builder] 생성된 차트 코드:\n {chart_code}")

        # 차트 생성 및 이미지 생성
        timestamp = datetime.now().strftime("%m%d-%H-%M-%S")
        filename = f'{timestamp}-chart{self.chart_counter:04}.png'
        self.chart_counter += 1
        os.makedirs('img', exist_ok=True)

        modified_code = chart_code.split("```python")[-1].split("```")[0]
        modified_code += f"\nplt.savefig('../img/{filename}')\nplt.show()"
        try:
            exec(modified_code, globals())
            print(f"📈 [create_chart] 차트 코드 수행 성공 및 이미지 저장 완료: ../img/{filename}")
            plt.close()
            return Command(update={"chart_filename": filename}, goto="Report_Builder")
        except Exception as e:
            print(f"📈 [create_chart] 차트 코드 실행 중 오류 발생: {e}")
            plt.close()
            return Command(update={"chart_filename": None}, goto="__end__")
        
        

    def report_builder(self, state):
        print("📝 [Report_builder] 시작")
        prompt = self.prompts.get_report_builder_prompt()

        dataframe_text = state.get("dataframe", "No dataframe generated.") 
        insights_text = state.get("insights", "No insights generated.")
        chart_filename = state.get("chart_filename", "No charts included.")
        report_chain = prompt | self.llm

        report_content = report_chain.invoke({
            "question":  state["messages"][0].content,
            "dataframe": dataframe_text,
            "insights": insights_text,
            "chart_filename": chart_filename,
        }).content

        # 생성 코드를 추출
        if "```python" in report_content:
            modified_code = report_content.split("```python")[-1].split("```")[0]
        else:
            print("Error: 'generated_code' key not found in report_content")
            modified_code = None

        # 예외처리를 위한 시도 횟수 설정
        retry_attempts = 0
        max_retries = 1
        success = False

        while retry_attempts <= max_retries and not success:
            try:
                if modified_code:
                    print("📝 [Report_builder] 코드 생성:\n", modified_code)
                    if not os.path.exists('../output'): # 'output' 디렉토리 존재 여부 확인 및 생성
                        os.makedirs('../output')
                    exec(modified_code, globals())
                    success = True  # 코드 실행 성공
                else:
                    print("📝 [Report_builder] No code to execute.")
                    return Command(update={"report_filename": "failed"}, goto="__end__")
                
            except Exception as e:
                print(f"🛑 [Report_builder]  코드 실행 중 오류 발생: {e}")
                if retry_attempts < max_retries:
                    print("🔄 오류 수정 후 재시도 중...")
                    # 오류 메시지를 기반으로 LLM에 수정 요청
                    error_prompt = self.prompts.get_error_fix_prompt()
                    fix_chain = error_prompt | self.llm
                    fix_response = fix_chain.invoke({
                        "error_message": str(e),
                        "original_code": modified_code
                    }).content

                    if "```python" in fix_response:
                        modified_code = fix_response.split("```python")[-1].split("```")[0]
                        print("🔧 수정된 코드:\n", modified_code)
                        exec(modified_code, globals())
                        success = True  # 코드 실행 성공
                    else:
                        print("Error: 'fixed_code' key not found in fix_response")
                        return Command(update={"report_filename": "failed"}, goto="__end__")
                else:
                    print("❌ 재시도 후에도 오류 발생. 보고서 생성을 보류합니다.")
                    return Command(update={"report_filename": "failed"}, goto="__end__")
                retry_attempts += 1

        if success:
            print("📝 [Report_Builder] 보고서 생성 완료.")
            report_status = "success"
        else:
            print("📝 [Report_Builder] 보고서 생성 실패.")
            report_status = "failed"

        return Command(update={
            "messages": [AIMessage(content="성공적으로 생성되었습니다.")],
            "report_filename": report_status
        }, goto=END)


    def build_graph(self):
        workflow_subgraph_sqlbuilder = StateGraph(State)
        workflow_subgraph_sqlbuilder.add_node("retrieve_documents", self.retrieve_documents)
        workflow_subgraph_sqlbuilder.add_node("sql_builder", self.sql_builder)
        workflow_subgraph_sqlbuilder.add_node("sql_rebuilder", self.sql_rebuilder)

        workflow_subgraph_sqlbuilder.set_entry_point("retrieve_documents")
        workflow_subgraph_sqlbuilder.add_edge("retrieve_documents", "sql_builder")
        workflow_subgraph_sqlbuilder.add_conditional_edges(
            "sql_builder", 
            self.route_after_sql_builder, 
            {
                "sql_rebuilder": "sql_rebuilder",
                END : END
            }
        )
        workflow_subgraph_sqlbuilder.add_conditional_edges(
            "sql_rebuilder", 
            self.route_after_sql_rebuilder, 
            {
                "sql_rebuilder": "sql_rebuilder",
                END : END
            }
        )
        subgraph_sqlbuilder = workflow_subgraph_sqlbuilder.compile()
        
        workflow = StateGraph(State)
        workflow.add_node("Supervisor", self.supervisor)
        workflow.add_node("General_Query_Handler", self.general_query_handler)
        workflow.add_node("SQL_Builder", subgraph_sqlbuilder)
        workflow.add_node("Insight_Builder", self.insight_builder)
        workflow.add_node("Chart_Builder", self.chart_builder)
        workflow.add_node("Report_Builder", self.report_builder)

        workflow.set_entry_point("Supervisor")
        workflow.add_edge("General_Query_Handler", END)
        workflow.add_edge("SQL_Builder", "Insight_Builder")

        workflow.add_conditional_edges(
            "Insight_Builder",
            self.route_after_insight_builder,
            {
                "Report_Builder": "Report_Builder",   # 차트가 필요 있으면 Chart_Builder로
                "Chart_Builder": "Chart_Builder"      # 차트가 필요 없으면 Supervisor로
            },
        )

        workflow.add_conditional_edges(
            "Chart_Builder",
            self.route_after_chart_builder,
            {
                "Report_Builder": "Report_Builder",   # 차트가 필요 있으면 Chart_Builder로
                END : END
            },
        )

        workflow.add_edge("Report_Builder", END)

        self.app = workflow.compile()


    def ask(self, query: str):
        print(f"질문: {query}")
        return self.app.invoke({"messages": [HumanMessage(content=query)]}, config={"recursion_limit": 15})

    # ✅ 조건부 엣지 추가: 차트 생성 여부에 따라 흐름 제어
    @staticmethod
    def chart_is_none(self, state: State) -> str:
        # print(f"📈 [chart_is_none?] Chart file name  {state.get("chart_filename", None)}")
        return state.get("chart_filename", None) is None  # 차트가 없으면 Supervisor로 이동

    def route_after_sql_builder(self, state: State) -> str:
        """SQL 빌더 후 다음 단계를 결정하는 라우터
        
        Returns:
            str: 다음 실행할 노드의 이름
        """
        print("➡️ [route_after_sql_builder] 전체 데이터 실행 후 경로 결정")
        
        if state.get("query"):  
            print("➡️ [route_after_sql_builder] SQL 생성 서브그래프 종료")
            return END
        else :
            print("➡️ [route_after_sql_builder] 쿼리 재생성 진행")
            return "sql_rebuilder"
        
    
    def route_after_sql_rebuilder(self, state: State) -> str:
        """SQL 빌더 후 다음 단계를 결정하는 라우터
        
        Returns:
            str: 다음 실행할 노드의 이름
        """
        print("➡️ [route_after_sql_rebuilder] 전체 데이터 실행 후 경로 결정")
        retry_count = state.get("sql_attempts", 0)
        
        if retry_count >= MAX_RETRIES:
            print("⚠️ 전체 데이터 실행 3회 실패 → 프로세스 종료")
            return END
        if state.get("query"):  
            print("➡️ [route_after_sql_rebuilder] 인사이트 생성 단계로 진행")
            return END
        else :
            print("➡️ [route_after_sql_rebuilder] 쿼리 재생성 진행")
            return "sql_rebuilder"
        
        
    def route_after_insight_builder(self, state: State) -> str:
        """인사이트 빌더 후 다음 단계를 결정하는 라우터
        
        Returns:
            str: 다음 실행할 노드의 이름
        """
        print("➡️ [route_after_insight_builder] 전체 데이터 실행 후 경로 결정")
        decision = state.get("chart_decision", "").strip().lower()
        
        if 'yes' not in decision:
            print("➡️ [route_after_insight_builder] 보고서 생성 단계로 진행")
            return "Report_Builder"
        else :
            print("➡️ [route_after_insight_builder] 차트 생성 단계로 진행")
            return "Chart_Builder"
        
    def route_after_chart_builder(self, state: State) -> str:
        """차트 빌더 후 다음 단계를 결정하는 라우터
        
        Returns:
            str: 다음 실행할 노드의 이름
        """
        print("➡️ [route_after_chart_builder] 전체 데이터 실행 후 경로 결정")
        decision = state.get("chart_decision", "").strip().lower()
        
        if 'yes' not in decision:
            print("➡️ [route_after_chart_builder] 보고서 생성 단계로 진행")
            return "Report_Builder"
        else :
            print("➡️ [route_after_chart_builder] 차트 생성 단계로 진행")
            return END

