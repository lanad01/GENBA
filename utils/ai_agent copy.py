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


class AIAnalysisAssistant:
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

    def metadata_analysis(self, state: State) -> Command:
        user_request = state["messages"][-1].content
        prompt = f"""
        You are a data analysis expert. Your task is to analyze the provided metadata in the context of the user's request to identify relevant tables and features suitable for building a machine learning classification model.

        1. **Objectives**:
           - Analyze the user's request to understand the purpose of the machine learning model.
           - Cross-reference the user's request with the provided metadata to recommend tables and features that align with the model's objective.
           - Exclude irrelevant features such as IDs, duplicated data, or unrelated tables.
           - Provide a structured report summarizing the selected tables and features.

        2. **Input Data**:
           - User request: {user_request}
           - Metadata document: {self.metadata_doc}

        3. **Output Results**:
           - `report(table_structure_summary)`: A report summarizing the selected tables and their attributes.
           - `recommended_schema`: A dictionary listing relevant tables and features for model training.
           - Ensure variable names are consistent to avoid KeyErrors in subsequent processes.
           - Generate executable Python code to analyze the metadata based on the user's request. The code should use Python’s Pandas library and return both `recommended_schema` and `table_structure_summary` as a dictionary.
        """

        response = self.llm.invoke(prompt)
        # Assume the LLM returns executable Python code marked with ### python_code
        python_code = response.split("### python_code")[-1].strip()
        result = {}
        exec(python_code, globals(), result)

        return Command(update={"recommended_schema": result.get("recommended_schema"), "report": result.get("table_structure_summary")}, goto="data_extraction")

    def data_extraction(self, state: State) -> Command:
        user_request = state["messages"][-1].content
        recommended_schema = state["recommended_schema"]

        prompt = f"""
        You are a data analysis expert. Your task is to generate SQL queries based on the recommended schema and extract data from a PostgreSQL database to create a data mart for machine learning model training.

        1. **Objectives**:
           - Generate SQL queries using the provided `recommended_schema`.
           - Extract data from the PostgreSQL database and load it into a Pandas DataFrame.
           - Assess the quality of the extracted data by identifying missing values and providing basic statistical summaries.

        2. **Input Data**:
           - User request: {user_request}
           - Recommended schema: {recommended_schema}

        3. **Output Results**:
           - Extracted dataset saved as `data_mart`.
           - Generate a report summarizing the extracted data and its attributes (`data usage report`).
           - Create a `data quality report` highlighting missing values and data inconsistencies.
           - Ensure all variable names are consistent to prevent KeyErrors in subsequent processes.
           - Generate executable Python code to connect to the PostgreSQL database, extract the data, and return both `data_mart` and `data_quality_report` as a dictionary.
        """

        response = self.llm.invoke(prompt)
        # Assume the LLM returns executable Python code marked with ### python_code
        python_code = response.split("### python_code")[-1].strip()
        result = {}
        exec(python_code, globals(), result)

        return Command(update={"data_mart": result.get("data_mart"), "report": result.get("data_quality_report")}, goto=END)

    def build_graph(self):
        workflow = StateGraph(State)
        workflow.add_node("metadata_analysis", self.metadata_analysis)
        workflow.add_node("data_extraction", self.data_extraction)

        workflow.set_entry_point("metadata_analysis")
        workflow.add_edge("metadata_analysis", "data_extraction")

        self.app = workflow.compile()

    def ask(self, query: str):
        return self.app.invoke({"messages": [HumanMessage(content=query)]})