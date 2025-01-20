# from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict

import langchain
import copy
import logging

config = RunnableConfig(recursion_limit=10)

COPIABLE_KEYS = [
    "callbacks",
    "metadata",
    "tags",
    "verbose",
    # 필요한 키 추가
]

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        transform_cnt: Number of times the transform node was executed
        include_words: include keyword
        exclude_words: exclude keyword
    """
    question: str
    generation: str
    documents: List[str]
    transform_cnt: int
    include_words: List[str]
    exclude_words: List[str]

class ChatNEWS():
    local_embeddings = None
    app = None

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        m_path = "/home/work/bdp-txt-ans/model/ko-sroberta-multitask"
        self.local_embeddings = SentenceTransformerEmbeddings(
            model_name=m_path,
            model_kwargs={"device": "cpu"}
        )

    def get_retrieval_grader(self):
        self.llm.format = "json"
        prompt = PromptTemplate(
            template=(
                "당신은 뉴스기사와 질의문 간의 관련성을 평가하는 평가자입니다. \n"
                "방법은 JSON 포맷으로 'score', 'reason' 각각의 키를 담아 평가하세요. \n"
                "뉴스기사와 질의문에서 찾으려는 내용과 연결적으로 관련성이 있다면, \n"
                "'score' 키에 'yes'로 답하고, 관련성이 낮다면 'no'로 답하세요. \n"
                "평가는 간단한 한글 문장으로 생성하고, 'reason' 키에 한글로 작성해주세요. \n\n"
                "검색질문: {question} \n"
                "뉴스기사: {document} \n\n"
                "JSON포맷으로 답하세요."
            ),
            input_variables=["question", "document"]
        )

        retrieval_grader = prompt | self.llm | JsonOutputParser()
        return retrieval_grader

    def get_summary_chain(self):
        self.llm.format = ""
        prompt = PromptTemplate(
            template=(
                "당신은 요청문장과 주어진 뉴스기사들의 내용을 요약해주는 도우미입니다. \n"
                "생각과 서론 없이 최대 2문장 이내로 하나의 문단을 생성하세요. \n"
                "'간단한 주제와 요약' 등의 표현은 포함하지 마세요. \n\n"
                "- 용어 참고사항 \n"
                "1) '국내'='한국' \n"
                "2) '농협'='농협은행' \n"
                "3) '농협은행'='농협중앙회' \n"
                "4) '농협은행'='NH' \n"
                "5) '농협은행'='축협' \n\n"
                "- 요청문장: {question} \n"
                "- 뉴스기사들: {context} \n\n"
                "- 서론 없이 간결하게 요약한 40글자 이내로 생성된 문장:"
            ),
            input_variables=["question", "context"]
        )

        rag_chain = prompt | self.llm | StrOutputParser()  # Chain
        return rag_chain

    def get_hallucination_grader(self):
        self.llm.format = "json"
        prompt = PromptTemplate(
            template=(
                "당신은 요약문장이 검색된 문서에 근거했는지 여부를 평가하는 평가자입니다. \n"
                "답변은 JSON 포맷으로 하고, 'score'와 아닌 다른 키는 절대 포함하지 마세요. \n"
                "- 요약문장이 검색된 문서를 근거로 생성되었다고 판단되면 'score' 키에 \n"
                "  'yes'로 답하고, 아니라면 'no'로 답하세요. \n"
                "- 검색된 문서에 없는 내용을 요약하였다면 'no', 아니라면 'yes'로 답하세요. \n\n"
                "- 요약 문장: {generation} \n"
                "- 검색된 문서: {documents} \n\n"
                "- JSON포맷의 답변:"
            ),
            input_variables=["generation", "documents"]
        )

        hallucination_grader = prompt | self.llm | JsonOutputParser()
        return hallucination_grader
        
    def get_answer_grader(self):
        self.llm.format = "json"
        prompt = PromptTemplate(
            template=(
                "당신은 요약된 문장이 사용자의 관심주제를 다루고 있는지 평가하는 평가자입니다. \n"
                "답변은 JSON 포맷으로 하고, 'score'와 아닌 다른 키는 절대 포함하지 마세요. \n"
                "- 요약된 문장이 사용자의 관심주제를 다루고 있다고 판단되면 'score' 키에 'yes'로 답하고, 아니라면 'no'로 답하세요. \n\n"
                "- 요약된 문장: {generation} \n"
                "- 관심질문: {question} \n\n"
                "- JSON포맷의 답변:"
            ),
            input_variables=["generation", "question"]
        )
        answer_grader = prompt | self.llm | JsonOutputParser()
        return answer_grader

    def get_question_rewriter(self):
        self.llm.format = ""
        re_write_prompt = PromptTemplate(
            template=(
                "당신은 입력된 질문과 검색된 문서를 분석해 더 향상된 질문을 작성하는 도우미입니다. \n"
                "입력된 질문의 의도를 정확히 분석하고, 불필요한 내용은 생략하고 새로운 질문을 작성하세요. \n"
                "다음은 초기 질문입니다: {question} \n\n"
                "서론 없이 개선된 질문을 다음과 같이 작성하세요: {generation}"
            ),
            input_variables=["generation", "question"]
        )
        question_rewriter = re_write_prompt | self.llm | StrOutputParser()
        return question_rewriter

    def news_search(self):
        retrieval_grader = self.get_retrieval_grader()
        summary_chain = self.get_summary_chain()
        hallucination_grader = self.get_hallucination_grader()
        answer_grader = self.get_answer_grader()
        question_rewriter = self.get_question_rewriter()

        # 유틸리티 함수
        def remove_duplicates_by_title(doc_list):
            seen_titles = set()
            unique_docs = []
            for doc in doc_list:
                title = doc.metadata.get('title')
                if title not in seen_titles:
                    unique_docs.append(doc)
                    seen_titles.add(title)
            return unique_docs

        def check_keywords(doc_list, in_list, ex_list):
            if len(in_list) > 0:
                print("in_list is not None")
                doc_list = [
                    doc for doc in doc_list 
                    if any(keyword in doc.page_content for keyword in in_list)
                ]
            if len(ex_list) > 0:
                print("ex_list is not None")
                doc_list = [
                    doc for doc in doc_list 
                    if not any(keyword in doc.page_content for keyword in ex_list)
                ]
            return doc_list

        def del_meta_keys(documents, keys_to_del):
            documents_copy = copy.deepcopy(documents)
            for doc in documents_copy:
                for key in keys_to_del:
                    if key in doc.metadata:
                        del doc.metadata[key]
            return documents_copy
        
        # node_function
        def initial_state(state):
            state["transform_cnt"] = 0
            ## LLM의 값처럼 JSON 답변을 보정하기 위한 처리.
            if (
                (state["include_words"] == "None")
                or (state["include_words"] == ["None"])
                or (state["include_words"] is None)
                or (state["include_words"] == [""])
            ):
                state["include_words"] = []

            if (
                (state["exclude_words"] == "None")
                or (state["exclude_words"] == ["None"])
                or (state["exclude_words"] is None)
                or (state["exclude_words"] == [""])
            ):
                state["exclude_words"] = []

            return state
        
        def retrieve(state):
            """
            Retrieve documents
            Args:
                state (dict): The current graph state
            Returns:
                state (dict): New key added to state, documents, that contains retrieved documents
            """
            question = state["question"]
            include_words = state["include_words"]
            exclude_words = state["exclude_words"]

            documents = self.retriever.invoke(question, include_words, exclude_words)
            ## bm25와 chromadb 둘다 나온 케이스 제거
            documents = remove_duplicates_by_title(documents)
            ## check include & exclude
            documents = check_keywords(documents, include_words, exclude_words)

            return {"documents": documents, "question": question}
        
        def generate(state):
            """
            Generate answer
            Args:
                state (dict): The current graph state
            Returns:
                state (dict): New key added to state, generation, that contains LLM generation
            """
            question = state["question"]

            # 2024.11.15 retrieve 및 평가가 완료된 document에 대해 dup_count 기준으로 정렬
            documents = sorted(state["documents"], key=lambda doc: doc.metadata.get("dup_count"), reverse=True)
            ### 2024.12.12 Doc 개수는 최대 4개만 사용하도록 함
            if len(documents) > 4:
                documents = documents[:4]

            ## 요약시에 doc 내에 다양한 metadata가 있더보니 요약문이 너무 길어진다.
            keys_to_del = ["reason", "search_src", "bm25_score", "distance", "dup_count"]
            context_doc = del_meta_keys(documents, keys_to_del)

            # RAG generation
            generation = summary_chain.invoke({"context": context_doc, "question": question})

            return {"documents": documents, "question": question, "generation": generation}


        def grade_documents(state):
            """
            Determines whether the retrieved documents are relevant to the question.
            Args:
                state (dict): The current graph state
            Returns:
                state (dict): Updates documents key with only filtered relevant documents
            """
            question = state["question"]
            documents = state["documents"]
            exclude_words = state["exclude_words"]

            # Score each doc
            filtered_docs = []
            for d in documents:
                score = retrieval_grader.invoke(
                    {
                        "question": question,
                        "document": d.page_content,
                        "exclude_words": exclude_words,
                    }
                )
                grade = score["score"]
                if grade == "yes":
                    if "reason" in score:
                        d.metadata["reason"] = score["reason"]
                    filtered_docs.append(d)
                else:
                    continue

            return {"documents": filtered_docs, "question": question}
        
        
        def transform_query(state):
            """
            Transform the query to produce a better question.
            Args:
                state (dict): The current graph state
            Returns:
                state (dict): Updates question key with a re-phrased question
            """
            question = state["question"]
            documents = state["documents"]
            transform_cnt = state["transform_cnt"]

            # Re-write question
            better_question = question_rewriter.invoke({"question": question})
            return {"documents": documents, "question": better_question, "transform_cnt": transform_cnt + 1}


        def decide_to_generate(state):
            """
            Determines whether to generate an answer, or re-generate a question.
            Args:
                state (dict): The current graph state
            Returns:
                str: Binary decision for next node to call
            """
            filtered_documents = state["documents"]
            if not filtered_documents:
                # 1 cycle만 실행하기로 하여 일단 주석 처리함
                if state["transform_cnt"] > 0:
                    print("transform_cnt limited")
                    return False
                # else:
                #     return "transform_query"
                return False

            else:
                # We have relevant documents, so generate answer
                return "generate"

        def grade_generation_v_documents_and_question(state):
            """
            Determines whether the generation is grounded in the document and answers question.
            Args:
                state (dict): The current graph state
            Returns:
                str: Decision for next node to call
            """
            question = state["question"]
            documents = state["documents"]
            generation = state["generation"]

            ## 요약시에 doc 내에 다양한 metadata가 있다보니 요약문이 너무 길어진다.
            keys_to_del = ["reason", "search_src", "bm25_score", "distance", "dup_count"]
            context_doc = del_meta_keys(documents, keys_to_del)
            score = hallucination_grader.invoke({"documents": context_doc, "generation": generation})
            grade = score["score"]

            # Check hallucination
            if grade == "yes":
                print("--GRADE GENERATION VS QUESTION--")
                score = answer_grader.invoke({"question": question, "generation": generation})
                grade = score["score"]
                if grade == "yes":
                    return "useful"
                else:
                # 1 cycle만 실행하기로 하여 일단 주석 처리함
                # if state["transform_cnt"] > 0:
                #     return False
                # else:
                # print("--DECISION: GENERATION DOES NOT ADDRESS QUESTION--")
                # return "not useful"
                    return False
            else:
                # print("--DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY--")
                return "not supported"
            
            
        workflow = StateGraph(GraphState)
        

        # Define the nodes
        workflow.add_node("init_state", initial_state)  # retrieve
        workflow.add_node("retrieve", retrieve)  # retrieve
        workflow.add_node("grade_documents", grade_documents)  # grade documents
        workflow.add_node("generate", generate)  # generate
        workflow.add_node("transform_query", transform_query)  # transform query

        # Build graph
        workflow.add_edge(START, "init_state")
        workflow.add_edge("init_state", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        # Compile
        self.app = workflow.compile()


    def ask(self, query: str, include_words: List[str] = None, exclude_words: List[str] = None):
        try:
            value = self.app.invoke({"question": query, "include_words": include_words, "exclude_words": exclude_words}, config=config)
            errbit = False
        except Exception as e:
            errbit = True

        if errbit:
            result = {
                "rag_result": "죄송합니다. 관련된 뉴스를 찾지 못했습니다.",
                "count": 0,
                "docs": None
            }
        else:
            result = value["generation"]
            result_cnt = len(value["documents"])
            docs = []
            if result_cnt > 0:
                for i, doc in enumerate(value["documents"][:result_cnt]):
                    if doc.metadata["search_src"] == "bm25":
                        doc.metadata["distance"] = doc.metadata["bm25_score"]
                    docs.append({
                        "title": doc.metadata["title"],
                        "page_content": doc.page_content,
                        "posts_dt": doc.metadata["posts_dt"],
                        "url": doc.metadata["url"],
                        "dup_count": doc.metadata["dup_count"],
                        "search_src": doc.metadata["search_src"],
                        "reason": doc.metadata["reason"],
                        "distance_or_score": doc.metadata["distance"]
                    })
            result = {
                "rag_result": value["generation"],
                "count": result_cnt,
                "docs": docs
            }
        return result

    def set_llm(self, llm):
        print(f"[ ChatNEWS ] changed llm : {llm}")
        self.llm = llm

    def set_retriever(self, retriever):
        print(f"[ ChatNEWS ] changed retriever : {retriever}")
        self.retriever = retriever
