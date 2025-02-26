from datetime import datetime
import traceback
import streamlit as st
from typing import Dict, Any, Optional, Union
from langchain.memory import ConversationBufferMemory
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# 사용자 패키지
from utils.vector_handler import save_chat_to_vector_db, search_similar_questions
from utils.thread_handler import rename_thread, save_thread
# ✅ MongoDB Atlas 연결 설정
uri = "mongodb+srv://swkwon:1q2w3e$r@cluster0.3rvbn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["chat_history"]
collection = db["conversations"]

# ✅ 메모리 저장소 (thread_id별로 관리)
memory_store = {}

def get_memory(thread_id: str) -> ConversationBufferMemory:
    """
    특정 thread_id에 대한 ConversationBufferMemory를 반환.
    기존 데이터가 있으면 불러오고, 없으면 새로 생성.
    """
    if thread_id not in memory_store:
        memory_store[thread_id] = ConversationBufferMemory(memory_key=f"history_{thread_id}", return_messages=True)
    
        # ✅ MongoDB에서 이전 대화 기록 불러오기
        existing_messages = collection.find({"internal_id": thread_id}).sort("timestamp", 1)  # 시간순 정렬
        if existing_messages:
            for document in existing_messages:
                for msg in document.get("messages", []):
                    if msg["role"] == "user":
                        memory_store[thread_id].chat_memory.add_user_message(msg["content"])
                    elif msg["role"] == "assistant":
                        memory_store[thread_id].chat_memory.add_ai_message(msg["content"])

    return memory_store[thread_id]

# ✅ 채팅 응답 처리
def handle_chat_response(
    assistant: Any,
    query: str,
    internal_id: str
) -> tuple[Optional[Dict[str, Any]], ConversationBufferMemory]:
    """
    채팅 응답 처리 (기존 질문과 AI 응답을 기억)
    
    Args:
        assistant: 챗봇 어시스턴트 인스턴스
        query: 사용자 질문
        thread_id: 스레드 ID

    Returns:
        tuple[응답 데이터 딕셔너리, 업데이트된 메모리 객체]
    """
    try:
        print(f"🕛 [handle_chat_response] 질문시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ✅ thread_id별 memory 가져오기
        memory = get_memory(internal_id)

        # ✅ 기존 대화 기록 가져오기
        messages = memory.load_memory_variables({}).get(f"history_{internal_id}", "")

        # 메시지 객체를 읽기 쉬운 대화 형식으로 변환
        previous_context = ""
        if messages:
            for msg in messages:
                if msg.type == 'human':
                    previous_context += f"사용자: {msg.content}\n"
                elif msg.type == 'ai':
                    previous_context += f"어시스턴트: {msg.content}\n"

        ##########################################################################################
        # ✅ Context Window 처리
        # ** 해당 쓰레드의 질문-답변 이력이 쌓여있는 벡터DB에서 사용자의 현재 질문과 유사한 질문 검색
        ##########################################################################################
        model = assistant.llm
        filtered_results = search_similar_questions(internal_id, query)
        if not filtered_results:
            return ""

        document_texts = "\n\n".join([
            f"[유사도: {score:.2f}]\n{doc.page_content}" 
            for doc, score in filtered_results
        ])

        # LLM을 사용하여 문서 요약
        prompt = f"""
다음은 이전 대화 내역에서 현재 질문 "{query}"와 관련성이 높은 검색된 문서들입니다.
이를 참고하여 현재 질문과 직접 연결되는 핵심 내용을 요약하세요.

1. 질문과 직접적으로 연결되는 정보만 남기고 불필요한 내용은 제거
2. 검색된 문서에서 코드가 있다면 그대로 유지
3. 분석 결과나 중요한 인사이트는 정리해서 포함
4. 정보를 다음 형식으로 구조화:
    - 핵심 개념 설명
    - 관련 코드
    - 주요 인사이트(3줄 이내)

{document_texts}
        """
        summarized_result = model.invoke(prompt).content.strip()
        
        ##########################################################################################
        # ✅ Query Rewriting
        #  """ 기존 문맥과 검색된 문서를 반영하여 새로운 질문을 생성 """
        ##########################################################################################
        # 이전 대화 기록이 없고 검색된 문서도 없으면 원본 질문 그대로 사용
        if not previous_context or not summarized_result:
            final_query = query  
        # 이전 대화 기록이 있고 검색된 문서도 있으면 검색된 문서를 반영하여 새로운 질문을 생성
        else:
            prompt = f"""
다음은 사용자의 이전 대화 기록을 요약한 것입니다.
이를 참고하여 사용자 질문을 보다 명확하고 완전한 질문으로 변환하세요.

[검색된 문서 요약]
{summarized_result}

[사용자의 원래 질문]
{query}

[재구성된 질문]
            """
            final_query = model.invoke(prompt).content.strip()

        print(f"🤵 재구성된 질문:\n{final_query}")
        result = assistant.ask(final_query)
        print(f"🤵 결과:\n{result}")
        
        # ✅ UI 렌더링을 위해 답변 결과(result)를 messages 리스트에 데이터 저장
        response_data = {
            "role": "assistant",
            "content": "분석이 완료되었습니다! 아래 결과를 확인해주세요.",  # 기본 응답 메시지 추가
            "validated_code": result.get("validated_code"),
            "analytic_result": result.get("analytic_result"),
            "chart_filename": result.get("chart_filename"),
            "insights": result.get("insights"),
            "report": result.get("report"),
            "request_summary": result.get("request_summary"),
        }

        # ✅ 일반 텍스트 응답일 경우 처리
        if "analytic_result" not in result:
            if "error_message" in result:
                response_data["content"] = result["error_message"]
                st.error(f"⚠️ 오류 발생: {result['error_message']}")
            else:
                response_data["content"] = (
                    result.get("general_response") or 
                    result.get("knowledge_response") or 
                    result.get("response", "응답을 생성할 수 없습니다.")
                )

        # ✅ 메모리에 질문 및 응답 저장 (에러가 없는 경우만)
        if "error_message" not in response_data:
            if isinstance(query, str):
                memory.chat_memory.add_user_message(query)
            if isinstance(response_data["content"], str):
                memory.chat_memory.add_ai_message(response_data["content"])

            # ✅ MongoDB에도 대화 이력 저장
            collection.update_one(
                {"internal_id": internal_id},
                {
                    "$push": {
                        "messages": {
                            "$each": [
                                {"role": "user", "content": query, "timestamp": datetime.now()},
                                {"role": "assistant", "content": response_data["content"], "timestamp": datetime.now()}
                            ]
                        }
                    }
                },
                upsert=True
            )
            
            # ✅ 응답에서 request_summary 확인 및 thread_id 변경
            if "request_summary" in response_data:
                rename_thread(internal_id, response_data["request_summary"])

            # ✅ 벡터DB 저장
            save_chat_to_vector_db(internal_id, query, response_data)

        return response_data

    except Exception as e:
        st.error(f"❌ 오류 발생 [handle_chat_response] : {traceback.format_exc()}")
        return None, None
