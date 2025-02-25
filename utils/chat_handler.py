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
        print(f"🔍 질문시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

        print(f"[DEBUG] [handle_chat_response] 이전 대화 기록:\n{previous_context}")

        # ✅ 벡터DB에서 유사 질문 검색
        retrieved_context = search_similar_questions(internal_id, query)

        # ✅ 질의 생성 (이전 대화이력 및 벡터DB 검색 결과 반영)
        full_query = ""

        # 이전 대화이력 추가 보류
#         if previous_context:
#             full_query += f"""##########################################################
# 사용자의 이전 대화 기록:
# {previous_context}
# ##########################################################"""

        if retrieved_context:  # retrieved_context가 있을 때만 추가
            full_query += f"""########################################################## 
previous response context:
{retrieved_context}
##########################################################"""

        full_query += f"""#### 사용자 질문:
{query}"""
        result = assistant.ask(full_query)
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
