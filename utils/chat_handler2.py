from datetime import datetime
import threading
import traceback
import streamlit as st
from typing import Dict, Any, Optional, Union
from langchain.memory import ConversationBufferMemory
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain.prompts import ChatPromptTemplate
# 사용자 패키지
from utils.vector_handler import save_chat_to_vector_db, search_similar_questions
from utils.thread_handler import rename_thread, save_thread
from common_txt import logo

# ✅ MongoDB Atlas 연결 설정
uri = "mongodb+srv://swkwon:1q2w3e$r@cluster0.3rvbn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["chat_history"]
collection = db["conversations"]


# ✅ 비동기 저장 함수 (MongoDB, Vector DB, Summarization을 별도 처리)
def save_chat_data(internal_id, query, response_data, llm):
    """
    UI를 먼저 업데이트한 후, 백그라운드에서 MongoDB 저장, 벡터DB 저장, Summarization을 수행하는 함수.
    """
    try:
        start_time = datetime.now()
        
        # content만 있는 경우와 분석 결과가 있는 경우를 구분
        if response_data.get('validated_code') or response_data.get('analytic_result'):
            base_text = f"""
validated_code: {response_data.get('validated_code', '')}
code_result: {response_data.get('analytic_result', '')}
insights: {response_data.get('insights', '')}
"""
        else:
            base_text = response_data.get('content', '')

        # ✅ 요약 적용 (대화가 일정 길이를 초과할 경우)
        if len(str(base_text).split()) < 300:  # 토큰 수가 적으면 요약 필요 없음
            print(f"💾 요약 필요 없음 | 토큰 수: {len(str(base_text).split())}")
            summary = base_text
        else :
            prompt = ChatPromptTemplate.from_messages([
            ("system", "다음 대화 및 분석 결과의 핵심 내용을 요약해주세요. 코드는 전부 포함하되, 결과 및 인사이트는 중요한 정보만 포함하고, 불필요한 내용은 제거하세요."),
            ("user", "{response_data}")
            ])
            chain = prompt | llm
            summary = chain.invoke({"response_data": base_text}).content.strip()

        # ✅ MongoDB 저장 (백그라운드에서 실행)
        collection.update_one(
            {"internal_id": internal_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {
                                "role": "user",
                                "content": query,
                                "timestamp": datetime.now()
                            },
                            {
                                "role": "assistant",
                                "content": response_data["content"],
                                "validated_code": response_data["validated_code"],
                                "generated_code": response_data["generated_code"],
                                "analytic_result": response_data["analytic_result"],
                                "chart_filename": response_data["chart_filename"],
                                "insights": response_data["insights"],
                                "report": response_data["report"],
                                "request_summary": response_data["request_summary"],
                                "context_summary": summary,  # ✅ 요약된 내용 추가
                                "timestamp": datetime.now(),
                            }
                        ]
                    }
                }
            },
            upsert=True
        )

        # ✅ 벡터DB 저장 (백그라운드에서 실행)
        save_chat_to_vector_db(internal_id, query, response_data)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"✅ 백그라운드 저장 완료 - 소요시간: {duration:.2f}초")

    except Exception as e:
        print(f"❌ 백그라운드 저장 중 오류 발생: {e}")
        print(f"상세 에러: {traceback.format_exc()}")


# ✅ 메모리 저장소 (thread_id별로 관리)
def get_history(thread_id: str) -> list:
    """
    특정 thread_id에 대한 대화 이력을 MongoDB에서 불러옴.
    query와 context_summary 쌍을 최근 7개만 반환
    """
    existing_messages = collection.find(
        {"internal_id": thread_id}
    ).sort("timestamp", -1).limit(7)
    
    conversation_pairs = []
    for document in existing_messages:
        messages = document.get("messages", [])
        
        # messages를 2개씩 묶어서 처리 (user + assistant 쌍)
        for i in range(0, len(messages)-1, 2):
            user_msg = messages[i]
            assistant_msg = messages[i+1]
            
            if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                conversation_pair = {
                    "query": user_msg["content"],
                    "response": assistant_msg.get("context_summary", assistant_msg["content"])
                }
                conversation_pairs.append(conversation_pair)
    
    # 최근 5개 쌍만 반환 (시간 역순)
    return conversation_pairs[:5]


# ✅ 백그라운드에서 저장을 실행하는 비동기 처리
def process_chat_response(assistant: Any, query: str, internal_id: str, start_from_analytics=False, feedback_point=None):
    """
    UI를 먼저 업데이트한 후, 백그라운드에서 MongoDB 저장, 벡터DB 저장, Summarization을 실행.
    """
    try:
        print("="*100)
        print(logo)
        print("="*100)
        print(f"🤵 질문시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🤵 Context Window 처리 시작")

        # ✅ 기존 대화 기록 가져오기
        context = get_history(thread_id=internal_id)
        print(f"🤵 컨텍스트 after get_history:\n{context}")
        result = assistant.ask(query, context, start_from_analytics=start_from_analytics, feedback_point=feedback_point)
        print(f"🤵 결과:\n{result}")

        # ✅ UI 렌더링을 위해 답변 결과(result)를 messages 리스트에 데이터 저장
        response_data = {
            "role": "assistant",
            "content": "분석이 완료되었습니다! 아래 결과를 확인해주세요.",  # 기본 응답 메시지 추가
            "validated_code": result.get("validated_code", {}),
            "generated_code": result.get("generated_code", {}),
            "analytic_result": result.get("analytic_result", {}),
            "chart_filename": result.get("chart_filename", {}),
            "insights": result.get("insights", {}),
            "report": result.get("report", {}),
            "request_summary": result.get("request_summary", {}),
            "feedback_point": result.get("feedback_point", {}),
        }

        # ✅ 메인 답변 처리
        if "analytic_result" not in result:
            if "error_message" in result:
                response_data["content"] = result["error_message"]
                # st.error(f"⚠️ 오류 발생: {result['error_message']}")
                pass
            else:
                response_data["content"] = (
                    result.get("general_response") or 
                    result.get("knowledge_response") or 
                    result.get("response", "응답을 생성할 수 없습니다.")
                )

        # 스레드 생성 및 시작
        save_thread = threading.Thread(
            target=save_chat_data, 
            args=(internal_id, query, response_data, assistant.llm), 
            daemon=True,
            name=f"SaveThread-{internal_id}"  # 스레드에 식별 가능한 이름 부여
        )
        save_thread.start()
        print(f"🔄 백그라운드 저장 스레드 시작 - {save_thread.name}")

        return response_data

    except Exception as e:
        print(f"❌ 오류 발생: {traceback.format_exc()}")
        # st.error(f"❌ 오류 발생 [handle_chat_response] : {traceback.format_exc()}")
        return { 
            "role": "assistant", 
            "content": f"❌ 실행 도중 오류가 발생했습니다.", 
            "error_message" : str(e),
            "generated_code" : response_data['generated_code'],
        } 