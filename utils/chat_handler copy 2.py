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
from common_txt import logo

# ✅ MongoDB Atlas 연결 설정
uri = "mongodb+srv://swkwon:1q2w3e$r@cluster0.3rvbn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["chat_history"]
collection = db["conversations"]

# ✅ 메모리 저장소 (thread_id별로 관리)
def get_history(thread_id: str) -> list:
    """
    특정 thread_id에 대한 대화 이력을 MongoDB에서 불러옴.
    """
    existing_messages = collection.find({"internal_id": thread_id}).sort("timestamp", -1).limit(5)  
    messages = []
    for document in existing_messages:
        messages.extend(document.get("messages", []))
    return messages


# ✅ 채팅 응답 처리
def handle_chat_response(
    assistant: Any,
    query: str,
    internal_id: str
) -> tuple[Optional[Dict[str, Any]]]:
    """
    채팅 응답 처리 (기존 질문과 AI 응답을 기억)
    
    Args:
        assistant: 챗봇 어시스턴트 인스턴스
        query: 사용자 질문
        thread_id: 스레드 ID

    Returns:
        tuple[응답 데이터 딕셔너리]
    """
    try:
        print("="*100)
        print(logo)
        print("="*100)
        print(f"🤵 질문시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🤵 Context Window 처리 시작")

        # ✅ thread_id별 memory 가져오기
        chat_history = get_history(internal_id)

        # ✅ 기존 대화 기록 가져오기
        previous_context = ""
        for msg in chat_history:
            if msg["role"] == "user":
                previous_context += f"사용자: {msg['content']}\n"
            elif msg["role"] == "assistant":
                previous_context += f"어시스턴트: {msg['content']}\n"

        ##########################################################################################
        # ✅ Context Window 처리
        # ** 해당 쓰레드의 질문-답변 이력이 쌓여있는 벡터DB에서 사용자의 현재 질문과 유사한 질문 검색
        ##########################################################################################
        model = assistant.llm
        filtered_results = search_similar_questions(internal_id, query)
        
        # 검색 결과가 있을 때만 document_texts 생성
        if filtered_results:
            document_texts = "\n\n".join([
                f"[유사도: {score:.2f}, 코사인 유사도: {cosine_sim:.2f}]\n{doc.page_content}" 
                for doc, score, cosine_sim in filtered_results
            ])

            prompt = f"""
# 🔍 문서 요약 및 관련성 평가
아래 **검색된 문서들**은 사용자 질문과 관련이 있을 가능성이 있는 정보입니다.  
하지만, 모든 문서가 현재 질문과 100% 관련이 있는 것은 아닙니다.  
따라서 **문서의 내용을 평가한 후** 직접적인 관련이 있는 정보만 선별하여 요약하세요.  

## ✅ 요약 프로세스
1. **문서와 질문의 연관성 평가:**  
- 각 문서의 내용을 질문과 비교하여 **직접적인 답변을 제공할 수 있는지** 평가  
- 관련성이 **낮은 문서(예: 질문과 무관한 개념 설명)**는 요약에서 제외  
2. **핵심 내용 요약:**  
- 질문의 핵심 주제와 연관된 정보만 유지  
- 코드가 포함된 경우, 해당 코드가 어떻게 질문과 관련 있는지 설명  
3. **결과 구조화:**  
- 핵심 개념 설명  
- 관련 코드 (있는 경우)  
- 주요 인사이트 (3줄 이내)  

## 🚀 현재 질문
"{query}"

## 📄 검색된 문서
{document_texts}

## 🎯 최종 요약 (질문과 직접 연결된 내용만 유지)
"""

            summarized_result = model.invoke(prompt).content.strip()
            print(f"🤵 검색된 문서 요약:\n{summarized_result}")
        else:
            # 검색 결과가 없을 경우
            summarized_result = ""
            print("🤵 유사한 질문이 없습니다.")

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
# 🔄 질문 재구성 및 문맥 보강
당신은 사용자의 대화 흐름을 유지하면서도 문맥을 강화하는 AI 비서입니다.  
아래 정보를 바탕으로 **질문을 자연스럽게 보강**하세요.  

## 🎯 재구성 목표
1. 원래 질문의 핵심 의도를 유지  
2. 검색된 문서 요약에서 **질문과 직접 관련된 정보만 반영**  
3. 코드가 포함된 경우, **질문을 코드 활용과 연관되도록 보강**  
4. 불필요한 내용 추가 금지 (질문이 지나치게 확장되지 않도록 주의)  

## 📄 검색된 문서 요약
{summarized_result}

## 🤔 원래 질문
"{query}"

## ✅ 보강된 질문 (원래 질문을 유지하면서 문맥 강화)
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

        # ✅ 메인 답변 처리
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

        # ✅ MongoDB 대화 이력 저장
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
                                "chart_filename": response_data["chart_filename"],
                                "insights": response_data["insights"],
                                "report": response_data["report"],
                                "request_summary": response_data["request_summary"],
                                "timestamp": datetime.now(),
                            }   
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
        return { "role": "assistant", "content": f"❌ 오류 발생 [handle_chat_response] : {traceback.format_exc()}" } 