import os
import json
import uuid
import streamlit as st
from datetime import datetime
from utils.vector_handler import delete_thread_vectorstore
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# ✅ 쓰레드 저장 경로 설정
THREADS_DB_PATH = "./threads"

uri = "mongodb+srv://swkwon:1q2w3e$r@cluster0.3rvbn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["chat_history"]
collection = db["conversations"]

def create_new_thread():
    """새로운 쓰레드 생성"""
    os.makedirs(THREADS_DB_PATH, exist_ok=True)

    login_id = "KSW"
    current_time = datetime.now().strftime("%Y%m%d_%H%M")  # 현재 시각을 문자열로 변환
    internal_id = f"temp_{login_id}_{current_time}"  # 임시 ID 생성
    # temp_thread_id = f"temp_{str(uuid.uuid4())}"  # ✅ 내부용 임시 ID
    thread_data = {
        "thread_id": "new_chat",  # ✅ UI 표시용 ID
        "internal_id": internal_id,  # ✅ 내부 관리용 ID
        "created_at": str(uuid.uuid4()),
        "messages": []
    }

    # ✅ JSON 파일로 저장
    thread_path = os.path.join(THREADS_DB_PATH, f"{internal_id}.json")
    with open(thread_path, "w", encoding="utf-8") as f:
        json.dump(thread_data, f, ensure_ascii=False, indent=2)

    return internal_id

def rename_thread(internal_id, request_summary):
    """첫 질문 이후 쓰레드 ID를 변경"""
    thread_path = os.path.join(THREADS_DB_PATH, f"{internal_id}.json")
    if os.path.exists(thread_path) :
        # 파일 내용도 업데이트
        with open(thread_path, "r", encoding="utf-8") as f:
            thread_data = json.load(f)
        
        thread_data["thread_id"] = request_summary  # ✅ UI 표시용 ID 업데이트
        thread_data.update({"thread_id": request_summary})
        
        with open(thread_path, "w", encoding="utf-8") as f:
            json.dump(thread_data, f, ensure_ascii=False, indent=2)
        
    return internal_id


def load_threads_list():
    """저장된 쓰레드 목록 불러오기"""
    os.makedirs(THREADS_DB_PATH, exist_ok=True)
    threads = []
    
    for file in os.listdir(THREADS_DB_PATH):
        if file.endswith(".json"):
            file_path = os.path.join(THREADS_DB_PATH, file)
            # 파일의 최종 수정 시간 가져오기
            modified_time = os.path.getmtime(file_path)
            
            with open(file_path, "r", encoding="utf-8") as f:
                thread_data = json.load(f)
                threads.append({
                    "thread_id": thread_data["thread_id"],
                    "created_at": thread_data["created_at"],
                    "internal_id": thread_data["internal_id"],
                    "modified_at": modified_time  # 파일 수정 시간 추가
                })

    # 파일 수정 시간 기준으로 정렬
    return sorted(threads, key=lambda x: x["modified_at"], reverse=True)

def load_thread(thread_id):
    """특정 쓰레드의 대화 기록 불러오기"""
    thread_path = os.path.join(THREADS_DB_PATH, f"{thread_id}.json")

    if os.path.exists(thread_path):
        with open(thread_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    return None

def sanitize_message_for_json(message):
    """메시지 객체를 JSON 직렬화 가능한 형태로 변환"""
    if isinstance(message, dict):
        sanitized = {}
        for key, value in message.items():
            if key == 'analytic_result':
                # DataFrame을 dict로 변환
                if isinstance(value, dict):
                    sanitized[key] = {
                        k: v.to_dict() if hasattr(v, 'to_dict') else str(v)
                        for k, v in value.items()
                    }
                else:
                    sanitized[key] = str(value)
            else:
                sanitized[key] = value
        return sanitized
    return message


def save_thread(internal_id, response_data):
    """특정 쓰레드의 대화 이력을 저장"""
    # print(f"🧵 [save_thread] 쓰레드 저장 시작 (세션: {internal_id})")
    # print(f"🧵 [save_thread] 쓰레드 저장 시작 (messages: \n{response_data})")
    # print(f"🧵 [save_thread] 쓰레드 저장 시작 (type(response_data): {type(response_data)})")
    thread_path = os.path.join(THREADS_DB_PATH, f"{internal_id}.json")
    
    # 저장할 메시지 형식 정규화
    normalized_res = []
    request_summary = "new_chat"  # 기본값으로 "new_chat" 설정
    
    for msg in response_data:
        # 튜플이나 None 등의 유효하지 않은 메시지 형식 건너뛰기
        if not isinstance(msg, dict):
            continue

        normalized_msg = {
            "role": msg["role"],
            "content": msg["content"]
        }
        
        # 분석 결과가 있는 경우만 추가 필드 저장
        if msg["role"] == "assistant":
            if "validated_code" in msg:
                normalized_msg.update({
                    "validated_code": msg.get("validated_code"),
                    "chart_filename": msg.get("chart_filename"),
                    # "analytic_results": msg.get("analytic_results"),
                    "insights": msg.get("insights"),
                    "report": msg.get("report")
                })

            # generated_code 필드 추가 (에러 발생 시에도 코드 유지)
            if "generated_code" in msg:
                normalized_msg["generated_code"] = msg.get("generated_code")
                
            # 에러 메시지 필드 추가
            if "error_message" in msg:
                normalized_msg["error_message"] = msg.get("error_message")
                
            # DataFrame 객체 처리
            if "analytic_result" in msg and msg["analytic_result"]:
                try:
                    analytic_result = msg["analytic_result"]
                    if isinstance(analytic_result, dict):
                        # 딕셔너리 내의 DataFrame 객체 처리
                        serialized_result = {}
                        for key, value in analytic_result.items():
                            if hasattr(value, 'to_dict'):  # DataFrame 확인
                                serialized_result[key] = value.to_dict('records')
                            else:
                                serialized_result[key] = value
                        normalized_msg["analytic_result"] = serialized_result
                    elif hasattr(analytic_result, 'to_dict'):  # DataFrame 확인
                        normalized_msg["analytic_result"] = analytic_result.to_dict('records')
                    else:
                        normalized_msg["analytic_result"] = analytic_result
                except Exception as e:
                    print(f"DataFrame 직렬화 중 오류: {e}")
                    normalized_msg["analytic_result"] = str(analytic_result)

            # request_summary가 "new_chat"일 때만 업데이트
            if request_summary == "new_chat" and "request_summary" in msg:
                request_summary = msg.get("request_summary")
    
        normalized_res.append(normalized_msg)
    
    # 현재 활성화된 마트 정보 저장
    active_marts = st.session_state.get("analysis_selected_data_marts", [])
    
    thread_data = {
        "thread_id": request_summary,
        "created_at": str(uuid.uuid4()),
        "internal_id": internal_id,
        "messages": normalized_res,
        "active_marts": active_marts  # 활성화된 마트 정보 추가
    }

    with open(thread_path, "w", encoding="utf-8") as f:
        json.dump(thread_data, f, ensure_ascii=False, indent=2)

def delete_thread(internal_id):
    """스레드 삭제"""
    try:
        # 1. 스레드 JSON 파일 삭제
        thread_path = os.path.join("./threads", f"{internal_id}.json")
        if os.path.exists(thread_path):
            os.remove(thread_path)
            print(f"✅ 스레드 JSON 파일 삭제 완료: {internal_id}")
            
        # 2. 벡터DB 삭제
        delete_thread_vectorstore(internal_id)
        
        # 3. MongoDB에서 스레드 데이터 삭제
        collection.delete_many({"internal_id": internal_id})
        print(f"✅ MongoDB 데이터 삭제 완료: {internal_id}")
        
        return True
    except Exception as e:
        print(f"❌ 스레드 삭제 중 오류 발생: {e}")
        return False
