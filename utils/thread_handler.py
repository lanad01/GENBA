import os
import json
import uuid
import streamlit as st
from datetime import datetime
# ✅ 쓰레드 저장 경로 설정
THREADS_DB_PATH = "./threads"

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
            with open(os.path.join(THREADS_DB_PATH, file), "r", encoding="utf-8") as f:
                thread_data = json.load(f)
                threads.append({
                    "thread_id": thread_data["thread_id"],
                    "created_at": thread_data["created_at"],
                    "internal_id": thread_data["internal_id"]
                })

    return sorted(threads, key=lambda x: x["created_at"], reverse=True)  # 최신순 정렬

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
    print(f"🧵 [save_thread] 쓰레드 저장 시작 (messages: \n{response_data})")
    # print(f"🧵 [save_thread] 쓰레드 저장 시작 (type(response_data): {type(response_data)})")
    thread_path = os.path.join(THREADS_DB_PATH, f"{internal_id}.json")
    
    # 저장할 메시지 형식 정규화
    normalized_res = []
    request_summary = None
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
                    "insights": msg.get("insights"),
                    "report": msg.get("report")
                })
            request_summary = msg.get("request_summary", request_summary)
    
        normalized_res.append(normalized_msg)

    thread_data = {
        "thread_id": request_summary,
        "created_at": str(uuid.uuid4()),
        "internal_id": internal_id,
        "messages": normalized_res
    }

    with open(thread_path, "w", encoding="utf-8") as f:
        json.dump(thread_data, f, ensure_ascii=False, indent=2)
