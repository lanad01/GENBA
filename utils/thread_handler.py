import os
import json
import uuid
import streamlit as st
from datetime import datetime
from utils.vector_handler import delete_thread_vectorstore
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from typing import Optional
import pandas as pd

# ✅ 쓰레드 저장 경로 설정
THREADS_DB_PATH = "./threads"

login_id = "KSW"
uri = "mongodb+srv://swkwon:1q2w3e$r@cluster0.3rvbn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["chat_history"]
collection = db["conversations"]

def create_new_thread():
    """새로운 쓰레드 생성"""
    os.makedirs(THREADS_DB_PATH, exist_ok=True)

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
            thread_data = json.load(f)
            
            # 메시지 내의 Dict to DataFrame 
            if "messages" in thread_data:
                for message in thread_data["messages"]:
                    if "analytic_result" in message:
                        for key, value in message["analytic_result"].items():
                            if isinstance(value, dict) and value.get('type') == 'dataframe' and 'data' in value:
                                # DataFrame으로 변환
                                try:
                                    message["analytic_result"][key] = pd.DataFrame(value['data'])
                                except Exception as e:
                                    print(f"DataFrame 변환 오류: {e}")
            
            return thread_data
    
    return None

def sanitize_message_for_json(result):
    """메시지 객체를 JSON 직렬화 가능한 형태로 변환"""
    LIMIT_SIZE = 100
    
    def limit_dict_size(d, max_items=LIMIT_SIZE):
        """딕셔너리 크기를 제한하는 헬퍼 함수"""
        if isinstance(d, dict):
            return dict(list(d.items())[:max_items])
        return d

    def convert_timestamp(obj):
        """Timestamp 객체를 문자열로 변환하는 헬퍼 함수"""
        if hasattr(obj, 'timestamp'):  # Timestamp 객체 확인
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return obj

    if isinstance(result, dict):
        sanitized = {}
        
        for key, value in result.items():
            # Timestamp 키와 값을 한 번에 처리
            value = convert_timestamp(value)
            
            # DataFrame이나 복잡한 객체를 포함할 수 있는 analytic_result 처리
            if isinstance(value, dict):
                sanitized[key] = {}
                # 최상위 딕셔너리 크기 제한
                limited_value = limit_dict_size(value)
                for k, v in limited_value.items():
                    v = convert_timestamp(v)  # 중첩된 Timestamp 값 처리
                    if hasattr(v, 'to_dict'):  # DataFrame인 경우
                        # DataFrame 정보 저장 (타입 정보 포함)
                        sanitized[key][k] = {
                            'type': 'dataframe',
                            'data': v.head(LIMIT_SIZE).to_dict('records')
                        }
                    elif isinstance(v, (list, dict)):  # 중첩된 구조체인 경우
                        try:
                            if isinstance(v, dict):
                                sanitized[key][k] = limit_dict_size(v)
                            elif isinstance(v, list):
                                sanitized[key][k] = [convert_timestamp(item) for item in v[:LIMIT_SIZE]]  # 리스트도 50개로 제한
                            else:
                                sanitized[key][k] = v
                        except:
                            sanitized[key][k] = str(v)
                    else:
                        sanitized[key][k] = v  # 원시 타입은 그대로 저장
            elif hasattr(value, 'to_dict'):  # 최상위 레벨의 DataFrame 처리
                sanitized[key] = {
                    'type': 'dataframe',
                    'data': value.head(LIMIT_SIZE).to_dict('records')
                }
            else:
                sanitized[key] = value  # 기타 타입은 그대로 저장
        return sanitized
    return result

def save_thread(internal_id, response_data):
    """특정 쓰레드의 대화 이력을 저장"""
    # print(f"🧵 [save_thread] 쓰레드 저장 시작 (세션: {internal_id})")
    # print(f"🧵 [save_thread] 쓰레드 저장 시작 (messages: \n{response_data})")
    thread_path = os.path.join(THREADS_DB_PATH, f"{internal_id}.json")
    
    # 저장할 메시지 형식 정규화
    normalized_res = []
    request_summary = "new_chat"  # 기본값으로 "new_chat" 설정
    question_count = 0  # 질문 카운터 초기화

    for msg in response_data:
        # 튜플이나 None 등의 유효하지 않은 메시지 형식 건너뛰기
        if not isinstance(msg, dict):
            continue
        
        # 사용자 메시지인 경우 질문 번호 증가
        question_count += 1
        question_id = f"{internal_id}_{question_count}"

        normalized_msg = {
            "role": msg["role"],
            "content": msg["content"],
            "question_id": question_id  # 질문 ID 추가
        }
        
        # 분석 결과가 있는 경우만 추가 필드 저장
        if msg["role"] == "assistant":
            if "validated_code" in msg:
                additional_fields = {
                    "validated_code": msg.get("validated_code"),
                    "chart_filename": msg.get("chart_filename"),
                    "analytic_result": msg.get("analytic_result"),
                    "insights": msg.get("insights"),
                    "report": msg.get("report"),
                    "feedback": msg.get("feedback"),
                    "feedback_point": msg.get("feedback_point")
                }
                normalized_msg.update(additional_fields)
                
                # JSON 직렬화를 위해 메시지 정규화
                if "analytic_result" in normalized_msg:
                    normalized_msg["analytic_result"] = sanitize_message_for_json(normalized_msg["analytic_result"])
                
            # generated_code 필드 추가 (에러 발생 시에도 코드 유지)
            if "generated_code" in msg: normalized_msg["generated_code"] = msg.get("generated_code")
                
            # 에러 메시지 필드 추가
            if "error_message" in msg: normalized_msg["error_message"] = msg.get("error_message")

            # request_summary가 "new_chat"일 때만 업데이트
            if request_summary == "new_chat" and "request_summary" in msg:
                request_summary = msg.get("request_summary")
    
        normalized_res.append(normalized_msg)
    
    # 현재 활성화된 마트 정보 저장
    active_marts = st.session_state.get("analysis_selected_data_marts", [])
    
    thread_data = {
        "login_id": login_id,
        "thread_id": request_summary,
        "created_at": str(uuid.uuid4()),
        "internal_id": internal_id,
        "active_marts": active_marts,  # 활성화된 마트 정보 추가
        "question_count": question_count,  # 총 질문 수 추가
        "messages": normalized_res,
    }
    # print(f"🧵 [save_thread] 쓰레드 저장 직전 (thread_data: \n{thread_data['messages']})")

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

def get_parent_message(internal_id: str, question_id: str) -> Optional[dict]:
    """
    주어진 question_id 해당하는 메시지의 분석 정보를 가져옵니다.
    
    Args:
        internal_id (str): 스레드 내부 ID
        question_id (str): 부모 질문 ID
    
    Returns:
        Optional[dict]: 부모 메시지의 분석 정보를 담은 딕셔너리 또는 None
    """
    parent_message = None
    thread_data = load_thread(internal_id)
    
    if thread_data and "messages" in thread_data:
        for msg in thread_data["messages"]:
            if msg.get("question_id") == question_id:
                if msg.get("role") == "user":
                    parent_message = {'query': msg.get('content', '')}
                elif msg.get("role") == "assistant":
                    parent_message = {
                        'validated_code': msg.get('validated_code', ''),
                        'analytic_result': msg.get('analytic_result', ''),
                        'insights': msg.get('insights', ''),
                        'report': msg.get('report', ''),
                        'feedback': msg.get('feedback', ''),
                        'chart_filename': msg.get('chart_filename', '')
                    }
    
                    return parent_message
                else:
                    return None  # 부모 메시지가 없는 경우 None 반환
