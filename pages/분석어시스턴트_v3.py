# ✅ 기본 라이브러리
from datetime import datetime
import json
import random
import os, sys, re
import traceback
import pandas as pd
import time
from glob import glob
from pathlib import Path

# ✅ 내부 유틸리티 모듈
from utils.vector_handler import get_text, get_text_chunks, load_vectorstore, load_document_list, save_document_list, get_vectorstore, rebuild_vectorstore_without_document    
from utils.pages_handler import get_available_marts, load_selected_mart, get_page_state, set_page_state
from utils.chat_handler import process_chat_response
from utils.thread_handler import load_threads_list, create_new_thread, save_thread, load_thread, rename_thread, delete_thread, get_parent_message
from utils.analytic_agent import DataAnayticsAssistant
from pages.styles.styles import apply_custom_styles

# ✅ 3자 패키지
import streamlit as st
import pyautogui

# 상수 정의
PAGE_NAME = "analysis"
ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_PATH = "../output/stage1/processed_data_info.xlsx"
DOCUMENT_LIST_PATH = str(ROOT_DIR / "documents" / PAGE_NAME)
VECTOR_DB_ANSS_PATH = str(ROOT_DIR / "src" / "vectordb" / PAGE_NAME)
CONSTANTS = {
    "PAGE_TITLE": "분석 어시스턴트",
    "PAGE_ICON": "🔍",
    "ASSISTANT_MESSAGE": "안녕하세요! AI 분석 어시스턴트입니다. 무엇이든 물어보세요!"
}
# print(f"🔢 [ 분석어시스턴트 ROOT_DIR ] : {ROOT_DIR}")

# ✅ 세션 상태 초기화 함수 수정
def initialize_session_state():
    """세션 상태 초기화"""
    internal_id = get_page_state(PAGE_NAME, "internal_id")
    
    # 스레드별 상태 초기화 (없는 경우에만)
    if internal_id:
        if f"{internal_id}_show_mart_manager" not in st.session_state:
            st.session_state[f"{internal_id}_show_mart_manager"] = False
        if f"{internal_id}_selected_data_marts" not in st.session_state:
            st.session_state[f"{internal_id}_selected_data_marts"] = []
        if f"{internal_id}_loaded_mart_data" not in st.session_state:
            st.session_state[f"{internal_id}_loaded_mart_data"] = {}
        
    # OpenAI API Key 검증
    if not (openai_api_key := os.getenv('OPENAI_API_KEY')):
        st.warning("⚠️ OpenAI API Key가 설정되지 않았습니다. 환경 변수를 확인하세요.")
        return
    
    page_session_state = {key: value for key, value in st.session_state.items() if key.startswith(PAGE_NAME)}
    # print(f"🔢[BA init] {get_page_state(PAGE_NAME, 'internal_id')} | show_mart : {st.session_state.get(f'{internal_id}_show_mart_manager', False)} | selected_mart : {st.session_state.get(f'{internal_id}_selected_data_marts', [] )} ")

    # AI Assistant 초기화
    if not get_page_state(PAGE_NAME, "assistant"):
        with st.spinner("🤖 AI Agent를 로드하는 중..."):
            assistant = DataAnayticsAssistant(openai_api_key)
            set_page_state(PAGE_NAME, "assistant", assistant)

    initial_states = {
        "show_popover": True,
        "messages": [{"role": "assistant", "content": "안녕하세요! AI 분석 어시스턴트입니다. 무엇이든 물어보세요!"}]
    }
    
    for key, value in initial_states.items():
        if not get_page_state(PAGE_NAME, key):
            set_page_state(PAGE_NAME, key, value)


def load_mart_data() :
    """마트 데이터 로드"""
    internal_id = get_page_state(PAGE_NAME, "internal_id")
            
    # 질의 전 마트 상태 확인 및 복원
    thread_path = os.path.join("./threads", f"{internal_id}.json")
    if os.path.exists(thread_path):
        with open(thread_path, "r", encoding="utf-8") as f:
            thread_data = json.load(f)
        
        # 스레드에 저장된 활성 마트 목록
        saved_marts = thread_data.get("active_marts", [])
        current_marts = set(get_page_state(PAGE_NAME, "selected_data_marts", []))
        
        # 활성화되어야 할 마트가 있다면 자동 활성화
        if saved_marts and not current_marts:
            loaded_mart_data = {}
            for mart in saved_marts:
                data = load_selected_mart(mart)
                if data is not None:
                    loaded_mart_data[mart] = data
                    current_marts.add(mart)
            
            # 마트 상태 업데이트
            if loaded_mart_data:
                set_page_state(PAGE_NAME, "loaded_mart_data", loaded_mart_data)
                set_page_state(PAGE_NAME, "selected_data_marts", list(current_marts))
                
                # AI Assistant 마트 상태 업데이트
                assistant = get_page_state(PAGE_NAME, "assistant")
                if assistant:
                    assistant.set_active_mart(
                        data_mart=loaded_mart_data,
                        mart_name=list(loaded_mart_data.keys())
                    )
                    set_page_state(PAGE_NAME, "assistant", assistant)


def close_popover():
    """Popover 닫기"""
    st.session_state["show_popover"] = False
    pyautogui.hotkey("esc")

# ✅ 마트 선택/해제 처리
def handle_mart_selection(mart):
    """마트 선택/해제 처리"""
    internal_id = get_page_state(PAGE_NAME, "internal_id")
    current_marts = set(st.session_state[f"{internal_id}_selected_data_marts"])
    loaded_mart_data = st.session_state[f"{internal_id}_loaded_mart_data"]

    if mart in current_marts:
        current_marts.remove(mart)
        if mart in loaded_mart_data:
            del loaded_mart_data[mart]
            print(f"🗑️ 마트 비활성화: {mart}")
            
            # Assistant 마트 상태 업데이트
            assistant = get_page_state(PAGE_NAME, "assistant")
            if assistant:
                if loaded_mart_data:
                    # 남은 마트들로 업데이트
                    assistant.set_active_mart(
                        data_mart=loaded_mart_data,
                        mart_name=list(loaded_mart_data.keys())
                    )
                else:
                    # 모든 마트가 비활성화된 경우
                    print("🗑️ 모든 마트가 비활성화되었습니다.")
                    assistant.active_marts = None
                    assistant.mart_info = None
                set_page_state(PAGE_NAME, "assistant", assistant)
    else:
        ##############################################################
        # 마트 선택 및 활성화
        ##############################################################
        current_marts.add(mart)
        data = load_selected_mart(mart)
        if data is not None:
            loaded_mart_data[mart] = data
            print(f"✅ 마트 활성화 클릭: {mart} (shape: {data.shape})")
            
            # AI Assistant에 활성화된 마트 설정
            assistant = get_page_state(PAGE_NAME, "assistant")
            if assistant:
                assistant.set_active_mart(
                    data_mart=loaded_mart_data,
                    mart_name=list(loaded_mart_data.keys())
                )
                set_page_state(PAGE_NAME, "assistant", assistant)
        else:
            print(f"❌ 마트 로드 실패: {mart}")
            
    # 현재 쓰레드가 있는 경우, 마트 상태 업데이트
    if internal_id:
        thread_path = os.path.join("./threads", f"{internal_id}.json")
        if os.path.exists(thread_path):
            with open(thread_path, "r", encoding="utf-8") as f:
                thread_data = json.load(f)
            
            # 마트 상태 업데이트
            thread_data["active_marts"] = list(current_marts)
            
            with open(thread_path, "w", encoding="utf-8") as f:
                json.dump(thread_data, f, ensure_ascii=False, indent=2)
    
    if loaded_mart_data:
        print(f"총 활성화된 마트 수: {len(loaded_mart_data)} | 활성화된 마트: {list(loaded_mart_data.keys())}\n")
    
    st.session_state[f"{internal_id}_selected_data_marts"] = list(current_marts)
    st.session_state[f"{internal_id}_loaded_mart_data"] = loaded_mart_data
    
    # 마트 선택이 변경된 경우에만 rerun
    if mart in current_marts or mart in loaded_mart_data:
        st.rerun()
    
    
@st.fragment
def render_mart_selector():
    """마트 선택 UI 렌더링"""
    internal_id = get_page_state(PAGE_NAME, "internal_id")
    show_mart_manager = st.session_state.get(f"{internal_id}_show_mart_manager", False)

    with st.container():
        left_content, middle_content, right_content = st.columns([0.3, 0.3, 0.4])
        
        # 마트 목록 표시 버튼(on & off)
        with left_content:
            if st.button(
                "📊 마트 ↓" if not show_mart_manager else "📊 마트 ↑",
                use_container_width=True
            ):
                
                # internal_id가 없는 경우 새 쓰레드 생성
                if not internal_id:
                    new_thread_id = create_new_thread()
                    internal_id = new_thread_id
                    # 새 스레드의 상태 초기화
                    st.session_state[f"{new_thread_id}_show_mart_manager"] = False
                    st.session_state[f"{new_thread_id}_selected_data_marts"] = []
                    st.session_state[f"{new_thread_id}_loaded_mart_data"] = {}
                    set_page_state(PAGE_NAME, "internal_id", new_thread_id)
                    set_page_state(PAGE_NAME, "messages", [{"role": "assistant", "content": CONSTANTS["ASSISTANT_MESSAGE"]}])
                
                # 마트 매니저를 열 때 저장된 마트 상태 확인 및 복원
                if not show_mart_manager:  # 닫힌 상태에서 열 때만 복원
                    thread_path = os.path.join("./threads", f"{internal_id}.json")
                    if os.path.exists(thread_path):
                        with open(thread_path, "r", encoding="utf-8") as f:
                            thread_data = json.load(f)
                            saved_marts = thread_data.get("active_marts", [])
                            
                            # 저장된 마트가 있다면 상태 복원
                            if saved_marts:
                                loaded_mart_data = {}
                                for mart in saved_marts:
                                    data = load_selected_mart(mart)
                                    if data is not None:
                                        loaded_mart_data[mart] = data
                                
                                # 세션 상태 업데이트
                                st.session_state[f"{internal_id}_selected_data_marts"] = saved_marts
                                st.session_state[f"{internal_id}_loaded_mart_data"] = loaded_mart_data
                                
                                # AI Assistant 마트 상태 업데이트
                                assistant = get_page_state(PAGE_NAME, "assistant")
                                if assistant and loaded_mart_data:
                                    assistant.set_active_mart(
                                        data_mart=loaded_mart_data,
                                        mart_name=list(loaded_mart_data.keys())
                                    )
                                    set_page_state(PAGE_NAME, "assistant", assistant)
                
                st.session_state[f"{internal_id}_show_mart_manager"] = not show_mart_manager
                st.rerun()

        # API Key 상태 표시
        with right_content:
            st.markdown(
                """
                <div style='float: right;'>
                    <span style='background-color: #E8F0FE; padding: 5px 10px; border-radius: 5px;'>
                        ✅ API Key
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 마트 선택
    if show_mart_manager:
        with st.container():
            tab1, tab2 = st.tabs(["📊 마트 선택", "📈 활성화된 마트"])
            
            with tab1:
                available_marts = get_available_marts()
                if not available_marts:
                    st.warning("사용 가능한 데이터 마트가 없습니다.")
                else:
                    cols = st.columns(3)
                    for i, mart in enumerate(available_marts):
                        with cols[i % 3]:
                            
                            is_selected = mart in st.session_state[f"{internal_id}_selected_data_marts"]
                            if st.button(
                                f"{'✅' if is_selected else '⬜'} {mart}",
                                key=f"mart_{mart}",
                                use_container_width=True,
                                type="primary" if is_selected else "secondary"
                            ):
                                handle_mart_selection(mart)
                                st.rerun()
            
            with tab2:
                if not st.session_state[f"{internal_id}_selected_data_marts"]:
                    st.info("활성화된 마트가 없습니다.")
                else:
                    for mart in st.session_state[f"{internal_id}_selected_data_marts"]:
                        with st.expander(f"📊 {mart}", expanded=True):
                            if mart in st.session_state[f"{internal_id}_loaded_mart_data"]:
                                data = st.session_state[f"{internal_id}_loaded_mart_data"][mart]
                                st.text(f"""Row : {data.shape[0]:,} | Column : {data.shape[1]}""")
                                st.dataframe(data.head(5), use_container_width=True, )

# sidebar 채팅 관리
def render_sidebar_chat():
    # st.sidebar.subheader("📚 대화 관리")

    # ✅ 새로운 쓰레드 생성 버튼
    if st.sidebar.button("새 대화 시작 ✙", use_container_width=True):
        new_thread_id = create_new_thread()
        # 새 스레드의 상태 초기화
        st.session_state[f"{new_thread_id}_show_mart_manager"] = False
        st.session_state[f"{new_thread_id}_selected_data_marts"] = []
        st.session_state[f"{new_thread_id}_loaded_mart_data"] = {}
        
        set_page_state(PAGE_NAME, "internal_id", new_thread_id)
        set_page_state(PAGE_NAME, "messages", [])
        st.rerun()

    # ✅ 저장된 쓰레드 목록 표시
    st.sidebar.markdown("#### 📝 기존 대화 목록")
    threads = load_threads_list()
    current_thread_id = get_page_state(PAGE_NAME, "internal_id") # 현재 활성화된 thread의 internal_id

    for idx, thread in enumerate(threads):
        # 현재 활성화된 thread인지 확인
        is_active = thread.get('internal_id') == current_thread_id
        
        button_icon = "🔵" if is_active else "💬"
        
        # 쓰레드 목록 및 삭제 버튼
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        with col1:
            if st.button(
                f"{button_icon} {thread['thread_id']}", 
                key=f"thread_{thread['created_at']}",
                help=f"ID: {thread.get('internal_id', '없음')}",
                use_container_width=True,
            ):
                thread_id = thread["internal_id"]
                
                # 현재 활성화된 스레드와 동일한 스레드 클릭 시 무시
                if thread_id == current_thread_id:
                    continue
                
                # 스레드 전환 시 해당 스레드의 상태 초기화 (없는 경우에만)
                if f"{thread_id}_selected_data_marts" not in st.session_state:
                    st.session_state[f"{thread_id}_selected_data_marts"] = []
                if f"{thread_id}_loaded_mart_data" not in st.session_state:
                    st.session_state[f"{thread_id}_loaded_mart_data"] = {}
                
                # 다른 쓰레드로 전환 
                st.session_state[f"{thread_id}_show_mart_manager"] = False # 마트 선택 UI 닫기
                set_page_state(PAGE_NAME, "internal_id", thread_id)
                loaded_thread = load_thread(thread["internal_id"])
                if loaded_thread and "messages" in loaded_thread:
                    # 메시지 로드 시 DataFrame 객체 처리
                    messages = loaded_thread["messages"]
 
                    for message in messages:
                        # print(f"🔢 [render_sidebar_chat] 스레드 전환: {message['analytic_result']}")
                        if "analytic_result" in message and message["analytic_result"]:
                            # 문자열로 저장된 DataFrame을 다시 DataFrame으로 변환
                            try:
                                if isinstance(message["analytic_result"], dict):
                                    for key, value in message["analytic_result"].items():
                                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                                            message["analytic_result"][key] = pd.DataFrame(value)
                                elif isinstance(message["analytic_result"], list) and len(message["analytic_result"]) > 0:
                                    message["analytic_result"] = pd.DataFrame(message["analytic_result"])
                            except Exception as e:
                                print(f"DataFrame 변환 중 오류: {e}")
                    
                    set_page_state(PAGE_NAME, "messages", messages)
                else:
                    set_page_state(PAGE_NAME, "messages", [{"role": "assistant", "content": CONSTANTS["ASSISTANT_MESSAGE"]}])
                st.rerun()
        with col2:
            # 삭제 버튼 (현재 활성화된 스레드는 삭제 불가)
            if st.button("🗑️", key=f"delete_{thread['created_at']}", help="스레드 삭제"):
                if delete_thread(thread["internal_id"]):
                    st.success("스레드가 삭제되었습니다.")
                    st.rerun()
                else:
                    st.error("스레드 삭제 중 오류가 발생했습니다.")

# sidebar 문서 관리
def render_sidebar_document():
    # 문서 관리 섹션
    st.sidebar.subheader("📚 문맥 관리")
    
    # 파일 업로드
    uploaded_files = st.sidebar.file_uploader(
        "분석에 필요한 문서를 업로드하세요",
        type=['pdf', 'docx', 'pptx', 'json', 'csv', 'xlsx', 'txt'],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.sidebar.button("📥 문서 등록", use_container_width=True):
            with st.spinner("⏳ 문서를 처리하는 중..."):
                try:
                    files_text = get_text(uploaded_files, document_list_path=DOCUMENT_LIST_PATH)
                    print(f"🔢 [files_text] 문서 텍스트 추출 완료: {len(files_text)}")
                    text_chunks = get_text_chunks(files_text)
                    
                    # 기존 vectorstore 로드 또는 새로 생성
                    if os.path.exists(VECTOR_DB_ANSS_PATH):
                        print(f"🔢 [render_sidebar] 기존 vectorstore 로드")
                        vectorstore = load_vectorstore('./vectordb/analysis')
                        vectorstore.add_documents(text_chunks)
                    else:
                        print(f"🔢 [render_sidebar] 새로운 vectorstore 생성")
                        vectorstore = get_vectorstore(text_chunks)
                    
                    print(f"🔢 [render_sidebar] vectorstore 저장 경로 {VECTOR_DB_ANSS_PATH}")
                    vectorstore.save_local('./vectordb/analysis')
                    set_page_state(PAGE_NAME, "vectorstore", vectorstore)
                    
                    # 문서 목록 업데이트
                    document_list = load_document_list(document_list_path=DOCUMENT_LIST_PATH)
                    new_documents = [file.name for file in uploaded_files]
                    document_list.extend(new_documents)
                    save_document_list(document_list_path=DOCUMENT_LIST_PATH, document_list=list(set(document_list)))
                    
                    st.sidebar.success("✅ 문서 등록이 완료되었습니다!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    print(f"❌ 오류 발생: {traceback.format_exc()}")
                    return
    
    # 등록된 문서 목록
    st.sidebar.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("##### 📑 등록된 문서 목록")
    
    document_list = load_document_list(document_list_path=DOCUMENT_LIST_PATH)
    if document_list:
        for doc in document_list:
            cols = st.sidebar.columns([0.85, 0.15])
            with cols[0]:
                st.markdown(f"- {doc}")
            with cols[1]:
                if st.button("🗑️", key=f"del_{doc}", help=f"문서 삭제: {doc}"):
                    try:
                        # 문서 파일 삭제
                        doc_path = Path(f"../documents/{doc}")
                        if doc_path.exists():
                            os.remove(doc_path)
                        
                        # vectorstore 재구축
                        if rebuild_vectorstore_without_document(doc, DOCUMENT_LIST_PATH):
                            document_list.remove(doc)
                            save_document_list(document_list_path=DOCUMENT_LIST_PATH, document_list=list(set(document_list)))
                            st.toast(f"🗑️ '{doc}' 문서가 삭제되었습니다.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            print("Vectorstore 재구축 중 오류가 발생했습니다.")
                    except Exception as e:
                        print(f"문서 삭제 중 오류 발생: {e}")
    else:
        st.sidebar.info("등록된 문서가 없습니다.")


@st.fragment
def render_chat_interface():
    """채팅 인터페이스 렌더링"""
    messages = get_page_state(PAGE_NAME, "messages", [])
    
    # ✅ 채팅 메시지를 표시할 고정 컨테이너 생성
    chat_container = st.container()
    for message in messages:
        with chat_container:
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                continue

            with st.chat_message(message["role"]):

                if "error_message" in message:
                    st.error(f"⚠️ 오류 발생: {message['error_message']}")

                # ✅ 일반 텍스트 메시지 출력 (질문 및 일반 답변)
                if "content" in message and message["content"]:
                    if message["role"] == "assistant":
                        if message["content"] != "안녕하세요! AI 분석 어시스턴트입니다. 무엇이든 물어보세요!":
                            st.markdown("💬 **응답**")
                        st.markdown(message["content"])
                    else:
                        st.write(message["content"])

                # ✅ 생성된 코드 출력 (에러가 있을 때만)
                if "error_message" in message and "generated_code" in message and message["generated_code"]:
                    st.markdown("""\n##### 🔢 생성된 코드 (에러 발생)\n""")
                    code_to_display = message["generated_code"]
                    if "```python" in code_to_display:
                        code_to_display = code_to_display.split("```python")[1].split("```")[0]
                    elif "```" in code_to_display:
                        code_to_display = code_to_display.split("```")[1]
                    st.code(code_to_display, language="python")

                # ✅ 실행된 코드 출력
                if "validated_code" in message and message["validated_code"]:
                    st.markdown("""\n##### 🔢 실행된 코드\n""")  
                    st.code(message["validated_code"].split("```python")[1].split("```")[0], language="python")

                # ✅ 분석 결과 (테이블)
                if "analytic_result" in message and message["analytic_result"]:
                    st.divider()
                    st.markdown("""\n##### 📑 분석 결과\n""")                
                    if isinstance(message["analytic_result"], dict):
                        for key, value in message["analytic_result"].items():
                            st.markdown(f"#### {key}")
                            if isinstance(value, pd.DataFrame):
                                if value.shape[0] <= 10:
                                    st.dataframe(value, use_container_width=True, )
                                else:
                                    st.dataframe(value.head(50), use_container_width=True, )
                            else:
                                st.write(value)
                    else:
                        df_result = pd.DataFrame(message["analytic_result"])
                        if df_result.shape[0] <= 10:
                            st.dataframe(df_result, use_container_width=True, )
                        else:
                            st.dataframe(df_result.head(50), use_container_width=True, )

                # ✅ 차트 출력
                if "chart_filename" in message:
                    if message["chart_filename"]:
                        st.divider()
                        st.markdown("""\n##### 📑 분석 차트\n""")
                        st.image(message["chart_filename"])
                    else:
                        if "q_category" in message and message["q_category"] == "Analytics":
                            st.warning("📉 차트가 생성되지 않았습니다.")

                # ✅ 인사이트 출력
                if "insights" in message and message["insights"]:
                    st.divider()
                    st.markdown("""\n##### 📑 분석 인사이트\n""")
                    st.text_area(message["insights"])

                # ✅ 리포트 텍스트 출력
                if "report" in message and message["report"]:
                    st.divider()
                    st.markdown("""
                        ##### 📑 분석 리포트
                    """)
                    st.markdown(message["report"])

                # ✅ 7. 피드백 텍스트 출력
                if "feedback" in message and message["feedback"]:
                    st.divider()
                    st.markdown("""
                        ##### 📑 분석 피드백
                    """)
                    st.markdown(message["feedback"])

                # ✅ 8. 상세 분석 제안
                if "feedback_point" in message and message["feedback_point"]:
                    st.divider()
                    st.markdown("""
                        ##### ☞ 제안드리는 상세 분석 목록
                    """)
                    
                    # 피드백 포인트가 문자열인 경우 리스트로 변환
                    feedback_points = message["feedback_point"]
                    if isinstance(feedback_points, str):
                        # 문자열을 리스트로 변환 (쉼표, 줄바꿈 등으로 구분된 경우)
                        feedback_points = re.split(r'[,\n]+', feedback_points)
                        feedback_points = [point.strip() for point in feedback_points if point.strip()]
                    
                    # 버튼 생성을 위한 컬럼 레이아웃
                    cols = st.columns(min(2, len(feedback_points)))
                    
                    # 현재 메시지의 question_id 가져오기
                    current_question_id = message.get("question_id", "")
                    
                    # print(f"🔢 [render_chat_interface] 제안드리는 상세 분석 목록 버튼별 id: {current_question_id}")

                    for i, point in enumerate(feedback_points):
                        with cols[i % len(cols)]:
                            # 각 제안을 버튼으로 표시
                            if st.button(
                                f"♣ {point}", 
                                key=f"analysis_btn_{current_question_id}_{i}",
                                use_container_width=True,
                                type="secondary"
                            ):
                                print(f"🔢 [render_chat_interface] 제안드리는 상세 분석 목록 버튼 클릭: {current_question_id}")
                                # 현재 thread json 파일에서 parent_question_id에 해당하는 메시지 정보 가져오기
                                thread_data = load_thread(get_page_state(PAGE_NAME, "internal_id"))
                                if thread_data and "messages" in thread_data:
                                    parent_message = None
                                    for msg in thread_data["messages"]:
                                        if msg.get("question_id") == current_question_id:
                                            parent_message = msg
                                            break
                                    
                                    if parent_message:
                                        # 버튼 클릭 시 해당 분석에 대한 질의 자동 생성
                                        auto_query = f"추가 데이터 분석 요청 : {point}"
                                        
                                        # 세션 상태에 자동 생성된 질의와 부모 질문 ID 저장
                                        st.session_state["auto_generated_query"] = auto_query
                                        st.session_state["auto_feedback_point"] = point
                                        st.session_state["parent_question_id"] = current_question_id
                                        
                                        # 페이지 리로드 (자동 질의 처리를 위해)
                                        st.rerun()
                                        
        
def process_chat_input():
    """채팅 입력 처리"""
    
    # 현재 처리 중인지 확인
    is_processing = get_page_state(PAGE_NAME, "is_processing", False)
    
    # 자동 생성된 질의가 있는지 확인
    auto_query = st.session_state.get("auto_generated_query", None)
    auto_feedback_point = st.session_state.get("auto_feedback_point", None)
    parent_question_id = st.session_state.get("parent_question_id", None)
    internal_id=get_page_state(PAGE_NAME, "internal_id")
    
    # 사용자 입력 또는 자동 생성된 질의 처리
    if (query := st.chat_input(
        "질문을 입력해주세요.",
        disabled=is_processing, # 처리 중일때 입력 비활성화
        key="chat_input"
    )) or auto_query:
        print(f"🔢 [process_chat_input] 부모 질문 ID: {parent_question_id}")
        
        # ✅ JavaScript 스크롤 기능 추가 (랜덤 ID 적용)
        random_id = random.randint(1000, 9999)
        js_code = f"""
        <div id="scroll-to-me" style='height: 1px;'></div>
        <script id="{random_id}">
            var e = document.getElementById("scroll-to-me");
            if (e) {{
                e.scrollIntoView({{behavior: "smooth"}});
                e.remove();
            }}
        </script>
        """
        st.components.v1.html(js_code, height=0)
        
        parent_message = None
        # (피드백) 자동 생성된 질의가 있으면 사용하고 세션에서 제거
        if auto_query:
            query = auto_query
            
            # 부모 메시지 정보 가져오기
            if parent_question_id:
                parent_message = get_parent_message(
                    internal_id=internal_id,
                    parent_question_id=parent_question_id
                )
            
            # 세션 상태에서 임시 데이터 제거
            del st.session_state["auto_generated_query"]
            if auto_feedback_point: del st.session_state["auto_feedback_point"]
            if parent_question_id: del st.session_state["parent_question_id"]
        
        # ✅ 마트 데이터 로드
        load_mart_data()
        
        # ✅ 사용자 메시지를 먼저 출력
        with st.chat_message("user"):
            st.write(query)
            
        user_message = {"role": "user", "content": query}
        st.session_state.setdefault(f"{PAGE_NAME}_messages", []).append(user_message)

        # ✅ 어시스턴트 응답 처리 - spinner를 밖으로
        with st.chat_message("assistant"):
            with st.spinner("🔍 답변을 생성 중..."):
                # 자동 생성된 질의인 경우 Analytics부터 시작하도록 설정
                start_from_analytics = True if auto_feedback_point else False
                
                response_data = process_chat_response(
                    st.session_state[f"{PAGE_NAME}_assistant"], 
                    query,
                    internal_id=get_page_state(PAGE_NAME, "internal_id"),
                    start_from_analytics=start_from_analytics,
                    feedback_point=auto_feedback_point if auto_feedback_point else None,
                    parent_message=parent_message
                )
                
                # 부모 질문 ID가 있는 경우 응답 데이터에 추가
                if parent_question_id:
                    response_data["parent_question_id"] = parent_question_id


        # 메시지 상태 업데이트
        messages = get_page_state(PAGE_NAME, "messages", [])
        messages.append(response_data)  # 새로운 응답 메시지를 추가
        set_page_state(PAGE_NAME, "messages", messages)  # 업데이트된 메시지 상태 저장

        # 대화 스레드 저장
        save_thread(
            get_page_state(PAGE_NAME, "internal_id"),  # 내부 ID를 가져와서
            get_page_state(PAGE_NAME, "messages")  # 현재 메시지 상태를 저장
        )

        # 페이지를 다시 로드하여 UI를 업데이트
        st.rerun()

def initialize_vectorstore():
    """벡터스토어 초기화 및 로딩을 처리하는 함수"""
    if not get_page_state(PAGE_NAME, "vectorstore"):
        with st.spinner("🔄 문맥을 불러오는 중..."):
            if not (vectorstore := load_vectorstore('./vectordb/analysis')):
                st.warning("⚠️ 문맥이 등록되지 않았습니다. 먼저 문서를 등록해주세요.")
                return None
            set_page_state(PAGE_NAME, "vectorstore", vectorstore)
    return get_page_state(PAGE_NAME, "vectorstore")

def render_right_sidebar():
    """오른쪽 사이드바 렌더링"""
    
    # 오른쪽 사이드바 내용
    st.markdown("### 🔍 분석 도구")
    st.markdown("---")
    
    # 여기에 오른쪽 사이드바의 내용을 추가할 수 있습니다
    with st.expander("📊 데이터 요약", expanded=True):
        st.markdown("데이터 요약 정보를 표시할 수 있습니다.")
    
    with st.expander("📈 시각화 옵션", expanded=False):
        st.markdown("차트 옵션을 설정할 수 있습니다.")
        
    with st.expander("⚙️ 분석 설정", expanded=False):
        st.markdown("분석 관련 설정을 할 수 있습니다.")

def main():
    """메인 함수"""
    st.set_page_config(
        page_title=CONSTANTS["PAGE_TITLE"],
        page_icon=CONSTANTS["PAGE_ICON"],
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    initialize_session_state()  # 세션 상태 초기화
    apply_custom_styles()  # 커스텀 스타일 적용
    
    # 왼쪽 사이드바 렌더링 (기존 Streamlit 사이드바 사용)
    render_sidebar_chat()  # 채팅 관리
    render_sidebar_document()  # 문서 관리
    
    # 메인 컨텐츠와 오른쪽 사이드바를 2단 컬럼으로 구성
    main_content, right_area = st.columns([4, 1])
    
    with main_content:
        render_mart_selector()  # 마트 선택 렌더링
        vectorstore = initialize_vectorstore()  # 벡터스토어 초기화
        if not vectorstore:
            return
        render_chat_interface()  # 채팅 기록 인터페이스 렌더링
        st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    with right_area:
        render_right_sidebar()  # 오른쪽 사이드바
    
    process_chat_input()  # 채팅 입력 처리

if __name__ == '__main__':
    main()
