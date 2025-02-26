# ✅ 기본 라이브러리
from datetime import datetime
import os
import pandas as pd
import time
from glob import glob
from pathlib import Path

# ✅ 내부 유틸리티 모듈
from utils.vector_handler import get_text, get_text_chunks, load_vectorstore, load_document_list, save_document_list, get_vectorstore, rebuild_vectorstore_without_document    
from utils.mart_handler import get_available_marts, load_selected_mart
from utils.chat_handler import handle_chat_response
from utils.thread_handler import load_threads_list, create_new_thread, save_thread, load_thread, rename_thread

from ai_agent_v2 import DataAnayticsAssistant

# ✅ 3자 패키지
import streamlit as st
import pyautogui

PROCESSED_DATA_PATH = "../output/stage1/processed_data_info.xlsx"
DOCUMENT_LIST_PATH = "../../documents/analysis"
VECTOR_DB_ANSS_PATH = "../../vectordb/analysis"

# 상수 정의
CONSTANTS = {
    "PAGE_TITLE": "분석 어시스턴트",
    "PAGE_ICON": "🔍",
}

# ✅ 세션 상태 초기화 함수 수정
def initialize_session_state():
    """세션 상태 초기화"""
    if "selected_data_marts" not in st.session_state:
        st.session_state.selected_data_marts = []
    if "loaded_mart_data" not in st.session_state:
        st.session_state.loaded_mart_data = {}

        
    # OpenAI API Key 검증
    if not (openai_api_key := os.getenv('OPENAI_API_KEY')):
        st.warning("⚠️ OpenAI API Key가 설정되지 않았습니다. 환경 변수를 확인하세요.")
        return
    
    # AI Assistant 초기화
    if "assistant" not in st.session_state:
        with st.spinner("🤖 AI Agent를 로드하는 중..."):
            st.session_state['assistant'] = DataAnayticsAssistant(openai_api_key)
    
    initial_states = {
        "show_popover": True,
        "messages": [{"role": "assistant", "content": "안녕하세요! AI 분석 어시스턴트입니다. 무엇이든 물어보세요!"}]
    }
    
    for key, value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_custom_styles():
    """UI 스타일 적용"""
    st.markdown(
        """
        <style>
            /* 사이드바 기본 너비 설정 */
            [data-testid="stSidebar"] {
                min-width: 330px !important;
                max-width: 800px !important;
            }
            
            /* 사이드바 리사이즈 핸들 스타일 */
            [data-testid="stSidebar"] > div:first-child {
                width: auto !important;
                resize: horizontal !important;
                overflow-x: auto !important;
            }
            
            /* 네비게이션 컨테이너 스타일 수정 */
            div[data-testid="stSidebarNav"] {
                height: auto !important;
                min-height: 300px !important;  /* 네비게이션 영역 최소 높이 */
                
            }
            
            /* 메뉴 영역 스타일 수정 */
            section[data-testid="stSidebarNav"] {
                top: 0 !important;
                padding-left: 1.5rem !important;
                height: auto !important;
                min-height: 300px !important;
            }
            
            /* 메뉴 아이템 컨테이너 */
            section[data-testid="stSidebarNav"] > div {
                height: auto !important;
                padding: 1rem 0 !important;
            }
            
            /* 스크롤바 숨기기 */
            section[data-testid="stSidebarNav"]::-webkit-scrollbar {
                display: none !important;
            }
            
            
            .stChatMessage { max-width: 90% !important; }
            .stMarkdown { font-size: 16px; }
            .reference-doc { font-size: 12px !important; }
            table { font-size: 12px !important; }
            
            /* 데이터프레임 스타일 수정 */
            .dataframe {
                font-size: 12px !important;
                white-space: nowrap !important;  /* 텍스트 줄바꿈 방지 */
                text-align: left !important;
            }
            
            /* 데이터프레임 셀 스타일 */
            .dataframe td, .dataframe th {
                min-width: 100px !important;  /* 최소 너비 설정 */
                max-width: 200px !important;  /* 최대 너비 설정 */
                padding: 8px !important;
                text-overflow: ellipsis !important;
            }
            
            /* 데이터프레임 헤더 스타일 */
            .dataframe thead th {
                text-align: left !important;
                font-weight: bold !important;
                background-color: #f0f2f6 !important;
            }
            
            /* 채팅 입력란 하단 고정 스타일 */
            section[data-testid="stChatInput"] {
                position: fixed !important;
                bottom: 0 !important;
                background: white !important;
                padding: 1rem !important;
                z-index: 999 !important;
                width: calc(100% - 350px) !important; /* 사이드바 너비 고려 */
                left: 350px !important; /* 사이드바 너비에 맞춤 */
                border-top: 1px solid #ddd !important;
            }
            
            /* 채팅 컨테이너에 하단 여백 추가 (입력란이 메시지를 가리지 않도록) */
            [data-testid="stChatMessageContainer"] {
                padding-bottom: 30px !important;
            }
            
            /* 반응형 조정: 사이드바가 접혀있을 때 */
            @media (max-width: 992px) {
                section[data-testid="stChatInput"] {
                    width: 100% !important;
                    left: 0 !important;
                }
            }
        </style>
        <script>
        // 스크롤을 자동으로 아래로 이동
        function scrollToBottom() {
            var chatContainer = document.querySelector('[data-testid="stChatMessageContainer"]');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        
        // Streamlit이 로드될 때 스크롤을 맨 아래로 이동
        setTimeout(scrollToBottom, 500);
        </script>
        """,
        unsafe_allow_html=True
    )

def close_popover():
    """Popover 닫기"""
    st.session_state["show_popover"] = False
    pyautogui.hotkey("esc")

# ✅ 마트 선택/해제 처리
def handle_mart_selection(mart):
    """마트 선택/해제 처리"""
    # 현재 선택된 마트 목록
    current_marts = set(st.session_state.get("selected_data_marts", []))
    
    if mart in current_marts:
        ##############################################################
        # 마트 선택 해제
        ##############################################################
        current_marts.remove(mart)
        if mart in st.session_state.loaded_mart_data:
            del st.session_state.loaded_mart_data[mart]
            print(f"🗑️ 마트 비활성화: {mart}")
            
            # Assistant 마트 상태 업데이트
            if hasattr(st.session_state, 'assistant'):
                if st.session_state.loaded_mart_data:
                    # 남은 마트들로 업데이트
                    st.session_state.assistant.set_active_mart(
                        data_mart=st.session_state.loaded_mart_data,
                        mart_name=list(st.session_state.loaded_mart_data.keys())
                    )
                else:
                    # 모든 마트가 비활성화된 경우
                    print("🗑️ 모든 마트가 비활성화되었습니다.")
                    st.session_state.assistant.active_marts = None
                    st.session_state.assistant.mart_info = None
    else:
        ##############################################################
        # 마트 선택 및 활성화
        ##############################################################
        current_marts.add(mart)
        data = load_selected_mart(mart)
        if data is not None:
            st.session_state.loaded_mart_data[mart] = data
            print(f"✅ 마트 활성화 클릭: {mart} (shape: {data.shape})")
            
            # AI Assistant에 활성화된 마트 설정
            if hasattr(st.session_state, 'assistant'):
                st.session_state.assistant.set_active_mart(
                    data_mart=st.session_state.loaded_mart_data,
                    mart_name=list(st.session_state.loaded_mart_data.keys())
                )
        else:
            print(f"❌ 마트 로드 실패: {mart}")
    
    ##############################################################
    # 현재 활성화된 마트 상태 출력
    ##############################################################
    if st.session_state.loaded_mart_data:
        print("\n📊 현재 활성화된 마트 상태:")
        for mart_name, data in st.session_state.loaded_mart_data.items():
            print(f"- {mart_name}: {data.shape} rows x {data.shape[1]} columns")
        print(f"총 활성화된 마트 수: {len(st.session_state.loaded_mart_data)}\n")
    
    st.session_state.selected_data_marts = list(current_marts)
    st.rerun()
    
@st.fragment
def render_mart_selector():
    """마트 선택 UI 렌더링"""
    # 전체 컨테이너를 사용하여 너비 확보
    with st.container():
        # 왼쪽 컨텐츠와 API Key 상태를 위한 컬럼 분할
        left_content, middle_content, right_content = st.columns([0.3, 0.3, 0.4])
        
        # 마트 활성화 버튼
        with left_content:
            if st.button(
                "📊 마트 ↓" if not st.session_state.get("show_mart_manager", False) else "📊 마트 ↑",
                use_container_width=True
            ):
                # 토글 동작 구현
                st.session_state.show_mart_manager = not st.session_state.get("show_mart_manager", False)
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

    # 마트 관리자 UI
    if "show_mart_manager" not in st.session_state:
        st.session_state.show_mart_manager = False

    if st.session_state.show_mart_manager:
        with st.container():
            st.markdown(
                """
                <style>
                [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    background-color: white;
                    margin: 10px 0;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # 탭 인터페이스
            tab1, tab2 = st.tabs(["📊 마트 선택", "📈 활성화된 마트"])
            
            with tab1:
                available_marts = get_available_marts()
                if not available_marts:
                    st.warning("사용 가능한 데이터 마트가 없습니다.")
                else:
                    # 마트 목록을 그리드로 표시
                    mart_data = []
                    # for mart in available_marts:
                    #     is_selected = mart in st.session_state.get("selected_data_marts", [])
                    #     status = "✅ 활성" if is_selected else "⬜ 비활성"
                        
                    #     if mart in st.session_state.loaded_mart_data:
                    #         data = st.session_state.loaded_mart_data[mart]
                    #         rows, cols = data.shape
                    #     else:
                    #         rows, cols = "-", "-"
                        
                        # mart_data.append({
                        #     "마트명": mart,
                        #     "상태": status,
                        #     "행 수": f"{rows:,}" if isinstance(rows, int) else rows,
                        #     "열 수": cols
                        # })
                    
                    # df = pd.DataFrame(mart_data)
                    # st.dataframe(
                    #     df,
                    #     hide_index=True,
                    #     use_container_width=True,
                    #     column_config={
                    #         "마트명": st.column_config.Column(width="large"),
                    #         "상태": st.column_config.Column(width="small"),
                    #         "행 수": st.column_config.Column(width="medium"),
                    #         "열 수": st.column_config.Column(width="small")
                    #     }
                    # )
                    
                    # 마트 선택 인터페이스
                    st.markdown("### 마트 선택")
                    cols = st.columns(3)
                    for i, mart in enumerate(available_marts):
                        with cols[i % 3]:
                            is_selected = mart in st.session_state.get("selected_data_marts", [])
                            if st.button(
                                f"{'✅' if is_selected else '⬜'} {mart}",
                                key=f"mart_{mart}",
                                use_container_width=True,
                                type="primary" if is_selected else "secondary"
                            ):
                                handle_mart_selection(mart)
                                st.rerun()
            
            with tab2:
                if not st.session_state.get("selected_data_marts"):
                    st.info("활성화된 마트가 없습니다.")
                else:
                    for mart in st.session_state["selected_data_marts"]:
                        with st.expander(f"📊 {mart}", expanded=True):
                            if mart in st.session_state.loaded_mart_data:
                                data = st.session_state.loaded_mart_data[mart]
                                st.markdown(f"""
                                    #### 마트 정보
                                    - **행 수:** {data.shape[0]:,}
                                    - **열 수:** {data.shape[1]}
                                    
                                    #### 미리보기
                                """)
                                st.dataframe(data.head(5), use_container_width=True)

def render_sidebar():
    """사이드바 렌더링"""
    st.sidebar.subheader("📚 대화 관리")

    # ✅ 새로운 쓰레드 생성 버튼
    if st.sidebar.button("🆕 새 대화 시작", use_container_width=True):
        new_thread_id = create_new_thread()
        st.session_state["internal_id"] = new_thread_id  # 세션 상태 업데이트
        st.session_state["messages"] = []  # 새로운 쓰레드이므로 대화 초기화
        st.rerun()

    # ✅ 저장된 쓰레드 목록 표시
    st.sidebar.markdown("### 📝 기존 대화 목록")
    threads = load_threads_list()
    
    for idx, thread in enumerate(threads):
        # ✅ 버튼 key를 internal_id나 created_at으로 구분
        button_key = f"thread_{thread['created_at']}"
        if st.sidebar.button(
            f"💬 {thread['thread_id']}", 
            key=button_key,
            help=f"ID: {thread.get('internal_id', '없음')}"  # 툴팁으로 internal_id 표시
        ):
            # print(f"""🔢 [render_sidebar] 쓰레드 목록 표시 시작 (internal_id: {thread["internal_id"]})""")
            st.session_state["internal_id"] = thread["internal_id"]
            loaded_thread = load_thread(thread["internal_id"])
           
            # 메시지를 세션 상태에 로드
            if loaded_thread and "messages" in loaded_thread:
                # print(f"""🔢 [render_sidebar] loaded_thread["messages"] {loaded_thread["messages"]})""")
                st.session_state["messages"] = loaded_thread["messages"]
            else:
                # messages가 없거나 비어있는 경우 기본 메시지 설정
                st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요! AI 분석 어시스턴트입니다. 무엇이든 물어보세요!"}]
            st.rerun()

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
                    text_chunks = get_text_chunks(files_text)
                    
                    # 기존 vectorstore 로드 또는 새로 생성
                    if os.path.exists(VECTOR_DB_ANSS_PATH):
                        vectorstore = load_vectorstore(db_path = VECTOR_DB_ANSS_PATH)
                        vectorstore.add_documents(text_chunks)
                    else:
                        vectorstore = get_vectorstore(text_chunks)
                    
                    vectorstore.save_local(VECTOR_DB_ANSS_PATH)
                    
                    # 문서 목록 업데이트
                    document_list = load_document_list(document_list_path=DOCUMENT_LIST_PATH)
                    new_documents = [file.name for file in uploaded_files]
                    document_list.extend(new_documents)
                    save_document_list(document_list_path=DOCUMENT_LIST_PATH, document_list=list(set(document_list)))
                    
                    st.sidebar.success("✅ 문서 등록이 완료되었습니다!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    print(f"❌ 오류 발생: {e}")
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
                        if rebuild_vectorstore_without_document(doc):
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
    """채팅 인터페이스 렌더링 (각 출력 영역별 타이틀 추가)"""
    for message in st.session_state["messages"]:
        print(f"🔢 [render_chat_interface] message: {message}") 
        with st.chat_message(message["role"]):
            
            # ✅ 1. 일반 텍스트 메시지 출력 (질문 및 일반 답변)
            if "content" in message and message["content"]:
                if message["role"] == "assistant":
                    if message["content"] != "안녕하세요! AI 분석 어시스턴트입니다. 무엇이든 물어보세요!":
                        st.markdown("💬 **응답**")
                    st.markdown(message["content"])
                else:
                    # st.markdown("❓ **질문**")
                    st.write(message["content"])

            # ✅ 2. 실행된 코드 출력
            if "validated_code" in message and message["validated_code"]:
                st.markdown("""
                    ##### 🔢 실행된 코드
                """)  
                st.code(message["validated_code"].split("```python")[1].split("```")[0], language="python")

            # ✅ 3. 분석 결과 (테이블)
            if "analytic_result" in message and message["analytic_result"]:
                st.divider()
                st.markdown("""
                    ### 📑 분석 결과
                """, )                
                if isinstance(message["analytic_result"], dict):
                    for key, value in message["analytic_result"].items():
                        st.markdown(f"#### {key}")
                        if isinstance(value, pd.DataFrame):
                            if value.shape[0] <= 10:
                                st.table(value)
                            else:
                                st.dataframe(value.head(50))
                        else:
                            st.write(value)
                else:
                    df_result = pd.DataFrame(message["analytic_result"])
                    if df_result.shape[0] <= 10:
                        st.table(df_result)
                    else:
                        st.dataframe(df_result.head(50))

            # ✅ 4. 차트 출력
            if "chart_filename" in message:
                if message["chart_filename"]:
                    st.divider()
                    st.markdown("""
                        ### 📑 분석 차트
                    """)
                    st.image(message["chart_filename"])
                else:
                    if "q_category" in message and message["q_category"] == "Analytics":
                        st.warning("📉 차트가 생성되지 않았습니다.")

            # ✅ 5. 인사이트 출력
            if "insights" in message and message["insights"]:
                st.divider()
                st.markdown("""
                    ### 📑 분석 인사이트
                """)
                st.markdown(message["insights"])

            # # ✅ 6. 리포트 다운로드 버튼 개선 (파일명 동적 설정)
            # if "report_filename" in message and message["report_filename"]:
            #     report_file = message["report_filename"]
            #     with open(report_file, "rb") as f:
            #         report_bytes = f.read()
                
            #     st.markdown("📑 **리포트 다운로드**")
            #     st.download_button(
            #         label="📥 분석 리포트 다운로드 (Excel)",
            #         data=report_bytes,
            #         file_name=Path(report_file).name,  # 🔹 파일명을 동적으로 설정
            #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            #     )
            # ✅ 6. 리포트 텍스트 출력
            if "report" in message and message["report"]:
                st.divider()
                st.markdown("""
                    ### 📑 분석 리포트
                """)
                st.markdown(message["report"])


    # ✅ 사용자 입력 필드 추가
    if query := st.chat_input("질문을 입력해주세요."):
        
        # 사용자가 처음 질문을 할 때 쓰레드 생성
        if "internal_id" not in st.session_state or st.session_state["internal_id"] == "새 대화":
            st.session_state["internal_id"] = create_new_thread()
            st.session_state["messages"] = []  # 새로운 쓰레드이므로 대화 초기화

        user_message = {"role": "user", "content": query}
        st.session_state.setdefault("messages", []).append(user_message)

        with st.chat_message("user"):
            st.write(query)
            
        with st.spinner("🔍 답변을 생성 중..."):
            # ✅ 채팅 응답 처리
            response_data = handle_chat_response(
                st.session_state['assistant'], 
                query,
                internal_id=st.session_state["internal_id"]
            )

        st.session_state["messages"].append(response_data)

        # ✅ 대화 이력을 쓰레드에 저장
        # print(f"""🔢 대화 이력을 쓰레드에 저장 대화 이력을 쓰레드에 저장 대화 이력을 쓰레드에 저장\n {st.session_state["messages"]}""")
        save_thread(st.session_state["internal_id"], st.session_state["messages"])

        st.rerun()  # UI 새로고침



def main():
    """메인 함수"""
    st.set_page_config(
        page_title=CONSTANTS["PAGE_TITLE"],
        page_icon=CONSTANTS["PAGE_ICON"],
        layout='wide'
    )
    
    # ✅ 세션 상태 초기화
    initialize_session_state()

    # ✅ 커스텀 스타일 적용
    apply_custom_styles()

    # ✅ 마트 선택 UI 렌더링
    render_mart_selector()
    
    # ✅ 사이드바 렌더링 (문서 관리 포함)
    render_sidebar()


    # 벡터스토어 초기화
    if "vectorstore" not in st.session_state:
        with st.spinner("🔄 문맥을 불러오는 중..."):
            if not (vectorstore := load_vectorstore(db_path = VECTOR_DB_ANSS_PATH)):
                st.warning("⚠️ 문맥이 등록되지 않았습니다. 먼저 문서를 등록해주세요.")
                return
            st.session_state["vectorstore"] = vectorstore

    render_chat_interface()

if __name__ == '__main__':
    main()
