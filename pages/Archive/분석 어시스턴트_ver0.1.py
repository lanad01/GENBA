# ✅ 기본 라이브러리
import os
import math
import pandas as pd
import time
import json
import traceback
from glob import glob
from pathlib import Path

# ✅ LangChain 관련 모듈
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings

# ✅ 내부 유틸리티 모듈
from utils.mart_agent import AIAnalysisAssistant
from utils.vector_handler import get_text, get_text_chunks, load_vectorstore, load_document_list, save_document_list, get_vectorstore, rebuild_vectorstore_without_document    
from genba.src.utils.pages_handler import get_available_marts, load_selected_mart

# ✅ 3자 패키지
from loguru import logger
import streamlit as st
import pyautogui

PROCESSED_DATA_PATH = "../output/stage1/processed_data_info.xlsx"

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

    # ✅ ConversationBufferMemory를 사용하여 이전 대화 기록 저장
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)

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
                max-width: 500px !important;
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
        </style>
        """,
        unsafe_allow_html=True
    )

def close_popover():
    """Popover 닫기"""
    st.session_state["show_popover"] = False
    pyautogui.hotkey("esc")


def handle_mart_selection(mart):
    """마트 선택/해제 처리"""
    current_marts = set(st.session_state.get("selected_data_marts", []))
    
    if mart in current_marts:
        # 마트 선택 해제
        current_marts.remove(mart)
        # 메모리에서 데이터 제거
        if mart in st.session_state.loaded_mart_data:
            del st.session_state.loaded_mart_data[mart]
            print(f"🗑️ 마트 제거됨: {mart}")
    else:
        # 마트 선택
        current_marts.add(mart)
        # 데이터 로드
        data = load_selected_mart(mart)
        if data is not None:
            st.session_state.loaded_mart_data[mart] = data
            print(f"✅ 마트 로드됨: {mart} (shape: {data.shape})")
        else:
            print(f"❌ 마트 로드 실패: {mart}")
    
    # 현재 메모리에 로드된 전체 마트 상태 출력
    print("\n📊 현재 메모리 상태:")
    for mart_name, data in st.session_state.loaded_mart_data.items():
        print(f"- {mart_name}: {data.shape} rows x {data.shape[1]} columns")
    print(f"총 로드된 마트 수: {len(st.session_state.loaded_mart_data)}\n")
    
    st.session_state.selected_data_marts = list(current_marts)
    st.rerun()

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
                    for mart in available_marts:
                        is_selected = mart in st.session_state.get("selected_data_marts", [])
                        status = "✅ 활성" if is_selected else "⬜ 비활성"
                        
                        if mart in st.session_state.loaded_mart_data:
                            data = st.session_state.loaded_mart_data[mart]
                            rows, cols = data.shape
                        else:
                            rows, cols = "-", "-"
                        
                        mart_data.append({
                            "마트명": mart,
                            "상태": status,
                            "행 수": f"{rows:,}" if isinstance(rows, int) else rows,
                            "열 수": cols
                        })
                    
                    df = pd.DataFrame(mart_data)
                    st.dataframe(
                        df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "마트명": st.column_config.Column(width="large"),
                            "상태": st.column_config.Column(width="small"),
                            "행 수": st.column_config.Column(width="medium"),
                            "열 수": st.column_config.Column(width="small")
                        }
                    )
                    
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
                    files_text = get_text(uploaded_files)
                    text_chunks = get_text_chunks(files_text)
                    
                    # 기존 vectorstore 로드 또는 새로 생성
                    if os.path.exists("./vectordb"):
                        vectorstore = load_vectorstore()
                        vectorstore.add_documents(text_chunks)
                    else:
                        vectorstore = get_vectorstore(text_chunks)
                    
                    vectorstore.save_local("./vectordb")
                    
                    # 문서 목록 업데이트
                    document_list = load_document_list()
                    new_documents = [file.name for file in uploaded_files]
                    document_list.extend(new_documents)
                    save_document_list(list(set(document_list)))
                    
                    st.sidebar.success("✅ 문서 등록이 완료되었습니다!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    print(f"❌ 오류 발생: {e}")
                    return
    
    # 등록된 문서 목록
    st.sidebar.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("##### 📑 등록된 문서 목록")
    
    document_list = load_document_list()
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
                            save_document_list(document_list)
                            st.toast(f"🗑️ '{doc}' 문서가 삭제되었습니다.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            print("Vectorstore 재구축 중 오류가 발생했습니다.")
                    except Exception as e:
                        print(f"문서 삭제 중 오류 발생: {e}")
    else:
        st.sidebar.info("등록된 문서가 없습니다.")

def handle_chat_response(assistant, query):
    """채팅 응답 처리 (기존 질문과 AI 응답을 기억)"""
    try:
        with st.spinner("🔍 답변을 생성 중..."):
            # 데이터프레임 선택 여부 확인
            selected_marts = st.session_state.get("selected_data_marts", [])
            if not selected_marts:
                st.warning("⚠️ 먼저 데이터프레임을 선택해주세요.")
                return None

            # 메모리 초기화 및 가져오기
            memory = st.session_state.setdefault("memory", ConversationBufferMemory(return_messages=True))
            processed_data_info = st.session_state.get("processed_data_info", {})

            # 이전 대화 기록 가져오기
            previous_context = memory.load_memory_variables({}).get("history", "이전 대화 기록이 없습니다.")

            # 데이터 마트 정보 캐싱 (최초 1회)
            if "llm_context_cached" not in st.session_state:
                mart_context = []
                for df_name in selected_marts:
                    if df_name in processed_data_info:
                        df_info = processed_data_info[df_name]
                        mart_info = f"📊 데이터프레임: {df_name}\n"
                        mart_info += df_info[["컬럼명", "데이터 타입", "인스턴스(예제)", "컬럼설명"]].to_string(index=False)
                        mart_context.append(mart_info)

                cache_prompt = f"""당신은 데이터 분석 전문가입니다. 앞으로 사용자가 데이터 마트 관련 요청을 하면, 이 정보를 바탕으로 답변해주세요.\n**현재 활성화된 데이터 마트 정보:**\n{'\n\n'.join(mart_context)}"""
                assistant.ask(cache_prompt)
                st.session_state["llm_context_cached"] = True

            # 이전 대화 기록을 자연어로 변환
            chat_history = ""
            if previous_context != "이전 대화 기록이 없습니다.":
                for msg in previous_context:
                    if msg.type == 'human':
                        chat_history += f"사용자: {msg.content}\n"
                    elif msg.type == 'ai':
                        chat_history += f"어시스턴트: {msg.content}\n"

            # 최종 프롬프트 생성
            if not chat_history:
                full_query = f"{query}"
            else :
                full_query = f"""사용자의 이전 대화 기록을 바탕으로 문맥을 유지하며 답변해주세요.\n**이전 대화 기록:**\n{chat_history}\n**사용자 질문:** {query}"""

            # LLM 호출 및 응답 처리
            result = assistant.ask(full_query)
            response = result["messages"][-1].content

            # 대화 기록 업데이트
            memory.save_context({"input": query}, {"output": response})
            st.session_state.memory = memory

            # UI 업데이트
            st.session_state["messages"].extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ])

            st.markdown(response)
            return response

    except Exception as e:
        st.error(f"❌ 답변 생성 중 오류 발생: {traceback.format_exc()}")
        return None


def render_chat_interface():
    """채팅 인터페이스 렌더링"""
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            if response := handle_chat_response(st.session_state.assistant, query):
                st.session_state["messages"].append({"role": "assistant", "content": response})


def load_processed_data_info():
    """사전에 분석된 데이터 정보 로드"""
    if "processed_data_info" not in st.session_state:
        if not os.path.exists(PROCESSED_DATA_PATH):
            st.error("⚠️ 데이터 분석 결과 파일이 존재하지 않습니다. 먼저 마트를 지정해주세요.")
            return None
        else:
            # 모든 시트 로드
            st.session_state["processed_data_info"] = pd.read_excel(PROCESSED_DATA_PATH, sheet_name=None)
            print(f"✅ 사전 분석된 데이터 로드 완료: {list(st.session_state['processed_data_info'].keys())}")

    return st.session_state["processed_data_info"]

# ✅ Streamlit 실행 시 데이터 로드
processed_data_info = load_processed_data_info()



def main():
    """메인 함수"""
    st.set_page_config(
        page_title=CONSTANTS["PAGE_TITLE"],
        page_icon=CONSTANTS["PAGE_ICON"],
        layout='wide'
    )
    
    initialize_session_state()
    apply_custom_styles()
    
    # OpenAI API Key 검증
    if not (openai_api_key := os.getenv('OPENAI_API_KEY')):
        st.warning("⚠️ OpenAI API Key가 설정되지 않았습니다. 환경 변수를 확인하세요.")
        return

    # 마트 선택 UI 렌더링
    render_mart_selector()
    
    # 사이드바 렌더링 (문서 관리 포함)
    render_sidebar()
    
    # 벡터스토어 초기화
    if "vectorstore" not in st.session_state:
        with st.spinner("🔄 문맥을 불러오는 중..."):
            if not (vectorstore := load_vectorstore()):
                st.warning("⚠️ 문맥이 등록되지 않았습니다. 먼저 문서를 등록해주세요.")
                return
            st.session_state["vectorstore"] = vectorstore

    # AI Assistant 초기화
    if "assistant" not in st.session_state:
        with st.spinner("🤖 AI Agent를 로드하는 중..."):
            st.session_state.assistant = AIAnalysisAssistant(st.session_state["vectorstore"], openai_api_key)
    
    render_chat_interface()

if __name__ == '__main__':
    main()
