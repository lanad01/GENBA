import os
from glob import glob
import time
import streamlit as st
from dotenv import load_dotenv
import warnings
import traceback  # 추가된 부분
from pathlib import Path

# ✅ LangGraph 및 LangChain 관련 모듈
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# ✅ 데이터마트 생성 어시스턴트
from utils.mart_agent import MartAssistant
from utils.vector_handler import *

warnings.filterwarnings('ignore')
st.set_page_config(page_title="분석 어시스턴트", page_icon="🔍", layout='wide')

VECTOR_DB_MART_PATH = "./vectordb/mart"

# 프로젝트 루트 디렉토리 경로 설정
ROOT_DIR = Path(__file__).parent.parent.parent
DOCUMENT_LIST_PATH = str(ROOT_DIR / "documents" / "mart")

# ✅ 스타일 최적화
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 350px;
            max-width: 250px;
        }
        .stChatMessage { max-width: 90% !important; }
        .stMarkdown { font-size: 16px; }
        .reference-doc {
            font-size: 12px !important;
        }
        /* 테이블 폰트 크기 조정 */
        table {
            font-size: 12px !important;  /* 테이블 폰트 크기 */
        }
    </style>
    """,
    unsafe_allow_html=True
)

def initialize_session_state():
    """세션 상태 초기화"""
    # ✅ OpenAI API Key 확인
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        st.warning("⚠️ OpenAI API Key가 설정되지 않았습니다. 환경 변수를 확인하세요.")
        return
    
    print(f"🔢 [initialize_session_state] st.session_state: {st.session_state}")
    
    # 벡터스토어 초기화
    if "vectorstore" not in st.session_state:
        with st.spinner("🔄 문맥을 불러오는 중..."):
            if not (vectorstore := load_vectorstore(db_path=VECTOR_DB_MART_PATH)):
                st.warning("⚠️ 문맥이 등록되지 않았습니다. 먼저 문서를 등록해주세요.")
                return
            st.session_state["vectorstore"] = vectorstore

    # ✅ 데이터마트 생성 어시스턴트 초기화
    if "mart_assistant" not in st.session_state:
        with st.spinner("🤖 AI Agent를 로드하는 중..."):
            st.session_state.mart_assistant = MartAssistant(vectorstore, openai_api_key)

    # ✅ 문맥과 API Key가 정상적으로 등록된 경우 채팅 활성화
    st.success("✅ 문맥이 정상적으로 로드되었습니다!")
    st.success("✅ OpenAI API Key가 정상적으로 등록되었습니다!")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "안녕하세요! 마트 생성 어시스턴트입니다. 무엇이든 물어보세요!"}
        ]
        
    st.session_state["login_id"] = "admin"


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
                    files_text = get_text(uploaded_files, document_list_path=DOCUMENT_LIST_PATH)
                    print(f"🔢 [files_text] 문서 텍스트 추출 완료: {len(files_text)}")
                    text_chunks = get_text_chunks(files_text)
                    
                    # 기존 vectorstore 로드 또는 새로 생성
                    if os.path.exists(VECTOR_DB_MART_PATH):
                        vectorstore = st.session_state["vectorstore"]
                        vectorstore.add_documents(text_chunks)
                    else:
                        vectorstore = get_vectorstore(text_chunks)
                    
                    vectorstore.save_local(VECTOR_DB_MART_PATH)
                    
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
                        doc_path = Path(DOCUMENT_LIST_PATH) / doc  # 실제 문서 파일 경로
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
                            st.error("Vectorstore 재구축 중 오류가 발생했습니다.")
                    except Exception as e:
                        st.error(f"문서 삭제 중 오류 발생: {str(e)}")
                        print(f"문서 삭제 중 오류 발생: {traceback.format_exc()}")
    else:
        st.sidebar.info("등록된 문서가 없습니다.")

@st.fragment
def render_chat_interface(result):
    """채팅 인터페이스 렌더링"""
    response = result["messages"][-1].content
    print(f" 결과 :\n {result}")
                        
    if "documents" in result:
        source_documents = result['documents']
        # print(f"🔍 source_documents: {response}")

    st.markdown(response)

    # ✅ 쿼리 
    if "dataframe" in result:
        st.markdown("**실행 쿼리 :**")
        st.code(result["query"])

    # ✅ 쿼리 실행 결과 출력
    if "dataframe" in result:
        st.markdown("**실행 쿼리 결과(최대 20행만 출력):**")
        st.dataframe(result["dataframe"])
        
        # 데이터마트 저장 버튼 추가
        cols = st.columns([0.8, 0.2])
        with cols[1]:
            if st.button("💾 데이터마트 저장", use_container_width=True):
                try:
                    # 저장할 디렉토리 생성
                    save_dir = ROOT_DIR / "data"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 현재 시간을 파일명에 포함
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    file_name = f"datamart_{timestamp}.pkl"
                    save_path = save_dir / file_name
                    
                    # DataFrame을 pkl 형식으로 저장
                    result["dataframe"].to_pickle(str(save_path))
                    
                    st.success(f"✅ 데이터마트가 저장되었습니다: {file_name}")
                except Exception as e:
                    st.error(f"❌ 데이터마트 저장 중 오류가 발생했습니다: {str(e)}")
                    print(f"데이터마트 저장 중 오류 발생: {traceback.format_exc()}")

    # ✅ 차트 이미지 출력
    if "chart_filename" in result and result["chart_filename"]:
        st.markdown("**차트 결과:**")
        try:
            st.image(f"../img/{result['chart_filename']}", caption="차트 결과")
        except Exception as e:
            st.error(f"❌ 차트 이미지 출력 중 오류가 발생했습니다: {str(e)}")
            print(f"차트 이미지 출력 중 오류 발생: {traceback.format_exc()}")

    # ✅ 인사이트 출력
    if "insights" in result:
        st.markdown("**인사이트:**")
        st.markdown(result["insights"])

    # ✅ 엑셀 다운로드 버튼 추가
    try:
        excel_file_path = max(glob(os.path.join("../output", '*.xlsx')), key=os.path.getctime)  # 생성된 엑셀 파일 경로
        if os.path.exists(excel_file_path):
            with open(excel_file_path, 'rb') as file:
                st.download_button(
                    label="📥 엑셀 보고서 다운로드",
                    data=file,
                    file_name='final_report.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )
    except Exception as e:
        st.error(f"❌ 엑셀 다운로드 중 오류가 발생했습니다: {str(e)}")
        print(f"엑셀 다운로드 중 오류 발생: {traceback.format_exc()}")

    st.session_state["messages"].append({"role": "assistant", "content": response})

    if "documents" in result:
        try:
            with st.expander("📂 참고 문서 확인"):
                st.markdown('<div class="reference-doc">', unsafe_allow_html=True)
                for i, doc in enumerate(source_documents[:3]):
                    st.markdown(f"📄 **출처 {i+1}:** {doc.metadata['source']}")
                    st.markdown(f"> {doc.page_content[:200]} ...")
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ 참고 문서 확인 중 오류가 발생했습니다: {str(e)}")
            print(f"참고 문서 확인 중 오류 발생: {traceback.format_exc()}")


def main():
    
    # ✅ 세션 상태 초기화
    initialize_session_state()
    
    # ✅ 사이드바 렌더링
    render_sidebar()
    
    # ✅ 이전 대화 표시
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("질문을 입력해주세요.")

    if query:
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            mart_assistant = st.session_state.mart_assistant
            try:
                with st.spinner("🔍 답변을 생성 중..."):
                    result = mart_assistant.ask(query)
                    render_chat_interface(result)
                    
            except Exception as e:
                st.error(f"❌ 답변 생성 중 오류가 발생했습니다:\n\n {traceback.format_exc()}")
                print(f"❌ 답변 생성 중 오류가 발생했습니다:\n\n {traceback.format_exc()}")


if __name__ == '__main__':
    main()
