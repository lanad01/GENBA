import os
from glob import glob
import streamlit as st
from loguru import logger
from openai import OpenAIError
from dotenv import load_dotenv
load_dotenv()  # 환경 변수 로드
import warnings
import traceback  # 추가된 부분

# ✅ LangGraph 및 LangChain 관련 모듈
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# ✅ AI Assistant LangGraph Class Import
from utils.ai_agent import AIAnalysisAssistant  # ai_agent.py에 해당 클래스 정의

warnings.filterwarnings('ignore')

st.set_page_config(page_title="분석 어시스턴트", page_icon="🔍", layout='wide')

# ✅ 스타일 최적화
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 250px;
            max-width: 250px;
        }
        .stChatMessage { max-width: 90% !important; }
        .stMarkdown { font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
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

def main():
    # ✅ OpenAI API Key 확인
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        st.warning("⚠️ OpenAI API Key가 설정되지 않았습니다. 환경 변수를 확인하세요.")
        return

    # ✅ 벡터스토어 (FAISS) 로드
    if "vectorstore" not in st.session_state:
        with st.spinner("🔄 문맥을 불러오는 중..."):
            vectorstore = load_vectorstore()
            if vectorstore:
                st.session_state["vectorstore"] = vectorstore
            else:
                st.warning("⚠️ 문맥이 등록되지 않았습니다. 먼저 '문맥 등록' 페이지에서 문서를 등록해주세요.")
                return
    else:
        vectorstore = st.session_state["vectorstore"]

    # ✅ LangGraph 기반 AI Assistant 초기화
    if "assistant" not in st.session_state:
        with st.spinner("🤖 AI Agent를 로드하는 중..."):
            st.session_state.assistant = AIAnalysisAssistant(vectorstore, openai_api_key)

    # ✅ 문서 목록을 좌상단에 표시
    with st.sidebar:
        st.subheader("📄 등록된 문서 목록")
        document_list = st.session_state.get("document_list", [])
        if document_list:
            for doc in document_list:
                st.write(f"- {doc}")
        else:
            st.info("등록된 문서가 없습니다.")

    # ✅ 문맥과 API Key가 정상적으로 등록된 경우 채팅 활성화
    st.success("✅ 문맥이 정상적으로 로드되었습니다!")
    st.success("✅ OpenAI API Key가 정상적으로 등록되었습니다!")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요! AI 분석 어시스턴트입니다. 무엇이든 물어보세요!"}]

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
            assistant = st.session_state.assistant
            try:
                with st.spinner("🔍 답변을 생성 중..."):
                    result = assistant.ask(query)
                    # print(f"🔍 result: {result}")
                    # response = result['generation']\
                    response =  ["messages"][-1].content
                    
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

                    # ✅ 차트 이미지 출력
                    if "chart_filename" in result and result["chart_filename"]:
                        st.markdown("**차트 결과:**")
                        st.image(f"./img/{result['chart_filename']}", caption="차트 결과")

                    # ✅ 인사이트 출력
                    if "insights" in result:
                        st.markdown("**인사이트:**")
                        st.markdown(result["insights"])

                    # ✅ 엑셀 다운로드 버튼 추가
                    if "report_filename" in result and result["report_filename"] == "success":
                        excel_file_path = max(glob(os.path.join("../output", '*.xlsx')), key=os.path.getctime)  # 생성된 엑셀 파일 경로
                        if os.path.exists(excel_file_path):
                            with open(excel_file_path, 'rb') as file:
                                st.download_button(
                                    label="📥 엑셀 보고서 다운로드",
                                    data=file,
                                    file_name='final_report.xlsx',
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                )

                    st.session_state["messages"].append({"role": "assistant", "content": response})

                    if "documents" in result:
                        with st.expander("📂 참고 문서 확인"):
                            st.markdown('<div class="reference-doc">', unsafe_allow_html=True)
                            for i, doc in enumerate(source_documents[:3]):
                                st.markdown(f"📄 **출처 {i+1}:** {doc.metadata['source']}")
                                st.markdown(f"> {doc.page_content[:200]} ...")
                            st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ 답변 생성 중 오류가 발생했습니다:\n\n {traceback.format_exc()}")


        st.session_state["messages"].append({"role": "assistant", "content": response})

# ✅ FAISS 인덱스 로드 함수
def load_vectorstore():
    if os.path.exists("./vectordb"):
        # embeddings = HuggingFaceEmbeddings(
        #     model_name="jhgan/ko-sroberta-multitask",
        #     model_kwargs={'device': 'cpu'},
        #     encode_kwargs={'normalize_embeddings': True}
        # )
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        return FAISS.load_local("./vectordb", embeddings, allow_dangerous_deserialization=True)
    else:
        return None  # 저장된 인덱스가 없을 경우 None 반환

if __name__ == '__main__':
    main()
