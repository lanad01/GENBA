import os
import pandas as pd
import streamlit as st
import json
import tiktoken
import time
from loguru import logger

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
import warnings

load_dotenv()  # .env 파일 로드
warnings.filterwarnings('ignore')

## custom modules
def main():
    st.set_page_config(page_title="분석어시스턴트", page_icon="pine.png", layout='wide')
        
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 250px;
            max-width: 250px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "RagComplete" not in st.session_state:
        st.session_state.RagComplete = None

    uploaded_files = st.file_uploader(
        "📂 파일을 업로드하세요",
        type=['pdf', 'docx', 'pptx', 'json', 'csv', 'xlsx', 'txt'],
        accept_multiple_files=True,
        
    )

    # 🔹 파일이 삭제되면 자동으로 등록 상태 리셋
    if not uploaded_files:
        st.session_state.RagComplete = None  # 파일이 없으면 상태 초기화

    process = st.button("📌 등록하기")

    if process:
        if not uploaded_files:
            st.warning("⚠️ 파일을 먼저 업로드해주세요!")
            return

        with st.spinner("⏳ 파일을 처리하는 중입니다..."):
            try:
                files_text = get_text(uploaded_files)
                text_chunks = get_text_chunks(files_text)
                vectordb = get_vectorstore(text_chunks)

                # 벡터 DB 저장
                vectordb.save_local("./vectordb")
                
                # 문서 목록 관리
                document_list = load_document_list()
                new_documents = [file.name for file in uploaded_files]
                document_list.extend(new_documents)
                save_document_list(list(set(document_list)))  # 중복 제거

                st.session_state.RagComplete = True
                logger.info("✅ 벡터 저장소 생성 완료")

            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")
                st.warning("등록한 파일이 현재 실행 중인지 확인해주세요.")
                st.session_state.RagComplete = False
                return

    # ✅ 문서 등록 완료 후 "분석 어시스턴트"로 이동하는 버튼 표시
    if st.session_state.RagComplete:
        # 등록 완료 시 사용자에게 직관적인 피드백 제공
        st.success("✅ 등록이 완료되었습니다!")
        st.toast("🎉 문맥 등록이 성공적으로 완료되었습니다!")
        # ✅ 버튼 추가: 클릭하면 "분석 어시스턴트.py"로 이동
        if st.button("🔍 분석 어시스턴트로 이동하기"):
            st.switch_page("./pages/분석 어시스턴트.py")  # Streamlit 페이지 이동

    elif st.session_state.RagComplete is None:
        st.info("📥 파일을 업로드하고 등록을 진행하세요!")

def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = f'../documents/{doc.name}'
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())

        print(f"📂 Uploaded: {file_name}")

        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()

        elif file_name.endswith('.docx'):
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()

        elif file_name.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        elif file_name.endswith('.json'):
            loader = JSONLoader(file_path=file_name, jq_schema='.[]', text_content=False)
            documents = loader.load()
            for doc in documents:
                content_dict = json.loads(doc.page_content)
                standard_term = content_dict.get("standard_term", "알 수 없음")
                similar_terms = ", ".join(content_dict.get("similar_terms", []))
                doc.page_content = f"{standard_term} (유사어: {similar_terms})" if similar_terms else f"{standard_term}"

        elif file_name.endswith('.csv'):
            loader = CSVLoader(file_name, encoding='utf-8')
            documents = loader.load()

        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(file_name)
            documents = []
            for _, row in df.iterrows():
                content = f"테이블명: {row['테이블명']}, 컬럼명: {row['컬럼명']}, 컬럼한글명: {row['컬럼한글명']}, 설명: {row['컬럼설명']}, 데이터 타입: {row['DATATYPE']}"
                documents.append(Document(page_content=content, metadata={"source": file_name}))

        elif file_name.endswith('.txt'):
            with open(file_name, "r", encoding="utf-8") as f:
                content = f.read()
            documents = [Document(page_content=content, metadata={"source": file_name})]
        
        doc_list.extend(documents)

    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )
    return text_splitter.split_documents(text)

def get_vectorstore(text_chunks):
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="jhgan/ko-sroberta-multitask",
    #     model_kwargs={'device': 'cpu'},
    #     encode_kwargs={'normalize_embeddings': True}
    # )      
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def load_document_list():
    """저장된 문서 목록 로드"""
    try:
        with open("./document_list.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_document_list(document_list):
    """문서 목록 저장"""
    with open("./document_list.json", "w", encoding="utf-8") as f:
        json.dump(list(set(document_list)), f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
