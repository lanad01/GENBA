import streamlit as st
import os
import shutil
import pandas as pd
from pathlib import Path
from langchain.schema import Document
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    JSONLoader,
    CSVLoader
)
from utils.vector_handler import load_vectorstore

DOCUMENT_DIR = "../documents"

def handle_document_upload():
    """문서 업로드 UI 및 벡터 DB 저장"""
    uploaded_files = st.sidebar.file_uploader(
        "📥 분석할 문서를 업로드하세요", 
        type=['pdf', 'docx', 'pptx', 'json', 'csv', 'xlsx', 'txt'], 
        accept_multiple_files=True
    )

    if uploaded_files:
        saved_files = save_uploaded_files(uploaded_files)
        st.sidebar.success(f"✅ {len(saved_files)}개 문서가 저장되었습니다.")

        # ✅ 벡터스토어 업데이트
        update_vectorstore(saved_files)
    
    # ✅ 문서 목록 UI
    render_uploaded_documents()


def save_uploaded_files(uploaded_files):
    """업로드된 파일을 저장"""
    os.makedirs(DOCUMENT_DIR, exist_ok=True)
    saved_files = []

    for uploaded_file in uploaded_files:
        file_path = Path(DOCUMENT_DIR) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        saved_files.append(str(file_path))
    
    return saved_files


def update_vectorstore(files):
    """업로드된 파일을 벡터 DB에 추가"""
    vectorstore = load_vectorstore()
    
    documents = []
    for file_path in files:
        loader = get_document_loader(file_path)
        if loader:
            documents.extend(loader.load())
    
    if vectorstore:
        vectorstore.add_documents(documents)
    else:
        st.error("⚠️ 벡터스토어를 로드할 수 없습니다. 먼저 문서를 등록해주세요.")


def get_document_loader(file_path):
    """파일 유형에 맞는 문서 로더 반환"""
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == "pdf":
        return PyPDFLoader(file_path)
    elif file_extension == "docx":
        return Docx2txtLoader(file_path)
    elif file_extension == "pptx":
        return UnstructuredPowerPointLoader(file_path)
    elif file_extension == "json":
        return JSONLoader(file_path=file_path, jq_schema='.[]', text_content=False)
    elif file_extension == "csv":
        return CSVLoader(file_path, encoding="utf-8")
    elif file_extension == "xlsx":
        return process_excel_file(file_path)
    elif file_extension == "txt":
        return load_text_file(file_path)
    else:
        st.warning(f"⚠️ 지원되지 않는 파일 형식: {file_extension}")
        return None


def process_excel_file(file_path):
    """엑셀 파일을 로드하여 Document 리스트로 변환"""
    df = pd.read_excel(file_path)
    documents = []

    for _, row in df.iterrows():
        content = f"테이블명: {row['테이블명']}, 컬럼명: {row['컬럼명']}, 컬럼한글명: {row['컬럼한글명']}, 설명: {row['컬럼설명']}, 데이터 타입: {row['DATATYPE']}"
        documents.append(Document(page_content=content, metadata={"source": file_path}))
    
    return documents


def load_text_file(file_path):
    """텍스트 파일을 Document 리스트로 변환"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": file_path})]


def render_uploaded_documents():
    """등록된 문서 목록 UI 및 삭제 기능"""
    st.sidebar.markdown("### 📑 등록된 문서 목록")

    document_list = get_saved_documents()
    if document_list:
        for doc in document_list:
            cols = st.sidebar.columns([0.85, 0.15])
            with cols[0]:
                st.markdown(f"- {doc}")
            with cols[1]:
                if st.button("🗑️", key=f"del_{doc}", help=f"문서 삭제: {doc}"):
                    delete_document(doc)
    else:
        st.sidebar.info("등록된 문서가 없습니다.")


def get_saved_documents():
    """저장된 문서 목록 가져오기"""
    if not os.path.exists(DOCUMENT_DIR):
        return []
    return [f for f in os.listdir(DOCUMENT_DIR) if os.path.isfile(os.path.join(DOCUMENT_DIR, f))]


def delete_document(doc_name):
    """문서 삭제 및 벡터 DB 재구축"""
    file_path = os.path.join(DOCUMENT_DIR, doc_name)

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        
        if rebuild_vectorstore_without_document(doc_name):
            st.toast(f"🗑️ '{doc_name}' 문서가 삭제되었습니다.")
            st.rerun()
        else:
            st.warning("⚠️ 벡터스토어 재구축 중 오류가 발생했습니다.")
    except Exception as e:
        st.error(f"❌ 문서 삭제 중 오류 발생: {e}")
