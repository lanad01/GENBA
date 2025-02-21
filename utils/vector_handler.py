# 표준 라이브러리
import os
import json
from glob import glob
from pathlib import Path

# 서드파티 라이브러리
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader, 
    UnstructuredPowerPointLoader,
    JSONLoader,
    CSVLoader
)

def load_vectorstore():
    if os.path.exists("./vectordb"):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        try:
            return FAISS.load_local("./vectordb", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ 벡터스토어 로드 중 오류 발생: {e}")
            return None
    return None


def get_text(docs):
    """문서 텍스트 추출"""
    doc_list = []
    for doc in docs:
        file_name = f'../documents/{doc.name}'
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())

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
    """텍스트 청크 분할"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )
    return text_splitter.split_documents(text)

def rebuild_vectorstore_without_document(doc_to_remove):
    """특정 문서를 제외하고 vectorstore 재구축"""
    try:
        document_list = load_document_list()
        if doc_to_remove not in document_list:
            return False
            
        print(f"🔄 Vectorstore 재구축 시작 (제외 문서: {doc_to_remove})")
        
        # 남은 문서들로 새로운 문서 리스트 생성
        remaining_docs = []
        for doc_name in document_list:
            if doc_name != doc_to_remove:
                file_path = Path(f"../documents/{doc_name}")
                if file_path.exists():
                    # 파일 타입에 따른 문서 로드
                    if file_path.suffix == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix == '.docx':
                        loader = Docx2txtLoader(str(file_path))
                    elif file_path.suffix == '.pptx':
                        loader = UnstructuredPowerPointLoader(str(file_path))
                    elif file_path.suffix == '.json':
                        loader = JSONLoader(file_path=str(file_path), jq_schema='.[]', text_content=False)
                    elif file_path.suffix == '.csv':
                        loader = CSVLoader(str(file_path), encoding='utf-8')
                    elif file_path.suffix == '.xlsx':
                        df = pd.read_excel(file_path)
                        documents = []
                        for _, row in df.iterrows():
                            content = f"테이블명: {row['테이블명']}, 컬럼명: {row['컬럼명']}, 컬럼한글명: {row['컬럼한글명']}, 설명: {row['컬럼설명']}, 데이터 타입: {row['DATATYPE']}"
                            documents.append(Document(page_content=content, metadata={"source": str(file_path)}))
                        remaining_docs.extend(documents)
                        continue
                    elif file_path.suffix == '.txt':
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        documents = [Document(page_content=content, metadata={"source": str(file_path)})]
                        remaining_docs.extend(documents)
                        continue
                    
                    documents = loader.load()
                    remaining_docs.extend(documents)
                    print(f"✅ 문서 로드 완료: {doc_name}")
        
        # 남은 문서가 없는 경우 vectordb 파일 삭제
        if not remaining_docs:
            vectordb_path = Path("./vectordb")
            if vectordb_path.exists():
                import shutil
                shutil.rmtree(vectordb_path)
                print("✅ 마지막 문서 삭제로 vectordb 폴더 제거")
            return True
            
        # 남은 문서가 있는 경우 vectorstore 재구축
        text_chunks = get_text_chunks(remaining_docs)
        vectorstore = get_vectorstore(text_chunks)
        vectorstore.save_local("./vectordb")
        
        print(f"✅ Vectorstore 재구축 완료 (총 {len(text_chunks)} chunks)")
        return True
    
    except Exception as e:
        print(f"❌ Vectorstore 재구축 중 오류 발생: {e}")
        return False
    
    
    
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
      
        
def get_vectorstore(text_chunks):
    """벡터스토어 생성"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb
