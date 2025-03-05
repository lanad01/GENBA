# 표준 라이브러리
import os
import json
from glob import glob
from pathlib import Path

# 서드파티 라이브러리
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader, 
    UnstructuredPowerPointLoader,
    JSONLoader,
    CSVLoader
)
from dotenv import load_dotenv
load_dotenv()

VECTOR_DB_SESSION_PATH = "./vector_db_session"
VECTOR_DB_BASE_PATH = "./vectordb"

def load_vectorstore(db_path):
    if os.path.exists(db_path):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        try:
            return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ 벡터스토어 로드 중 오류 발생: {e}")
            return None
    print(f"⚠️ 해당 경로에 벡터스토어가 존재하지 않습니다: {db_path}")
    return None


def get_text(docs, document_list_path):
    """문서 텍스트 추출"""
    doc_list = []
    for doc in docs:
        # document_list_path를 Path 객체로 변환하여 파일 경로 생성
        file_path = Path(document_list_path) / doc.name
        
        # 디렉토리가 없으면 생성
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 파일 저장
        with open(file_path, "wb") as file:
            file.write(doc.getvalue())

        # 파일 타입에 따른 처리
        if file_path.suffix == '.pdf':
            loader = PyPDFLoader(str(file_path))
            documents = loader.load_and_split()
        elif file_path.suffix == '.docx':
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load_and_split()
        elif file_path.suffix == '.pptx':
            loader = UnstructuredPowerPointLoader(str(file_path))
            documents = loader.load_and_split()
        elif file_path.suffix == '.json':
            loader = JSONLoader(file_path=str(file_path), jq_schema='.[]', text_content=False)
            documents = loader.load()
        elif file_path.suffix == '.csv':
            loader = CSVLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
        elif file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)
            documents = []
            for _, row in df.iterrows():
                content = f"테이블명: {row['테이블명']}, 컬럼명: {row['컬럼명']}, 컬럼한글명: {row['컬럼한글명']}, 설명: {row['컬럼설명']}, 데이터 타입: {row['DATATYPE']}"
                documents.append(Document(page_content=content, metadata={"source": str(file_path)}))
        elif file_path.suffix == '.txt':
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            documents = [Document(page_content=content, metadata={"source": str(file_path)})]
        
        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    """텍스트 청크 분할"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )
    return text_splitter.split_documents(text)

def rebuild_vectorstore_without_document(doc_to_remove, document_list_path):
    """특정 문서를 제외하고 vectorstore 재구축"""
    try:
        document_list = load_document_list(document_list_path=document_list_path)
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
            vectordb_path = Path(VECTOR_DB_BASE_PATH)
            if vectordb_path.exists():
                import shutil
                shutil.rmtree(vectordb_path)
                print("✅ 마지막 문서 삭제로 vectordb 폴더 제거")
            return True
            
        # 남은 문서가 있는 경우 vectorstore 재구축
        text_chunks = get_text_chunks(remaining_docs)
        vectorstore = get_vectorstore(text_chunks)
        vectorstore.save_local(VECTOR_DB_BASE_PATH)
        
        print(f"✅ Vectorstore 재구축 완료 (총 {len(text_chunks)} chunks)")
        return True
    
    except Exception as e:
        print(f"❌ Vectorstore 재구축 중 오류 발생: {e}")
        return False
    
def load_document_list(document_list_path):
    """저장된 문서 목록 로드"""
    try:
        with open(f"{document_list_path}/document_list.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_document_list(document_list_path, document_list):
    """문서 목록 저장"""
    with open(f'{document_list_path}/document_list.json', "w", encoding="utf-8") as f:
        json.dump(list(set(document_list)), f, ensure_ascii=False, indent=2)
      
def get_vectorstore(text_chunks):
    """벡터스토어 생성"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_vector_db_path(thread_id):
    """세션 ID 기반으로 개별 벡터DB 경로 생성"""
    return os.path.join(VECTOR_DB_SESSION_PATH, f"{thread_id}_vectorstore")

def initialize_vector_store(thread_id):
    """세션별 FAISS 벡터스토어 초기화 및 불러오기"""
    vector_db_path = get_vector_db_path(thread_id)
    
    if os.path.exists(vector_db_path):
        return FAISS.load_local(vector_db_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        return FAISS.from_texts([""], OpenAIEmbeddings())

def save_chat_to_vector_db(internal_id, query, response):
    """사용자의 질문과 AI 응답을 벡터DB에 저장"""
    try:
        # print(f"🔢 [save_chat_to_vector_db] 벡터DB 저장 시작 (세션: {thread_id})")
        # print(f"🔢 [save_chat_to_vector_db] 벡터DB 저장 시작 (query: {query})")
        # print(f"🔢 [save_chat_to_vector_db] 벡터DB 저장 시작 (응답:\n {response})")
        vectorstore = initialize_vector_store(internal_id)  # 세션별 벡터스토어 로드
        
        # response가 문자열인 경우와 딕셔너리인 경우를 모두 처리
        if isinstance(response, str):
            document_text = f"""
사용자 질문: {query}
AI 응답: {response}
"""
        else:
            document_text = f"""
사용자 질문: {query}
AI 응답: {response.get("content", "응답 없음")}
실행된 코드: {response.get("validated_code", "코드 없음")}
분석 결과: {response.get("analytic_result", "결과 없음")}
생성된 인사이트: {response.get("insights", "인사이트 없음")}
리포트: {response.get("report", "리포트 없음")}
"""
        
        # ✅ 벡터DB에 저장
        vectorstore.add_texts([document_text])
        vector_db_path = os.path.join(VECTOR_DB_SESSION_PATH, f"{internal_id}_vectorstore")
        vectorstore.save_local(vector_db_path)
        print(f"📩 벡터DB 저장 완료 (세션: {internal_id})")

    except Exception as e:
        print(f"❌ [save_chat_to_vector_db] 벡터DB 저장 중 오류 발생: {e}")

def search_similar_questions(internal_id, query, top_k=5, similarity_threshold=0.7):
    """해당 쓰레드의 질문-답변 이력이 쌓여있는 벡터DB에서 사용자의 현재 질문과 유사한 질문 검색"""
    vectorstore = initialize_vector_store(internal_id)  # 세션별 벡터스토어 로드
    
    # 🔎 유사도 점수와 함께 검색 실행
    search_results = vectorstore.similarity_search_with_score(query, k=top_k*2)  # 더 많은 결과를 가져와서 필터링
    
    # 유사도 점수가 threshold를 넘는 결과만 필터링
    filtered_results = []
    seen_content = set()  # 중복 콘텐츠 확인용 집합
    
    for doc, score in search_results:
        # FAISS의 score는 L2 거리이므로 코사인 유사도로 변환 (1 - score/2가 코사인 유사도의 근사값)
        cosine_sim = 1 - (score / 2)
        
        if cosine_sim >= similarity_threshold:
            # 콘텐츠 핵심 부분 추출 (사용자 질문 부분만)
            content_key = ""
            for line in doc.page_content.split('\n'):
                if "사용자 질문:" in line:
                    content_key = line.strip()
                    break
            
            # 중복 콘텐츠 건너뛰기
            if content_key and content_key in seen_content:
                continue
            
            # 가중치 계산 (콘텐츠 품질 기반)
            weight = 1.0
            if "validated_code: None" in doc.page_content or "코드 없음" in doc.page_content:
                weight *= 0.8  # 코드가 없는 경우 가중치 감소
            
            if "인사이트: None" in doc.page_content or "인사이트 없음" in doc.page_content:
                weight *= 0.9  # 인사이트가 없는 경우 가중치 감소
                
            if "실행된 코드:" in doc.page_content and "코드 없음" not in doc.page_content:
                weight *= 1.3  # 실행된 코드가 있는 경우 가중치 증가
                
            if "생성된 인사이트:" in doc.page_content and "인사이트 없음" not in doc.page_content:
                weight *= 1.2  # 인사이트가 있는 경우 가중치 증가
            
            # 최종 스코어 조정
            adjusted_score = cosine_sim * weight
            
            if content_key:
                seen_content.add(content_key)
            
            filtered_results.append((doc, adjusted_score, cosine_sim))
    
    # 조정된 점수로 상위 결과 선택
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    filtered_results = filtered_results[:1]  # 상위 1개만 유지
    return filtered_results

def delete_thread_vectorstore(internal_id):
    """스레드의 벡터DB 삭제"""
    try:
        vector_db_path = get_vector_db_path(internal_id)
        if os.path.exists(vector_db_path):
            import shutil
            shutil.rmtree(vector_db_path)
            print(f"✅ 벡터DB 삭제 완료 (스레드: {internal_id})")
        return True
    except Exception as e:
        print(f"❌ 벡터DB 삭제 중 오류 발생: {e}")
        return False
