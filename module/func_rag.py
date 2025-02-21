
import os
import joblib
import pickle, time
import shutil, hashlib, pickle, joblib

# from langchain_community.embeddings.sentence_transformer import SentenceTransformer
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceEmbeddings

# User Package
from module.CustomRetriever import CustomChromaRetriever, CustomBM25Retriever, CustomEnsembleRetriever

class RAGInitializer:
    def __init__(self, base_dir, chromadb_path, model_path):
        """
        RAGInitializer 초기화.

        Args:
            base_dir (str): 기본 디렉토리 경로.
            chromadb_path (str): ChromaDB 경로.
            model_path (str): SentenceTransformer 모델 경로.
        """
        self.base_dir = base_dir
        self.chromadb_path = chromadb_path
        self.model_path = model_path
        self.data_dir = os.path.join(base_dir, "data")
        self.model_kwargs = {"device": "cpu"}

    def df_to_bm25docs(self, df):
        """
        DataFrame을 BM25 문서 리스트로 변환.

        Args:
            df (pd.DataFrame): 변환할 DataFrame.

        Returns:
            list: BM25 문서 리스트.
        """
        documents = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=row['content'],
                metadata={
                    "title": row['title'],
                    "posts_dt": str(row['bas_dt']),
                    "url": row['url'],
                    "search_src": "bm25"
                }
            )
            documents.append(doc)
        return documents
    

    def initialize(self, col_name, bm25_joblib):
        """
        RAG 초기화 및 엔진 반환.

        Args:
            col_name (str): 컬렉션 이름.
            bm25_joblib (str): BM25 retriever joblib 파일 경로.

        Returns:
            CustomEnsembleRetriever: 초기화된 앙상블 리트리버.
        """
        # BM25 Retriever 초기화
        if os.path.exists(bm25_joblib):
            with open(bm25_joblib, "rb") as f:
                bm25_retriever = joblib.load(f)
        else:
            pkl_path = os.path.join(self.data_dir, f"{col_name}.pkl")
            with open(pkl_path, "rb") as f:
                uniq_df = pickle.load(f)
            bm25_docs = self.df_to_bm25docs(uniq_df[~uniq_df["tic_txc"].isna()])
            bm25_retriever = CustomBM25Retriever(bm25_docs)
            with open(bm25_joblib, "wb") as f:
                joblib.dump(bm25_retriever, f)

        # Sentence Transformer Embeddings 초기화
        # embeddings = SentenceTransformer(
        #     model_name=self.model_path,
        #     model_kwargs=self.model_kwargs
        # )
        
        print(f"ChromaDB Retriever 초기화 이전")

        # ChromaDB Retriever 초기화
        client = PersistentClient(path=self.chromadb_path)
        collection = client.get_collection(col_name)
        chroma_retriever = CustomChromaRetriever(collection, k=5)

        # 앙상블 Retriever 초기화
        ensemble_retriever = CustomEnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )

        return ensemble_retriever


    def insert_chroma(self, chromadb_path, model_path, bas_dt, df_news):
        
        # local_embeddings = SentenceTransformerEmbeddings(
        #     model_name=model_path, 
        #     model_kwargs={"device": "cuda"}
        # )
        local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        settings = Settings(
            persist_directory=chromadb_path,
            anonymized_telemetry=False,
            allow_reset=True
        )

        client = PersistentClient(path=chromadb_path, settings=settings)
        collection = client.get_or_create_collection(bas_dt)

        for index, row in df_news.iterrows():
            article_content = row['content']
            title = row['title']
            if index % 500 == 0: 
                print(index)
            try:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
                chunks = text_splitter.split_text(article_content)
                for i, chunk in enumerate(chunks):
                    hash_object = hashlib.md5(row['title'].encode())
                    unique_id = hash_object.hexdigest()
                    document_id = f"{unique_id}_{i}"
                    metadata = {
                        "title": row['title'],
                        "posts_dt": str(row['bas_dt']),
                        "url": row['url'],
                        # "dup_count": row['count'],
                        "search_src": "chromadb",
                        "chunk_index": i
                    }
                    embedding = local_embeddings.embed_documents([chunk])
                    collection.add(
                        documents=[chunk],
                        metadatas=[metadata],
                        ids=[document_id],
                        embeddings=embedding
                    )
            except Exception as e:
                print(f"row = {row}")
                
            
    def delete_before_repo(self, chromadb_path, bm25_joblib):
        # settings = Settings(
        #     persist_directory=chromadb_path,
        #     anonymized_telemetry=False,
        #     allow_reset=True
        # )

        import stat
        ## bm25 파일 삭제
        if os.path.exists(f"{bm25_joblib}"):
            print(f"{bm25_joblib} 존재")
            os.chmod(bm25_joblib , stat.S_IWRITE)
            os.remove(bm25_joblib)
        else:
            print(f"{bm25_joblib} 미존재")

        ## chromadb 폴더 삭제
        if os.path.exists(f"{chromadb_path}"):
            print(f"{chromadb_path} 존재")
            os.chmod(chromadb_path , stat.S_IWRITE)
            shutil.rmtree(chromadb_path)
            # os.sync()
            time.sleep(10)
        else:
            print(f"{chromadb_path} 미존재")
