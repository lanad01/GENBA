# from langchain.schema import Runnable
from langchain.schema import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from typing import List, Type, Dict

from langchain.schema import BaseRetriever
from itertools import chain

RetrieverLike = BaseRetriever

# 경로 설정
dir_root = "c:\\Users\\user\\RAG\\17-News"

# 직접 정의한 Runnable 사용
class Runnable:
    def run(self, query):
        raise NotImplementedError("Subclasses should implement this!")

def kiwi_tokenize(text):
    """
    주어진 텍스트를 형태소 단위로 토큰화합니다.

    Args:
        text (str): 입력 텍스트.

    Returns:
        List[str]: 토큰화된 단어 리스트.
    """
    kiwi = Kiwi()
    tokens = [word for word, *_ in kiwi.tokenize(text)]
    return tokens


class CustomBM25Retriever(Runnable):
    def __init__(self, documents, k=5):
        self.documents = documents
        self.tokenized_docs = [kiwi_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.k = k

    def invoke(self, query: str, include_words: List[str] = None, exclude_words: List[str] = None) -> List[Document]:
        tokenized_query = kiwi_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]

        filtered_results = []
        relevant_scores = []

        for idx in top_k_indices:
            documents = self.documents[idx]
            score = scores[idx]

            if exclude_words and all(term in documents.page_content for term in exclude_words):
                print(f"[debug] [bm25] 키워드 제외 조건: {documents}")
                continue

            if include_words and not all(term in documents.page_content for term in include_words):
                print(f"[debug] [bm25] 포함 키워드 누락 조건: {documents}")
                continue

            filtered_results.append(
                Document(
                    page_content=documents.page_content,
                    metadata={**documents.metadata, "bm25_score": score},
                )
            )
            relevant_scores.append(scores[idx])

        # Relevant scores 정규화 (Min-Max Scaling)
        if relevant_scores:
            min_sim = min(relevant_scores)
            max_sim = max(relevant_scores)
            for doc in filtered_results:
                doc.metadata["normalized_score"] = (doc.metadata["bm25_score"] - min_sim) / (max_sim - min_sim) if max_sim != min_sim else 1.0

        return filtered_results


class CustomChromaRetriever(Runnable):
    def __init__(self, collection, k=5, limit_distance=130):
        self.collection = collection
        self.k = k
        self.limit_distance = limit_distance
    

    def _create_where_document(self, include_words, exclude_words):
        if (not include_words) and (not exclude_words):
            return None

        contains_conditions = [{"$contains": term} for term in include_words]

        if include_words:
            conditions = [{"$contains": term} for term in include_words]
        else:
            conditions = []

        if exclude_words:
            conditions += [{"$not_contains": term} for term in exclude_words]

        if len(conditions) == 1:
            return conditions[0]
        else:
            return {"$or": conditions}
        
    def invoke(self, query: str, include_words: List[str] = None, exclude_words: List[str] = None) -> List[Document]:
        m_path = f"{dir_root}/model/sentence"
        embeddings = SentenceTransformerEmbeddings(
            model_name=m_path,
            model_kwargs={"device": "cpu"}
        )
        # embeddings = SentenceTransformer("jhgan/ko-sroberta-multitask")

        emb_query = embeddings.embed_documents([query])

        where_document = self._create_where_document(include_words, exclude_words) if include_words or exclude_words else None

        if where_document is None:
            results = self.collection.query(
                query_embeddings=emb_query,
                n_results=self.k,
                include=["metadatas", "documents", "distances"]
            )
        else:
            results = self.collection.query(
                query_embeddings=emb_query,
                n_results=self.k,
                include=["metadatas", "documents", "distances"],
                where_document=where_document
            )
            
        documents = []
        similarities = []
        for doc, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            if distance > self.limit_distance:
                print(f"distance [{distance}] 제외")
                continue

            ## distance를 similarity로 변환 (Inverse Transformation)
            similarity = 1 / (1 + distance)
            similarities.append(similarity)

            metadata["similarity"] = similarity
            metadata["distance"] = distance

            documents.append(Document(page_content=doc, metadata=metadata))

        # similarity를 정규화 (Min-Max Scaling)
        if similarities:
            min_sim = min(similarities)
            max_sim = max(similarities)
            for doc in documents:
                doc.metadata["normalized_score"] = (doc.metadata["similarity"] - min_sim) / (max_sim - min_sim) if max_sim != min_sim else 1.0
        else:
            return []

        return documents


class CustomEnsembleRetriever():
    def __init__(self, retrievers: List[RetrieverLike], weights=None):
        print(f'RET | {retrievers}')
        self.retrievers = retrievers
        self.weights = weights

    def invoke(self, query, include_words=None, exclude_words=None):
        retrieve_list = []
        for retriever in self.retrievers:
            retriever_results = retriever.invoke(query, include_words, exclude_words)
            retrieve_list.append(retriever_results)

        return self.rank_fusion(retrieve_list)  # 앙상블된 결과 반환
    
    def unique_by_key(iterable, key):
        seen = set()
        for item in iterable:
            k = key(item)
            if k not in seen:
                seen.add(k)
                yield item

    def rank_fusion(self, doc_lists):
        id_key = None
        rrf_score: Dict[str, float] = defaultdict(float)

        # Duplicated contents across retrievers are collapsed & scored cumulatively
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[doc.page_content if id_key is None else doc.metadata[id_key]] += weight / (rank + 60)

        # Docs are deduplicated by their contents then sorted by their scores
        all_docs = chain.from_iterable(doc_lists)
        # sorted_docs = sorted(
        #     unique_by_key(
        #         all_docs,
        #         key=lambda doc: doc.page_content if id_key is None else doc.metadata[id_key],
        #     ),
        #     reverse=True,
        #     key=lambda doc: rrf_score[doc.page_content if id_key is None else doc.metadata[id_key]],
        # )

        return all_docs
