from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

connection = "postgresql+psycopg://pinetree:pinetree123@localhost:1989/postgres"
collection_name = "test"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def pgvector(embeddings=embeddings, collection_name=collection_name, connection=connection, use_jsonb=True):
    """도커 컨테이너에서 돌아가는 pgvector 벡터 스토어를 생성하는 함수입니다."""
    return PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=use_jsonb
    )