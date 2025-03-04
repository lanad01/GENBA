##################################################
### Library Import
##################################################

### Built-in Modules
import os
import re
import argparse

### Third-party Library
import faiss
from huggingface_hub import snapshot_download
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.llms import VLLM
from langchain_community.vectorstores import FAISS
from langchain.core.output_parsers import StrOutputParser
from langchain.core.prompts import ChatPromptTemplate
from langchain.core.runnables import RunnablePassthrough
from langchain.huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

##################################################
### Path
##################################################
data_path = "/data"  # 데이터 경로
model_path = "/model"  # 모델 경로

##################################################
### Functions
##################################################

### get arguments
def get_args():
    """
    Parse arguments provided in the shell script.

    Args:
        mdl_nm (str): Model name to use (default: "default", options: "Instruction", "DPO").
    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--mdl_nm", type=str, default="default", help="Specify the model name.")

    return parser.parse_args()

### Sentence Transformer Load
def load_sentence_transformer(device, model_path=model_path):
    """
    Load a pre-trained Sentence Transformer model downloaded from HuggingFace.

    Args:
        device     (str): Device to use (e.g., "cuda" or "cpu").
        model_path (str): Path to the pre-downloaded HuggingFace model.

    Returns:
        SentenceTransformer: Loaded Sentence Transformer object.
    """

    transformer_path = f"{model_path}/huggingface/sentence-transformer/all-MiniLM-L6-v2"
    sentence_transformer = SentenceTransformer(transformer_path, device=device)

    return sentence_transformer

### Embedding Load
def load_embedding(device, model_path=model_path):
    """
    Load a pre-trained Embedding model downloaded from HuggingFace.

    Args:
        device     (str): Device to use (e.g., "cuda" or "cpu").
        model_path (str): Path to the pre-downloaded HuggingFace embedding model.

    Returns:
        HuggingFaceEmbeddings: Loaded embedding object.
    """

    embedding = HuggingFaceEmbeddings(
        model_name    = f"{model_path}/huggingface/embedding/models--intfloat--multilingual-e5-large-instruct/snapshots/c9e87c786ffac96aeae",
        model_kwargs  = {"device": device},
        encode_kwargs = {"normalize_embeddings": True},
    )

    return embedding

### DB Generation
def gen_db(embedding):
    """
    Generate a FAISS vector store using the specified embedding function.

    Args:
        embedding (HuggingFaceEmbeddings): Embedding function to use.

    Returns:
        FAISS: A newly generated FAISS vector store.
    """

    index = faiss.IndexFlatL2(1024)  # Initialize FAISS index with dimension size 1024
    db = FAISS(
        embedding_function   = embedding,
        index                = index,
        docstore             = InMemoryDocstore({}),  # Initialize an empty in-memory docstore
        index_to_docstore_id = {},
    )

    return db

### DB Load
def load_db(folder_path, index_name, embedding):
    """
    Load a FAISS vector store from local storage.

    Args:
        folder_path (str)                  : Path to the folder where the DB files are stored.
        index_name  (str)                  : Name of the FAISS index file.
        embedding   (HuggingFaceEmbeddings): Embedding object to use for query generation.

    Returns:
        FAISS: The loaded FAISS vector store.
    """

    db = FAISS.load_local(
        folder_path                     = folder_path,
        index_name                      = index_name,
        embeddings                      = embedding,
        allow_dangerous_deserialization = True,
    )

    return db

### Retriever Load
def load_retriever(db, k=3):
    """
    Create a Retriever object based on a VectorStore for document retrieval.

    This method uses the Maximal Marginal Relevance (MMR) search strategy
    to retrieve documents that balance relevance and diversity.

    Args:
        db (VectorStore): The vector store database to search.
        k  (int)        : The number of documents to return (default: 3).

    Returns:
        Retriever: A retriever object configured to fetch `k` documents.
    """

    retriever = db.as_retriever(
        search_type = "mmr",  # Maximal Marginal Relevance
        search_kwargs = {
            "k"           : k,    # Number of documents to retrieve
            "fetch_k"     : 100,  # Number of documents to consider for MMR
            "lambda_mult" : 0.6,  # Diversity adjustment (0.0: similarity-only, 1.0: diversity-only)
        }
    )

    return retriever

### Reranker Load
def load_reranker(device, model_path=model_path, top_n=3):
    """
    Create a Reranker object using a Cross-Encoder model.

    This method generates a reranker based on a HuggingFace Cross-Encoder,
    which is used for reranking retrieved documents.

    Args:
        device     (str): Device to use (e.g., "cuda" or "cpu").
        model_path (str): Path to the pre-trained Cross-Encoder model.
        top_n      (int): Number of top documents to return after reranking (default: 3).

    Returns:
        CrossEncoderReranker: A Cross-Encoder Reranker object for document reranking.
    """

    cross_encoder = HuggingFaceCrossEncoder(
        model_name=f"{model_path}/huggingface/cross-encoder/models--Dongjin-kr/ko-reranker/snapshots",
        model_kwargs={"device": device},
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

    return reranker

### Reranker Retriever Load
def load_reranker_retriever(reranker, retriever):
    """
    Combine a Reranker with a Retriever to create a reranking retriever.

    This function integrates a Cross-Encoder Reranker with a Retriever
    to improve the relevance of retrieved documents.

    Args:
        reranker  (CrossEncoderReranker): Reranker object to apply reranking.
        retriever (Retriever)           : Retriever object for initial document retrieval.

    Returns:
        ContextualCompressionRetriever: A retriever with reranking capabilities.
    """

    reranker_retriever = ContextualCompressionRetriever(
        base_compressor = reranker,
        retriever       = retriever,
    )

    return reranker_retriever

### LLM Load
def load_llm(model_path=model_path, mdl_nm='default'):
    """
    Load a vLLM (Large Language Model) based on the specified model name and path.

    This function determines the model type (e.g., Instruction-tuned, DPO, or default),
    and loads it from a local directory. If the model is not present locally,
    it downloads the model using HuggingFace snapshot.

    Args:
        model_path (str): Path to the directory containing the model files.
        mdl_nm     (str): Model name to load (default: 'default').
                          Options: 'Instruction', 'DPO', or 'default'.

    Returns:
        VLLM: A vLLM object loaded with the specified model.
    """

    # Model directory determination
    if mdl_nm == "Instruction":
        local_dir = f"{model_path}/finetuned/lora-llama3.2-output_merged_instruction"
        print("####### Instruction fine-tuned LLM model loaded #######")
    elif mdl_nm == "DPO":
        local_dir = f"{model_path}/finetuned/lora-llama3.2-output_merged_dpo"
        print("####### DPO fine-tuned LLM model loaded #######")
    else:  # Default model
        repo_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
        local_dir = f"{model_path}/huggingface/llm" + repo_id.replace('/', '_')
        if not os.path.isdir(local_dir):
            snapshot_download(
                repo_id                = repo_id,
                local_dir              = local_dir,
                local_dir_use_symlinks = False,
            )
        print("####### Default LLM model loaded #######")

    # Load vLLM
    llm = VLLM(
        model             = local_dir,
        max_new_tokens    = 4096,           # Maximum number of new tokens to generate
        trust_remote_code = True,           # Allow execution of remote code from the downloaded model
        dtype             = "float16",      # Use float16 precision for model weights

        # Sampling control variables for output diversity
        temperature       = 0.01,           # Controls randomness (lower = conservative, higher = creative)
        top_p             = 0.95,           # Nucleus sampling: consider tokens with cumulative probability ≤ top_p
        top_k             = -1,             # Restrict to the top_k tokens by probability (default: -1 for unlimited)

        # Repetition penalty controls
        presence_penalty  = 0.5,            # Penalizes new tokens based on their presence (-2 ≤ v ≤ 2)
        # frequency_penalty = 0.2,            # Penalizes frequent tokens to reduce repetition (-2 ≤ v ≤ 2)
        
        # vLLM Engine Arguments
        vllm.kwargs = {
            gpu_memory_utilization : 0.7,   # Allocates 70% of GPU memory
            max_model_memory       : 16384, # Maximum model memory allocation
            max_batched_tokens     : 16384, # Maximum tokens in a single batch
        }
    )

    return llm

### Prompt Load - Generation Response
def load_prompt():
    """
    Load and return the prompt template to be applied to the response generation chain.

    Returns:
        ChatPromptTemplate: A formatted prompt template for generating responses.
    """
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Follow these instructions to generate a response to the user's question:
        1. **Respond based only on the "CONTEXT"**:
           - If the answer cannot be determined from the "CONTEXT", respond with: "I don't know."
        
        2. **Response Requirements**:
           - All responses must be written in **Korean**.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question}
        Context: {context}
        Answer:
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt = ChatPromptTemplate.from_template(template)

    return prompt

### Chain Creation - Generation Response
def create_chain(prompt, llm):
    """
    Create and return a response generation chain using the provided prompt and LLM.

    Args:
        prompt (ChatPromptTemplate): The prompt template to use for generating responses.
        llm    (VLLM)              : The language model object for generating responses.

    Returns:
        Chain: A response generation chain object.
    """

    chain = (
        {'context': RunnablePassthrough(), 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


### Prompt Load - Query Reformation
def load_prompt_query_reformation():
    """
    Load and return the prompt template for query reformation.

    This function is used to generate alternative versions of a user's query
    to retrieve relevant documents from a vector database.

    Returns:
        ChatPromptTemplate: A formatted prompt template for query reformation.
    """

    template = """
    You are an AI language model assistant.
    Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database.
    The goal is to help the user overcome some of the limitations of the distance-based similarity search.
    by generating multiple perspectives on the user's query.

    Provide these alternative questions separated by newlines.
    All responses must be written in **Korean**.

    Example:
    Question: SlipRing 내부 윤활 작업방법은?
    Answer:
    1. SlipRing 내부 윤활 작업 방법에 대해 설명해 주세요.
    2. SlipRing 내부 윤활 작업의 방법과 절차를 설명해 주세요.
    3. SlipRing 내부 윤활 작업 방법에 대한 정보를 제공해 주세요.
    4. SlipRing 내부 윤활 작업 절차를 설명해 주세요.
    5. SlipRing 내부 윤활 작업 방법에 대한 설명을 제공해 주세요.

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    return prompt

### Chain Creation - Query Reformation
def create_chain_query_reformation(prompt, llm):
    """
    Create and return a query reformation chain using the provided prompt and LLM.

    Args:
        prompt (ChatPromptTemplate): The prompt template for query reformation.
        llm    (VLLM)              : The language model object for generating reformed queries.

    Returns:
        Chain: A query reformation chain object.
    """

    chain = (
          prompt
        | llm
        | StrOutputParser()
        | (lambda x: [q[0] for q in [re.findall(r'^[1-5]\..*', val.strip()) for val in x.split('\n')] if q][:5])
    )

    return chain

### Prompt Load - Evaluation Context
def load_prompt_evaluation_context():
    """
    Load and return the prompt template for evaluating and selecting between two contexts.

    This function generates a prompt for comparing two parsing results (`context1`, `context2`)
    based on their suitability for FAISS-based vector similarity search.

    Returns:
        ChatPromptTemplate: A formatted prompt template for context evaluation.
    """

    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a language model evaluator.
    Your task is to compare two contexts and select the one that is better suited for FAISS-based vector similarity search.
    1. Compare Context1 with Context2 based on their suitability for FAISS-based vector similarity search.
    2. Select the context that is clearer, more structured, and semantically richer.
    3. If both contexts seem equally good or if the comparison is unclear, select Context1 by default.

    Follow these rules:
    1. Always prioritize the context with clearer, more structured, and semantically richer information.
    2. If it is unclear which context is better, or if both contexts are equally suitable, select Context1.
    3. Output only the full text of the chosen context without any additional text or commentary.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Context1: {context1}
    Context2: {context2}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt = ChatPromptTemplate.from_template(template)

    return prompt

### Chain Creation - Evaluation Context
def create_chain_evaluation_context(prompt, llm):
    """
    Create and return a context evaluation chain using the provided prompt and LLM.

    Args:
        prompt (ChatPromptTemplate): The prompt template for context evaluation.
        llm    (VLLM)              : The language model object for generating evaluations.

    Returns:
        Chain: A context evaluation chain object.
    """

    chain = (
        {"context1": RunnablePassthrough(), "context2": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain