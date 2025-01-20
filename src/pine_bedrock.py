import boto3
import streamlit as st
from loguru import logger
import json
import config
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

import warnings
warnings.filterwarnings('ignore')

def main():
    st.set_page_config(
        page_title="AWS Bedrock QA Chat",
        page_icon=":books:")

    st.title("_Private Data :red[QA Chat]_ with AWS Bedrock :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        # aws_access_key = st.text_input("AWS Access Key", type="password")
        # aws_secret_key = st.text_input("AWS Secret Key", type="password")
        aws_access_key = config.AWS_ACCESS_KEY
        aws_secret_key = config.AWS_SECRET_ACCESS_KEY
        process = st.button("Process")

    if process:
        if not (aws_access_key and aws_secret_key):
            st.info("Please add your AWS credentials to continue.")
            st.stop()

        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        logger.info("Text chunks created.")

        st.session_state.conversation = get_conversation_chain(text_chunks)

        st.session_state.processComplete = True
        st.success("Document processing complete.")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "Hello! Feel free to ask any questions about the uploaded documents."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask a question about your document..."):
        if st.session_state.conversation is None:
            st.error("The conversation chain is not initialized. Process your data first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            response, sources = st.session_state.conversation(query)

            st.markdown(response)
            with st.expander("View Sources"):
                for source in sources:
                    st.markdown(f"**Source:** {source}")

            st.session_state.messages.append({"role": "assistant", "content": response})

def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
        else:
            continue

        documents = loader.load_and_split()
        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_conversation_chain(chunks):
    def query_bedrock(query, chunks):
        # Initialize Bedrock client
        client = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Embed chunks and build a similarity search (mocking embeddings)
        embeddings = [chunk.page_content for chunk in chunks]  # Simulated embeddings

        # RAG Workflow: Find the most relevant chunks
        most_relevant_chunks = sorted(embeddings, key=lambda x: query.lower() in x.lower(), reverse=True)[:3]

        # Combine the most relevant chunks into a context
        context = "\n".join(most_relevant_chunks)

        # Prepare the payload for the model
        payload = {
            "prompt": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:",
            # "max_tokens_to_sample": 300,
            "temperature": 0.1,
            "top_p": 0.9,
        }

        try:
            # Invoke the Bedrock model
            response = client.invoke_model(
                modelId="mistral.mistral-7b-instruct-v0:2",  # Replace with your valid model ID
                body=json.dumps(payload),  # Convert payload to JSON string
                contentType="application/json",  # Specify the content type
                accept="application/json",  # Specify the accept type
            )
            # Read and decode the response body
            response_body = response["body"].read().decode("utf-8")  # Decode bytes to string
            print(f"[DEBUG] Raw Response Body: {response_body}")

            # Parse the JSON response
            result = json.loads(response_body)["outputs"][0]["text"]
            
        except client.exceptions.ValidationException as e:
            print(f"Validation Error: {str(e)}")
            raise e

        # Return the model output and the sources (relevant chunks)
        return result, most_relevant_chunks

    def conversation_chain(query):
        response, sources = query_bedrock(query, chunks)
        return response, sources

    return conversation_chain
if __name__ == '__main__':
    main()
