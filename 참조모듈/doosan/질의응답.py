##################################################
### Library Import
##################################################
### Built-in Modules
import os
import time
import pickle

### Third-party Library
import torch
import streamlit as st

### Local Library
from model import *
from utils import *  

##################################################
### 초기화면
##################################################
st.set_page_config(page_title="DOOSAN 풍력발전 O&M AI Chat", layout="wide")
st.title(":blue[_DOOSAN 풍력발전 O&M AI Chat")
st.subheader("좌측 메뉴에서 질문의 대상이 되는 풍력발전기 모델을 선택해주세요.")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 420px;
        max-width: 420px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
##################################################
### Setting
##################################################
### GPU Setting
os.environ["USER_AGENT"]           = "myagent"
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_args()

### Path 지정
data_path = "../data"
model_path = "../model"

### WTG 모델 선택
WTG_model_list = ["01.WINDS3000_91_100", "02.WINDS3000_134", "03.WINDS5560_140"]

with st.sidebar:
    clear_btn = st.button("대화내용 삭제하기")
    WTG_model = st.selectbox("풍력발전기 모델을 선택해주세요.", WTG_model_list)

### 대화내용 삭제 버튼
if clear_btn:
    st.session_state["messages"].clear()

### Query 처리 pickle 파일 생성
if 'queue.pkl' not in os.listdir():
    pd.DataFrame([{'id': 0, 'status': 'finish'}]).to_pickle('queue.pkl')

##################################################
### Session State 초기화/생성
##################################################
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "message_type" not in st.session_state:
    st.session_state["message_type"] = []

if "options" not in st.session_state:
    st.session_state["options"] = []

if "log" not in st.session_state:
    st.session_state["log"] = {}

if "process" not in st.session_state:
    st.session_state["process"] = 0

### Embedding 생성
if "embedding" not in st.session_state:
    st.session_state["embedding"] = load_embedding(device)

### DB 생성
# EW DB: Error & Warning DB
if "EW_100" not in st.session_state:
    st.session_state["EW_100"] = load_db(f'{data_path}/db_faiss', 'EW_100', st.session_state["embedding"])

if "EW_134" not in st.session_state:
    st.session_state["EW_134"] = load_db(f'{data_path}/db_faiss', 'EW_134', st.session_state["embedding"])

if "EW_140" not in st.session_state:
    st.session_state["EW_140"] = load_db(f'{data_path}/db_faiss', 'EW_140', st.session_state["embedding"])

if "EW_db_list" not in st.session_state:
    st.session_state["EW_db_list"] = [st.session_state["EW_100"], st.session_state["EW_134"], st.session_state["EW_140"]]

### MD DB: Manual & Drawing DB
if "MD_100" not in st.session_state:
    st.session_state["MD_100"] = load_db(f'{data_path}/db_faiss', 'MD_100', st.session_state["embedding"])

if "MD_134" not in st.session_state:
    st.session_state["MD_134"] = load_db(f'{data_path}/db_faiss', 'MD_134', st.session_state["embedding"])

if "MD_140" not in st.session_state:
    st.session_state["MD_140"] = load_db(f'{data_path}/db_faiss', 'MD_140', st.session_state["embedding"])

### Retriever 생성
if "retriever1" not in st.session_state:
    st.session_state["retriever1"] = load_retriever(st.session_state["MD_100"])

if "retriever2" not in st.session_state:
    st.session_state["retriever2"] = load_retriever(st.session_state["MD_134"])

if "retriever3" not in st.session_state:
    st.session_state["retriever3"] = load_retriever(st.session_state["MD_140"])

if "retriever_list" not in st.session_state:
    st.session_state["retriever_list"] = [st.session_state["retriever1"], st.session_state["retriever2"], st.session_state["retriever3"]]

### Context List 생성
if "context_list1" not in st.session_state:
    st.session_state["context_list1"] = []

if "context_list2" not in st.session_state:
    st.session_state["context_list2"] = []

### Reranker 생성
if "reranker" not in st.session_state:
    st.session_state["reranker"] = load_reranker(device)

### LLM 생성
@st.cache_resource
def load_llm(mdl):
    return load_llm(mdl_nm=mdl)

if "llm" not in st.session_state:
    with st.spinner("Chat 모델을 불러오는 중입니다. 잠시만 기다려주세요..."):
        st.session_state["llm"] = load_llm(args.mdl_nm)

### Prompt 생성
if "prompt" not in st.session_state:
    st.session_state["prompt"] = load_prompt()

### Chain 생성
if "chain" not in st.session_state:
    st.session_state["chain"] = create_chain(st.session_state["prompt"], st.session_state["llm"])

### Generate Queries Prompt 생성
if "prompt_query_reformation" not in st.session_state:
    st.session_state["prompt_query_reformation"] = load_prompt_query_reformation()

### Generate Queries Chain 생성
if "chain_query_reformation" not in st.session_state:
    st.session_state["chain_query_reformation"] = create_chain_query_reformation(st.session_state["prompt_query_reformation"], st.session_state["llm"])

##################################################
### 1. 질의/응답
##################################################
print_history()

# 사용자 입력 처리
if user_input := st.chat_input("질문을 입력해주세요."):
    # Queue에 ID 추가
    id = add_queue()
    queue = load_queue()
    
    # Queue 상태 확인
    if queue.loc[queue['id'] == id, 'status'].iloc[0] != 'finish':
        with st.spinner("다른 사용자가 질문 중입니다. 잠시만 기다려주세요!"):
            for i in range(120):
                time.sleep(0.5)
                queue = load_queue()
                if queue.loc[queue['id'] == id, 'status'].iloc[0] == 'finish':
                    change_status(id, 'using')
                    ### 1. 사용자 질의 처리
                    question_time = now_time()
                    save_log("question_time", question_time)
                    save_log("WTG_model", WTG_model)

                    # 사용자 입력을 기록 및 표시
                    st.chat_message("user").write(user_input)
                    save_log("question", user_input)
                    add_history(role="user", content=user_input, message_type="text")

                    # WTG 모델 인덱스 가져오기
                    WTG_model_idx = WTG_model_list.index(WTG_model)
                    EW_db = st.session_state["EW_db_list"][WTG_model_idx]
                    retriever = st.session_state["retriever_list"][WTG_model_idx]

                    ### 2. 질의 분류 (Error/Warning 관련 여부)
                    ew_yn = 0
                    ew_info = None
                    search_result = EW_db.similarity_search_with_score(user_input, k=1)[0]

                    if WTG_model in ('01.WINDS3000_91_100', '03.WINDS5560_140') and search_result[1] < 0.3:
                        ew_yn = 1
                        ew_info = search_result[0].page_content
                        user_input = f"{ew_info} {user_input}"
                    elif WTG_model == '02.WINDS3000_134' and search_result[1] < 0.25:
                        ew_yn = 1
                        ew_info = search_result[0].page_content
                        user_input = f"{ew_info} {user_input}"

                    ### 3. 첫 번째 응답 (Naive RAG)
                    st.chat_message("ai").write("첫 번째 응답입니다.")
                    add_history(role="ai", content="첫 번째 응답입니다.", message_type="text")
                    answer1 = ""

                    with st.spinner("첫 번째 응답을 생성 중입니다. 잠시만 기다려주세요..."):
                        # Retriever
                        st.session_state["context_list1"] = retriever.invoke(user_input)

                        # 에러/경고 관련 질문일 경우 관련 정보 출력
                        if ew_yn == 1:
                            ew_content = make_ew_content(ew_info, WTG_model, EW_db)
                            answer1 += ew_content
                            st.write(ew_content)
                            add_history(role=None, content=ew_content, message_type="text")

                        # LLM 응답 출력
                        st.session_state["llm"].temperature = 0.01
                        st.session_state["llm"].max_new_tokens = 4096
                        st.session_state["llm"].top_k = -1
                        st.session_state["llm"].top_p = 0.95
                        response = st.session_state["chain"].invoke({"context": st.session_state["context_list1"], "question": user_input})
                        answer1 += response

                        # 텍스트와 테이블 분리
                        if "|" in response:
                            plain_text, table_text = split_text_and_table(response)

                            # 테이블을 DataFrame으로 변환
                            try:
                                df = markdown_to_dataframe(table_text).reset_index(drop=True)
                            except Exception:
                                df = markdown_to_dataframe(table_text, 2).reset_index(drop=True)

                            # Streamlit에서 텍스트 출력
                            st.write(plain_text)
                            add_history(role=None, content=plain_text, message_type="text")

                            # Streamlit에서 테이블 출력
                            if not df.empty:
                                st.table(df)
                                add_history(role=None, content=table_text, message_type="table")
                        else:
                            st.write(response)
                            add_history(role=None, content=response, message_type="text")

                        # 참조 문서 링크 출력
                        context_link = ""
                        for idx, context in enumerate(st.session_state["context_list1"], 1):
                            context_link += f"\n\n참조 문서 {idx}: {context.metadata['file_path']}/{context.metadata['file_name']}"
                            if "page_num" in context.metadata:
                                context_link += f" ({context.metadata['page_num']} page)"
                        answer1 += context_link
                        st.write(context_link)
                        add_history(role=None, content=context_link, message_type="text")
                        save_log("answer1", answer1)

                    ### 4. 두 번째 응답 (Advanced RAG: Query Reformation + Reranker)
                    st.chat_message("ai").write("두 번째 응답입니다.")
                    add_history(role="ai", content="두 번째 응답입니다.", message_type="text")
                    answer2 = ""

                    with st.spinner("두 번째 응답을 생성 중입니다. 잠시만 기다려주세요..."):
                        # Query Reformation & Reranker
                        st.session_state["llm"].temperature = 0.7
                        st.session_state["llm"].max_new_tokens = 512
                        st.session_state["llm"].top_k = 3
                        st.session_state["llm"].top_p = 0.5

                        # Retriever Chain 생성
                        retriever_chain = st.session_state["chain_query_reformation"].retriever.map().get_unique_union()
                        
                        # Reranker Retriever 생성
                        reranker_retriever = load_reranker_retriever(st.session_state["reranker"], retriever_chain)
                        
                        # 두 번째 Context 리스트 생성
                        st.session_state["context_list2"] = reranker_retriever.invoke(user_input)

                        # 에러/경고 관련 질문일 경우 관련 정보 출력
                        if ew_yn == 1:
                            ew_content = make_ew_content(ew_info, WTG_model, EW_db)
                            answer2 += ew_content
                            st.write(ew_content)
                            add_history(role=None, content=ew_content, message_type="text")

                        # LLM 응답 출력
                        st.session_state["llm"].temperature = 0.01
                        st.session_state["llm"].max_new_tokens = 4096
                        st.session_state["llm"].top_k = -1
                        st.session_state["llm"].top_p = 0.95
                        response = st.session_state["chain"].invoke({"context": st.session_state["context_list2"], "question": user_input})
                        answer2 += response

                        # 텍스트와 테이블 분리
                        if "|" in response:
                            # 텍스트와 테이블 분리
                            plain_text, table_text = split_text_and_table(response)
                            
                            # 테이블을 DataFrame으로 변환
                            try:
                                df = markdown_to_dataframe(table_text).reset_index(drop=True)
                            except Exception:
                                df = pd.DataFrame()  # 비어 있는 DataFrame 생성

                            # Streamlit에서 텍스트 출력
                            st.write(plain_text)
                            add_history(role=None, content=plain_text, message_type="text")

                            # Streamlit에서 테이블 출력
                            if not df.empty:
                                st.table(df)
                                add_history(role=None, content=table_text, message_type="table")
                            else:
                                st.write(response)
                                add_history(role=None, content=response, message_type="text")
                        else:
                            st.write(response)
                            add_history(role=None, content=response, message_type="text")

                        ### 참조 문서 링크 출력
                        context_link = ""
                        for idx, context in enumerate(st.session_state["context_list2"], 1):
                            context_link += f"\n\n참조 문서 {idx}: {context.metadata['file_path']}/{context.metadata['file_name']}"
                            if "page_num" in context.metadata:
                                context_link += f" (Page {context.metadata['page_num']} page)"
                        answer2 += context_link
                        st.write(context_link)
                        add_history(role=None, content=context_link, message_type="text")
                        save_log("answer2", answer2)

                    # 로그 저장
                    answer_time = now_time()
                    save_log("answer_time", answer_time)
                    change_status(id, 'finish')
                    st.session_state["process"] = 2
                    break
            change_status(id, 'finish')
    else:
        change_status(id, 'using')
        ### 1. 사용자 질의 처리
        question_time = now_time()
        save_log("question_time", question_time)
        save_log("WTG_model", WTG_model)

        # 사용자 입력을 기록 및 표시
        st.chat_message("user").write(user_input)
        save_log("question", user_input)
        add_history(role="user", content=user_input, message_type="text")

        # WTG 모델 인덱스 가져오기
        WTG_model_idx = WTG_model_list.index(WTG_model)
        EW_db = st.session_state["EW_db_list"][WTG_model_idx]
        retriever = st.session_state["retriever_list"][WTG_model_idx]

        ### 2. 질의 분류 (Error/Warning 관련 여부)
        ew_yn = 0
        ew_info = None
        search_result = EW_db.similarity_search_with_score(user_input, k=1)[0]

        if WTG_model in ('01.WINDS3000_91_100', '03.WINDS5560_140') and search_result[1] < 0.3:
            ew_yn = 1
            ew_info = search_result[0].page_content
            user_input = f"{ew_info} {user_input}"
        elif WTG_model == '02.WINDS3000_134' and search_result[1] < 0.25:
            ew_yn = 1
            ew_info = search_result[0].page_content
            user_input = f"{ew_info} {user_input}"

        ### 3. 첫 번째 응답 (Naive RAG)
        st.chat_message("ai").write("첫 번째 응답입니다.")
        add_history(role="ai", content="첫 번째 응답입니다.", message_type="text")
        answer1 = ""

        with st.spinner("첫 번째 응답을 생성 중입니다. 잠시만 기다려주세요..."):
            # Retriever
            st.session_state["context_list1"] = retriever.invoke(user_input)

            # 에러/경고 관련 질문일 경우 관련 정보 출력
            if ew_yn == 1:
                ew_content = make_ew_content(ew_info, WTG_model, EW_db)
                answer1 += ew_content
                st.write(ew_content)
                add_history(role=None, content=ew_content, message_type="text")

            # LLM 응답 출력
            st.session_state["llm"].temperature = 0.01
            st.session_state["llm"].max_new_tokens = 4096
            st.session_state["llm"].top_k = -1
            st.session_state["llm"].top_p = 0.95
            response = st.session_state["chain"].invoke({"context": st.session_state["context_list1"], "question": user_input})
            answer1 += response

            # 텍스트와 테이블 분리
            if "|" in response:
                plain_text, table_text = split_text_and_table(response)

                # 테이블을 DataFrame으로 변환
                try:
                    df = markdown_to_dataframe(table_text).reset_index(drop=True)
                except Exception:
                    df = markdown_to_dataframe(table_text, 2).reset_index(drop=True)

                # Streamlit에서 텍스트 출력
                st.write(plain_text)
                add_history(role=None, content=plain_text, message_type="text")

                # Streamlit에서 테이블 출력
                if not df.empty:
                    st.table(df)
                    add_history(role=None, content=table_text, message_type="table")
            else:
                st.write(response)
                add_history(role=None, content=response, message_type="text")

            # 참조 문서 링크 출력
            context_link = ""
            for idx, context in enumerate(st.session_state["context_list1"], 1):
                context_link += f"\n\n참조 문서 {idx}: {context.metadata['file_path']}/{context.metadata['file_name']}"
                if "page_num" in context.metadata:
                    context_link += f" ({context.metadata['page_num']} page)"
            answer1 += context_link
            st.write(context_link)
            add_history(role=None, content=context_link, message_type="text")
            save_log("answer1", answer1)

        ### 4. 두 번째 응답 (Advanced RAG: Query Reformation + Reranker)
        st.chat_message("ai").write("두 번째 응답입니다.")
        add_history(role="ai", content="두 번째 응답입니다.", message_type="text")
        answer2 = ""

        with st.spinner("두 번째 응답을 생성 중입니다. 잠시만 기다려주세요..."):
            # Query Reformation & Reranker
            st.session_state["llm"].temperature = 0.7
            st.session_state["llm"].max_new_tokens = 512
            st.session_state["llm"].top_k = 3
            st.session_state["llm"].top_p = 0.5

            # Retriever Chain 생성
            retriever_chain = st.session_state["chain_query_reformation"].retriever.map().get_unique_union()
            
            # Reranker Retriever 생성
            reranker_retriever = load_reranker_retriever(st.session_state["reranker"], retriever_chain)
            
            # 두 번째 Context 리스트 생성
            st.session_state["context_list2"] = reranker_retriever.invoke(user_input)

            # 에러/경고 관련 질문일 경우 관련 정보 출력
            if ew_yn == 1:
                ew_content = make_ew_content(ew_info, WTG_model, EW_db)
                answer2 += ew_content
                st.write(ew_content)
                add_history(role=None, content=ew_content, message_type="text")

            # LLM 응답 출력
            st.session_state["llm"].temperature = 0.01
            st.session_state["llm"].max_new_tokens = 4096
            st.session_state["llm"].top_k = -1
            st.session_state["llm"].top_p = 0.95
            response = st.session_state["chain"].invoke({"context": st.session_state["context_list2"], "question": user_input})
            answer2 += response

            # 텍스트와 테이블 분리
            if "|" in response:
                # 텍스트와 테이블 분리
                plain_text, table_text = split_text_and_table(response)
                
                # 테이블을 DataFrame으로 변환
                try:
                    df = markdown_to_dataframe(table_text).reset_index(drop=True)
                except Exception:
                    df = pd.DataFrame()  # 비어 있는 DataFrame 생성

                # Streamlit에서 텍스트 출력
                st.write(plain_text)
                add_history(role=None, content=plain_text, message_type="text")

                # Streamlit에서 테이블 출력
                if not df.empty:
                    st.table(df)
                    add_history(role=None, content=table_text, message_type="table")
                else:
                    st.write(response)
                    add_history(role=None, content=response, message_type="text")
            else:
                st.write(response)
                add_history(role=None, content=response, message_type="text")

            ### 참조 문서 링크 출력
            context_link = ""
            for idx, context in enumerate(st.session_state["context_list2"], 1):
                context_link += f"\n\n참조 문서 {idx}: {context.metadata['file_path']}/{context.metadata['file_name']}"
                if "page_num" in context.metadata:
                    context_link += f" (Page {context.metadata['page_num']} page)"
            answer2 += context_link
            st.write(context_link)
            add_history(role=None, content=context_link, message_type="text")
            save_log("answer2", answer2)

        # 로그 저장
        answer_time = now_time()
        save_log("answer_time", answer_time)
        change_status(id, 'finish')
        st.session_state["process"] = 2

##################################################
### 2. 응답 선택
##################################################
if st.session_state["process"] == 2:
    st.sidebar.subheader("1. 응답 선택")
    st.sidebar.write("두 응답 중 더 잘 응답했다고 생각하는 응답을 더블 클릭해주세요.")

    bt1, bt2, bt3 = st.sidebar.columns(3)

    if bt1.button("첫 번째 응답", use_container_width=True):
        save_log("select_answer", 1)
        st.session_state["options"] = [
            f'{context.metadata["file_path"]}/{context.metadata["file_name"]}'
            for context in st.session_state["context_list1"]
        ] + ["위 보기 문서에 없음"]
        st.session_state["context_list1"], st.session_state["context_list2"] = [], []
        st.session_state["process"] = 3

    if bt2.button("두 번째 응답", use_container_width=True):
        save_log("select_answer", 2)
        st.session_state["options"] = [
            f'{context.metadata["file_path"]}/{context.metadata["file_name"]}'
            for context in st.session_state["context_list2"]
        ] + ["위 보기 문서에 없음"]
        st.session_state["context_list1"], st.session_state["context_list2"] = [], []
        st.session_state["process"] = 3

    if bt3.button("둘 다 아님", use_container_width=True):
        save_log("select_answer", 3)
        st.session_state["options"] = ["위 보기 문서에 없음"]
        st.session_state["context_list1"], st.session_state["context_list2"] = [], []
        st.session_state["process"] = 3

##################################################
### 3. 사용자 Feedback
##################################################
elif st.session_state["process"] == 3:
    st.sidebar.subheader("2. 응답 피드백")
    st.sidebar.write("선택하신 응답에 대해 피드백을 남겨주세요.")

    with st.sidebar.form("응답에 대한 피드백"):
        ### 1. 점수 평가
        st.subheader("1) 점수 평가")
        st.write("0점: 생성된 응답이 없거나 아예 관련 없음")
        st.write("10점: 원하는 대답을 완벽하게 생성")
        score = st.slider("0~10점 중 점수를 매겨주세요.", 0, 10, 10)

        ### 2. 참조 문서 선택
        st.subheader("2) 참조 문서 선택")
        st.write("아래의 문서 중 응답의 내용이 원하는 응답 내용이 있는 문서를 선택해주세요 (다중 선택 가능)")

        ref_doc = []
        option_dup_list = []
        option_num_list = []

        # 참조 문서 옵션 생성
        for num, option in enumerate(st.session_state["options"]):
            if option in option_dup_list:
                continue
            option_dup_list.append(option)
            option_num_list.append(num)
            globals()[f"option_{num}"] = st.checkbox(option)

        # 선택된 문서를 ref_doc에 추가
        ref_doc = []
        for num, option in zip(option_num_list, option_dup_list):
            if globals()[f"option_{num}"]:
                ref_doc.append(option)

        # 참조 문서 직접 입력
        st.write("위 보기 문서에 없다면, 참조문서 목록에서 응답 내용이 있는 문서를 찾아서 입력해주세요.")
        file_input = st.text_input("참조문서 목록에서 찾은 내용을 입력해주세요.")
        if file_input:
            ref_doc.append(file_input)

        ### 3. 응답 입력
        st.subheader("3) 응답 입력")
        true_answer = st.text_input("응답이 올바르지 않다고 생각하신다면, 실제 원하는 응답을 작성해주세요.")

        ### 4. 질의/응답 및 피드백에 대한 log 정보 생성
        submitted = st.form_submit_button("피드백 제출")
        if submitted:
            st.success("피드백이 성공적으로 제출되었습니다.\n\n다음 질문을 입력해주세요.")
            
            # 로그 저장
            save_log("score", score)
            save_log("ref_doc", ref_doc)
            save_log("true_answer", true_answer)

            # Excel에 저장
            save_evaluation_to_excel(f'{data_path}/log.xlsx', st.session_state["log"])

            # session state 초기화
            st.session_state["log"] = {}
            st.session_state["process"] = 1