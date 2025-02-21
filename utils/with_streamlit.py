import streamlit as st
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import ChatMessage

########
# !!!! ChatMessage를 직접 받지 말고 스트링이랑 role을 받아서 여기서 ChatMessage로 변환
# ChatMessage도 가져오지 말고 직접 정의하자.

class message(TypedDict):
    """stremlit 챗메세지 기본 형식"""
    role: str
    content: str
    type: Annotated[Literal['text','df','chart'], ...]


def initialize_general_settings(title="AI Buddy"):
    """초기 세팅"""
    st.title(title)

    # 대화기록 저장소 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

def user_message():
    """사용자 메세지 출력
    사용 예시::
    user = user_message()
    user.write("안녕")
    user.write("나는 철수야")

    Output::
    "안녕", "나는 철수야" 가 한 프레임 안에 출력됨
    """
    return st.chat_message("user")

def ai_message():
    """AI 메세지 출력"""
    return st.chat_message("assistant")

def print_message_history():
    """화면에 이전 대화 출력"""
    for message in st.session_state["messages"]:
        print_message(message)

def add_message(message: ChatMessage):
    """저장소에 메세지 추가"""
    st.session_state["messages"].append(message)

def print_message(message, role=None):
    """화면에 메세지 출력"""
    if isinstance(message, ChatMessage):
        st.chat_message(message.role).write(message.content)
    else:
        st.chat_message(role).write(message)

def get_user_input():
    """사용자 입력 받기"""
    user_input = st.chat_input("메세지를 입력하세요..")
    if user_input:
        return ChatMessage(role="user", content=user_input)
    else:
        return None