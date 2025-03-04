import streamlit as st
import time

# ✅ 채팅 메시지를 저장할 세션 상태 초기화
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

# ✅ 채팅 메시지 렌더링
chat_container = st.container()
for message in st.session_state["chat_messages"]:
    with chat_container:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# ✅ JavaScript를 사용한 자동 스크롤 기능 추가
scroll_js = """
    <script>
        var chatContainer = window.parent.document.querySelector("section[data-testid='stChatMessageContainer']");
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
"""
st.components.v1.html(scroll_js, height=0)  # ✅ JavaScript 실행

# ✅ 사용자 입력 필드 추가
query = st.chat_input("질문을 입력해주세요.")

if query:
    # ✅ 메시지 저장
    st.session_state["chat_messages"].append({"role": "user", "content": query})

    # ✅ 사용자 입력 메시지 표시
    with chat_container:
        with st.chat_message("user"):
            st.write(query)

    # ✅ 응답 생성 (예제)
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("🔍 답변을 생성 중..."):
                time.sleep(1)  # (실제 응답 대기 시간)
                response = f"'{query}'에 대한 답변입니다!"  # (실제 AI 응답으로 대체 가능)
                st.write(response)
    
    # ✅ 응답 메시지 저장
    st.session_state["chat_messages"].append({"role": "assistant", "content": response})

    # ✅ 채팅 입력 후 자동 스크롤 트리거
    st.components.v1.html(scroll_js, height=0)

    # ✅ UI 새로고침 (스크롤 반영)
    st.rerun()
