import streamlit as st

# JavaScript를 사용한 스크롤 다운 기능
def scroll():
    js = """
    <script>
        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    </script>
    """
    st.components.v1.html(js, height=0)

st.title("Streamlit Scroll Test")

# 첫 번째 컨테이너
with st.container():
    st.header("첫 번째 컨테이너")
    for i in range(20):
        st.write(f"1st Area : {i}")

# 두 번째 컨테이너
with st.container():
    st.header("두 번째 컨테이너")
    for j in range(20):
        st.write(f"2nd Area : {j}")

# 스크롤 버튼 (사이드바)
st.sidebar.button('Scroll to bottom of main', on_click=scroll)
