import streamlit as st
import random

# 세션 상태에 버튼 클릭 여부 저장
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

# 버튼 클릭 이벤트 핸들러
def on_button_click():
    st.session_state.button_clicked = not st.session_state.button_clicked

# HTML & JS를 활용한 커스텀 버튼 구현
custom_button = """
    <style>
        .custom-button {
            background-color: #4CAF50;
            color: white;
            width : 200px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: 0.3s;
        }
        .custom-button:hover {
            background-color: #45a049;
        }
    </style>
    <script>
        function buttonClicked() {
            fetch('/button_click', {method: 'POST'}).then(response => response.json()).then(data => {
                console.log("Button Clicked:", data);
            });
        }
    </script>
    <button class="custom-button" onclick="buttonClicked()">커스텀 버튼</button>
"""

# HTML 코드 렌더링
st.markdown(custom_button, unsafe_allow_html=True)

# 버튼 클릭 후 상태 변경 확인
if st.session_state.button_clicked:
    st.success("버튼이 클릭되었습니다!")
else:
    st.info("버튼을 눌러보세요.")

st.title("Markdown bug")
st.caption('Output')

def tweet_button(tag: str, 
                 link: str, 
                text: str, 
                user: str):
    # 트위터 공유 URL 생성
    twitter_url = f"https://twitter.com/intent/tweet?text={text}&url={link}&hashtags={tag}&via={user}"
    
    # 랜덤 ID 생성 (중복 방지)
    random_id = random.randint(1000, 9999)
    
    # HTML과 JavaScript를 포함한 버튼 코드
    button_code = f"""
    <style>
        .twitter-button-{random_id} {{
            background-color: #ff9988;
            color: white;
            padding: 10px 15px;
            text-decoration: none;
            border-radius: 5px;
            display: inline-flex;
            align-items: center;
            font-weight: bold;
            margin: 10px 0;
            border: none;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }}
        .twitter-button-{random_id}:hover {{
            background-color: #ff8877;
        }}
    </style>
    
    <button class="twitter-button-{random_id}" onclick="window.open('{twitter_url}', '_blank')">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" style="margin-right: 5px;" viewBox="0 0 16 16">
            <path d="5z"/>
        </svg>
        트윗하기
    </button>
    """
    
    return button_code

st.write("")
tweet = tweet_button(tag='streamlit, share', 
             link='https://30days.streamlit.app/', 
             text='Streamlit share button', 
             user='streamlit')

# st.components.v1.html을 사용하여 JavaScript가 포함된 HTML 실행
st.components.v1.html(tweet, height=50)

st.write("")
st.write('📌 NOTE: 트위터 계정이 있어야 작동합니다.')
