import os
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv()  # .env 파일 로드
import warnings
warnings.filterwarnings('ignore')

##################################################
### 초기화면
##################################################
st.set_page_config(page_title="PINE GenBA",page_icon="../img/pine.png", layout='wide')

st.markdown(
    """
    <style>
    .title {
        font-size: 40px !important;
        font-weight: bold;
        color: red;
        text-align: left;
    }
    </style>
    <h1 class="title"><span style="color:#ff;">PINE GENBA</span></h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .subtitle {
        font-size: 15px !important;
        font-weight: bold;
        color: red;
        text-align: left;
    }
    </style>
    <h1 class="subtitle">문의 : 데이터사업본부 - 권상우 선임
    """,
    unsafe_allow_html=True
)
st.write("")
st.write("")
st.markdown(f'''
## Menu
👨🏾‍💻 **데이터마트 생성** : 분석용 데이터마트를 생성(SQL) \n
🤖 **분석 어시스턴트** : 데이터분석을 지원하는 생성형 Assistant \n
''')
