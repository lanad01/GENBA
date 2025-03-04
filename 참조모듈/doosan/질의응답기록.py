### Library Import
import os
import pandas as pd
import streamlit as st

### WEB
st.set_page_config(page_title="DOOSAN 풍력발전 O&M AI Chat", page_icon=":blue_circle:", layout="wide")
st.title(":blue[_DOOSAN_] 풍력발전 O&M AI Chat")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 200px;
        max-width: 200px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 데이터 경로 설정
data_path = "/home/pinetree/workplace/SPK14backup/workplace/doosan/data"
file_name = f"{data_path}/log.xlsx"

# 데이터 로드 및 표시
if os.path.exists(file_name):
    data = pd.read_excel(file_name, engine="openpyxl")
    st.dataframe(data, width=1600, hide_index=True)
else:
    st.error(f"데이터 파일을 찾을 수 없습니다: {file_name}")
