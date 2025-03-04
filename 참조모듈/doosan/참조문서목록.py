### Library Import
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
WTG_model_list = ["01.WINDS3000 91 100", "02.WINDS3000 134", "03.WINDS5560 140"]

# 사용자 선택을 위한 SelectBox
WTG_model = st.selectbox("풍력발전기 모델을 선택해주세요.", WTG_model_list)
st.write("참조문서를 클릭하고 빨간색 박스가 생기면, Ctrl+C를 통해 복사가 가능합니다.")

# 데이터 로드 및 처리
file_name = f"{data_path}/WTG_doc_clf.xlsx"
if os.path.exists(file_name):
    data = pd.read_excel(file_name, engine="openpyxl")
    # 선택된 모델에 따른 데이터 필터링
    data = data[data["model name"] == WTG_model]
    data["참조문서"] = data["file path"] + "/" + data["file name"]
    result = data[["참조문서"]]
    # 데이터프레임 출력
    st.dataframe(result, width=1600, hide_index=True)
else:
    st.error(f"데이터 파일을 찾을 수 없습니다: {file_name}")
