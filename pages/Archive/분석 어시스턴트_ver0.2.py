import os
import streamlit as st
import pandas as pd
import math
import pyautogui

st.set_page_config(page_title="분석 어시스턴트", page_icon="🔍", layout='wide')

# ✅ 마트 활성화 및 관리 UI
data_marts = ["df_cust", "df_enroll", "df_product", "df_sales", "df_orders", "df_inventory", "df_customers", "df_regions", "df_transactions"]  # 등록된 마트 목록
if "selected_data_marts" not in st.session_state:
    st.session_state["selected_data_marts"] = []
if "temp_selected_marts" not in st.session_state:
    st.session_state["temp_selected_marts"] = set()
if "show_popover" not in st.session_state:
    st.session_state["show_popover"] = True


with st.popover("📊 마트 활성화하기",):
    st.markdown("<h3 style='text-align: center; color: #333;'>📊 활성화할 데이터 마트를 선택하세요</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    # ✅ 3열로 정렬하여 체크박스 출력
    cols_per_row = 3
    rows = math.ceil(len(data_marts) / cols_per_row)
    
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            mart_idx = row * cols_per_row + col_idx
            if mart_idx < len(data_marts):
                mart = data_marts[mart_idx]
                checked = mart in st.session_state["temp_selected_marts"]
                with cols[col_idx]:
                    if st.checkbox(mart, value=checked, key=f"chk_{mart}"):
                        st.session_state["temp_selected_marts"].add(mart)
                    else:
                        st.session_state["temp_selected_marts"].discard(mart)
    
    st.markdown("---")

    # ✅ 기존 마우스 위치 저장
    original_x, original_y = pyautogui.position()

    # ✅ 팝업 아래쪽 빈 공간 클릭 좌표 설정
    screen_width, screen_height = pyautogui.size()
    click_x = screen_width // 2  # 화면 중앙
    click_y = int(screen_height * 0.5)  # 화면 아래쪽의 중간 부분 클릭

    def close_popover():
        """Popover를 닫고, 바깥쪽을 클릭한 후 원래 마우스 위치로 이동"""
        print(f"{original_x, original_y} close_popover 호출")
        st.session_state["show_popover"] = False
        pyautogui.click(click_x, click_y)  # 바깥쪽 클릭
        pyautogui.moveTo(original_x, original_y, duration=0.0)  # 원래 위치로 이동

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 등록"):
            st.session_state["selected_data_marts"] = list(st.session_state["temp_selected_marts"])
            close_popover()

    with col2:
        if st.button("❌ 닫기"):
            close_popover()


# ✅ 현재 활성화된 마트 목록 표시
st.sidebar.subheader("📌 현재 활성화된 마트 목록")
if st.session_state["selected_data_marts"]:
    for mart in st.session_state["selected_data_marts"]:
        st.sidebar.markdown(f"<div style='padding: 10px; margin: 5px; border-radius: 8px; background-color: #f8f9fa; text-align: center; font-weight: bold;'>{mart}</div>", unsafe_allow_html=True)
else:
    st.sidebar.info("활성화된 마트가 없습니다.")

# ✅ 분석 어시스턴트 메인 화면
def chat_interface():
    st.subheader("🤖 분석 어시스턴트")
    st.write("질문을 입력해주세요.")
    user_input = st.text_input("질문을 입력하세요", "", key="user_query")
    if user_input:
        st.write(f"**사용자:** {user_input}")
        st.write("**AI 답변:** [분석 중...]")

chat_interface()
