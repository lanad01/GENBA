import streamlit as st

def apply_custom_styles():
    """UI 스타일 적용"""
    st.markdown(
        """
        <style>
            /* 사이드바 기본 너비 설정 */
            [data-testid="stSidebar"] {
                min-width: 330px !important;
                max-width: 800px !important;
            }
            
            /* 사이드바 리사이즈 핸들 스타일 */
            [data-testid="stSidebar"] > div:first-child {
                width: auto !important;
                resize: horizontal !important;
                overflow-x: auto !important;
            }
            
            /* 네비게이션 컨테이너 스타일 수정 */
            div[data-testid="stSidebarNav"] {
                height: auto !important;
            }
            
            /* 메뉴 영역 스타일 수정 */
            section[data-testid="stSidebarNav"] {
                top: 0 !important;
                padding-left: 1.5rem !important;
                height: auto !important;
                min-height: 300px !important;
            }
            
            /* 메뉴 아이템 컨테이너 */
            section[data-testid="stSidebarNav"] > div {
                height: auto !important;
                padding: 1rem 0 !important;
            }
            
            /* 스크롤바 숨기기 */
            section[data-testid="stSidebarNav"]::-webkit-scrollbar {
                display: none !important;
            }
            
            .stChatMessage { max-width: 90% !important; }
            .stMarkdown { font-size: 16px; }
            .reference-doc { font-size: 12px !important; }
            table { font-size: 12px !important; }
            
            /* 데이터프레임 스타일 수정 */
            .dataframe {
                font-size: 12px !important;
                white-space: nowrap !important;  /* 텍스트 줄바꿈 방지 */
                text-align: left !important;
            }
            
            /* 데이터프레임 셀 스타일 */
            .dataframe td, .dataframe th {
                min-width: 100px !important;  /* 최소 너비 설정 */
                max-width: 200px !important;  /* 최대 너비 설정 */
                padding: 8px !important;
                text-overflow: ellipsis !important;
            }
            
            /* 데이터프레임 헤더 스타일 */
            .dataframe thead th {
                text-align: left !important;
                font-weight: bold !important;
                background-color: #f0f2f6 !important;
            }
            [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                background-color: white;
                margin: 10px 0;
            }
        </style>
        """,
        unsafe_allow_html=True
    )