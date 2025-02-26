import streamlit as st
import pandas as pd
import io
import contextlib
from streamlit_ace import st_ace  # Ace Editor 지원

# 기본 실행 코드 (초기값)
DEFAULT_CODE = """import pandas as pd

# 데이터 로드
df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))

# 데이터 미리보기
print(df.head())
print('Data shape:', df.shape)
print('Columns:', list(df.columns))
print('Preview done')
"""

# Streamlit 타이틀
st.title("CSV File Preview with Edit & Rerun")

# 파일 업로드
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        # 파일 내용 확인
        file_contents = uploaded_file.getvalue()
        if len(file_contents) == 0:
            st.error("❌ Uploaded file is empty. Please upload a valid CSV file.")
        else:
            # CSV 파일을 StringIO를 통해 pandas에서 읽기
            file_io = io.StringIO(file_contents.decode('utf-8'))
            df = pd.read_csv(file_io)

            # 초기 실행 (최초 업로드 시)
            if "output_logs" not in st.session_state:
                st.session_state.output_logs = ""
                st.session_state.editable = False  # 코드 편집 가능 여부
                st.session_state.user_code = DEFAULT_CODE  # 사용자 코드 저장

                # 실행 결과 캡처를 위한 스트림 생성
                output_buffer = io.StringIO()
                with contextlib.redirect_stdout(output_buffer):
                    local_vars = {
                        "uploaded_file": uploaded_file,
                        "df": df,
                        "pd": pd,
                        "io": io,
                    }
                    exec(DEFAULT_CODE, globals(), local_vars)

                # 실행된 결과 저장
                st.session_state.output_logs = output_buffer.getvalue()

            # 버튼 배치 (가로 정렬)
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Edit Code"):
                    st.session_state.editable = True  # 코드 편집 활성화
                    st.rerun()
            with col2:
                if st.button("Rerun"):
                    try:
                        # 실행 결과 캡처를 위한 스트림 생성
                        output_buffer = io.StringIO()
                        with contextlib.redirect_stdout(output_buffer):
                            local_vars = {
                                "uploaded_file": uploaded_file,
                                "df": df,
                                "pd": pd,
                                "io": io,
                            }
                            exec(st.session_state.user_code, globals(), local_vars)

                        # 실행된 결과 저장
                        st.session_state.output_logs = output_buffer.getvalue()

                    except Exception as e:
                        st.session_state.output_logs = f"❌ Error in User Code: {e}"

                    # 실행 결과 다시 표시
                    st.rerun()

            # 코드 편집기 (Ace Editor 사용)
            st.subheader("Edit Your Code")
            st.session_state.user_code = st_ace(
                value=st.session_state.user_code,
                language="python",
                theme="monokai",
                height=300,
                key="code_editor",
                readonly=not st.session_state.editable,  # "Edit Code" 누를 때만 활성화
            )

            # 실행 결과 출력
            st.subheader("Execution Output")
            st.text(st.session_state.output_logs)

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
