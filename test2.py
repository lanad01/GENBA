import streamlit as st
import pandas as pd
import scipy.stats as stats

st.title("📊 AI 정규성 검정 어시스턴트")

# ✅ 데이터가 이미 존재한다고 가정
# 실제 운영 환경에서는 `df`를 외부에서 전달받거나 미리 로드해야 함
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame({
        "변수1": [12, 15, 14, 10, 13, 18, 21, 19, 17, 16],
        "변수2": [102, 99, 98, 105, 110, 95, 96, 103, 108, 107]
    })  # 예제 데이터

df = st.session_state.df  # 현재 사용 중인 데이터프레임
numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()  # 숫자형 변수만 선택

# ✅ 채팅 메시지 저장 (세션 상태)
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ 채팅 메시지 UI 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ✅ 사용자 입력 받기
user_input = st.chat_input("분석 관련 질문을 입력하세요.")

if user_input:
    # 사용자 메시지 저장
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # **STEP 1**: 정규성 검정 요청 확인
    with st.chat_message("assistant"):
        if not numeric_columns:
            st.write("숫자형 변수가 없습니다. 분석을 수행할 수 없습니다.")
        else:
            
            # **STEP 2**: 정규성 검정 방식 선택
            st.session_state.messages.append({"role": "assistant", "content": "어떤 정규성 검정 방식을 사용할까요?"})
            test_method = st.radio("정규성 검정 방식을 선택하세요.", ["Shapiro-Wilk", "Kolmogorov-Smirnov"], key="method_select")

            # **STEP 3**: 검정 수행
            if st.button("검정 수행"):
                data = df.dropna()

                if len(data) < 3:
                    st.write("데이터 개수가 너무 적어 검정할 수 없습니다.")
                else:
                    if test_method == "Shapiro-Wilk":
                        stat, p_value = stats.shapiro(data)
                    else:
                        stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))

                    # 결과 반환
                    result_msg = f"📋 **{test_method} 검정 결과**\n\n" \
                                    f"**검정 통계량**: {stat:.4f}\n" \
                                    f"**p-value**: {p_value:.4f}\n\n" \
                                    f"📌 **해석**: {'정규성을 만족합니다 ✅' if p_value > 0.05 else '정규성을 만족하지 않습니다 ⚠️'}"

                    st.session_state.messages.append({"role": "assistant", "content": result_msg})
                    st.write(result_msg)
