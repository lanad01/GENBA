import pandas as pd
import os
from dotenv import load_dotenv
import warnings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# ✅ 환경 변수 로드 및 설정
pd.options.display.float_format = '{:.3f}'.format
warnings.filterwarnings('ignore')
load_dotenv()

# ✅ OpenAI API Key 확인
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)

# ✅ 결과 저장을 위한 객체 선언
results = {}
list_df = {}

# ✅ 데이터 로드: ../data 경로의 모든 pkl 파일 읽기
data_path = os.path.join('..', 'data')
for file in os.listdir(data_path):
    if file.endswith('.pkl'):
        file_path = os.path.join(data_path, file)
        df_name = file.replace('.pkl', '')
        list_df[df_name] = pd.read_pickle(file_path)

list_df_text = ", ".join(list_df.keys())

# ✅ 프롬프트 및 함수 코드 불러오기
prompt = open(f'prompt/001_prompt_data_summary.txt', 'r', encoding='utf-8').read().format(list_df_text=list_df_text)
func_code = open(f'sample_func/func_data_summary.py', 'r', encoding='utf-8').read()

# ✅ LLM에 첫 번째 코드 요청 (데이터 구조 분석 단계)
chain = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("user", "### 참고 코드:\n{func_code}\n\n"),
    ("user", "### list_df:\n{list_df_text}\n\n")
]) | llm

response = chain.invoke({"func_code": func_code, "list_df_text": list_df_text}).content

attempt_count = 0  # 실행 시도 횟수
success = False  # 코드 실행 성공 여부

while attempt_count < 2:  # 최초 실행 + 1회 재시도 가능
    try:
        if "```python" in response:
            modified_code = response.split("```python")[-1].split("```")[0]
            print(f"[LOG] 실행할 코드:\n{modified_code}")
            exec(modified_code, globals())  # LLM이 생성한 코드 실행
            print(f"[Stage 1. 데이터 구조 파악] 1차 시도 | ✅ 성공적으로 실행되었습니다!")
            success = True
            break  # 실행 성공 시 루프 종료
    except Exception as e:
        error_message = str(e)
        print(f"❌ {attempt_count+1}차 시도 : LLM 생성 코드 실행 중 오류 발생: {error_message}")

        if attempt_count == 0:  # 최초 실행 실패 시 1회만 재생성
            print("🔄 오류 메시지를 기반으로 코드 수정 요청 중...")

            # 오류 메시지를 기반으로 LLM에 코드 수정 요청
            prompt_error_fix = f"""
            ### 코드 수정 요청

            이전 코드 실행 중 다음 오류가 발생했습니다:
            ```
            {error_message}
            ```

            위 오류를 해결한 새로운 코드를 생성하세요.
            - 기존 코드에서 오류를 수정한 버전으로 제공해야 합니다.
            - 코드 실행 가능하도록 보완해야 합니다.

            ```python
            # 필요한 코드 삽입
            ```
            """
            chain_error_fix = ChatPromptTemplate.from_messages([
                ("system", prompt_error_fix),
                ("user", "### 기존 코드:\n{modified_code}\n\n")
            ]) | llm

            response = chain_error_fix.invoke({"modified_code": modified_code}).content
        else:
            print("❌ 코드 실행 최종적으로 실패. 프로세스를 중단합니다.")

        attempt_count += 1  # 재시도 횟수 증가

# ✅ Stage 1 결과를 메모리 내 변수로 저장
stage1_results = analyze_multiple_dataframes(list_df, save_format="memory")
results["stage1"] = stage1_results  # ✅ Stage 1 결과 저장

# ✅ RAG 기반 컬럼 설명 추가
def load_vectorstore():
    if os.path.exists("./vectordb"):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        try:
            return FAISS.load_local("./vectordb", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ 벡터스토어 로드 중 오류 발생: {e}")
            return None
    return None

vectorstore = load_vectorstore()

def search_column_descriptions(col_desc_df, vectorstore):
    """RAG를 사용하여 컬럼 설명을 검색하는 함수"""
    data = []
    df_after_llm = col_desc_df[['데이터프레임명', '컬럼명']]
    
    for i, row in df_after_llm.iterrows():
        table_name = row['데이터프레임명']
        col = row['컬럼명']
        
        search_query = f"테이블명 : {table_name} | 컬럼명 : {col}"
        docs = vectorstore.similarity_search(search_query, k=1)
        
        if docs:
            best_match = docs[0].page_content
            data.append({
                '데이터프레임명': table_name,
                '컬럼명': col,
                '컬럼설명': best_match.split('\n')[3].split('설명')[1].strip()
            })
        else:
            data.append({
                '데이터프레임명': table_name,
                '컬럼명': col,
                '컬럼설명': "설명 없음"
            })
    
    return pd.DataFrame(data)

# ✅ Stage 2: RAG 기반 컬럼 설명 추가
stage1_df = stage1_results["데이터 개요"]
rag_results = search_column_descriptions(stage1_df, vectorstore)

# ✅ Stage 1 결과와 RAG 결과를 통합
final_df = pd.merge(
    stage1_df,     # Stage 1 결과 (데이터 구조 정보)
    rag_results,   # RAG 검색 결과 (컬럼 설명)
    on=['데이터프레임명', '컬럼명'],
    how='left'
)

# ✅ 최종 결과를 메모리 내 저장
results["stage2"] = final_df

print("✅ 모든 단계가 성공적으로 완료되었습니다.")
