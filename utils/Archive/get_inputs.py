import os
import json
import functools
import pandas as pd
from threading import Thread


input_path = "prompt"

class InputHandler_file_only:
    """input_path 내의 직접적인 파일만 읽어옴."""
    def __init__(self, path=None):
        self.input_path = path if path else input_path
        self.files = self._explore()

    def _explore(self):
        return [f for f in os.listdir(self.input_path) if os.path.isfile(os.path.join(self.input_path, f))]
    
    def read_file(self, filename):
        return open(os.path.join(self.input_path, filename), 'r', encoding='utf-8').read()

class InputHandler_subdir_only:
    """input_path 내의 서브디렉토리만 재귀 추출, 직접적인 파일은 읽어오지 않음."""
    def __init__(self, path=None):
        self.input_path = path if path else input_path
        self.files = self._explore()

    def _explore(self):
        result = {}
        # input_path의 서브디렉토리 목록 가져오기
        subdirectories = [f for f in os.listdir(self.input_path) if os.path.isdir(os.path.join(self.input_path, f))]
        
        for subdir in subdirectories:
            subdir_path = os.path.join(self.input_path, subdir)
            # 서브디렉토리 내의 파일 목록 가져오기 (파일명만 저장)
            files = [f for f in os.listdir(subdir_path) if (os.path.isfile(os.path.join(subdir_path, f))) and (not f.startswith('~$'))]
            # files = [f for f in os.listdir(subdir_path) if (os.path.isfile(os.path.join(subdir_path, f)))]
            result[subdir] = files
        
        return result

    def get_full_path(self, subdir, filename):
        """주어진 서브디렉토리와 파일명으로 전체 경로를 반환합니다."""
        return os.path.join(self.input_path, subdir, filename)

    def get_first_file_full_path(self):
        """첫 번째 서브디렉토리와 첫 번째 파일의 전체 경로를 반환합니다."""
        files_dict = self.files
        if files_dict:
            first_subdir = next(iter(files_dict))  # 첫 번째 서브디렉토리
            first_file = files_dict[first_subdir][0]  # 첫 번째 파일명
            return self.get_full_path(first_subdir, first_file)
        return None

    def read_file(self, subdir, filename):
        """주어진 서브디렉토리와 파일명으로 파일을 읽어 문자열로 반환합니다."""
        full_path = self.get_full_path(subdir, filename)
        if filename.endswith('.db'):
            return None  # .db 파일은 처리하지 않음
        elif filename.endswith('.json'):
            with open(full_path, 'r', encoding='utf-8') as file:
                return json.dumps(json.load(file), indent=4)  # JSON 파일을 예쁘게 출력
        elif filename.endswith('.xlsx'):
            sheet_name = 'test'
            columns_to_keep = ['no','쿼리타입','쿼리종류','주제영역','question','answer','추출테이블']
            df = pd.read_excel(full_path, sheet_name=sheet_name).loc[:, columns_to_keep]
            return df
        else:
            with open(full_path, 'r', encoding='utf-8') as file:
                return file.read()

    def read_files_from_subdir(self, subdir):
        """특정 서브디렉토리의 모든 파일을 읽어 각각 같은 이름의 변수에 저장합니다."""
        files_dict = self.files  # 서브디렉토리와 파일 목록을 가져옴
        files = files_dict.get(subdir, [])  # 서브디렉토리의 파일 목록 가져오기
        file_contents = {}
        
        for filename in files:
            content = self.read_file(subdir, filename)
            if content is not None:
                # 파일명을 변수명으로 사용
                var_name = filename
                file_contents[var_name] = content
        
        return file_contents

class FuncTimeoutException(Exception):
    pass

def timeout(timeout):
    """
    왜 그래프 안에서는 작동을 안 할까???
    """
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [FuncTimeoutException(f"[{func.__name__}] 실행 시간 초과 ({timeout}초)")]
            # 모든 timeout 데코레이터 사용한 메서드에 res 초기값을 error로 초기화
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                    #res[0] : api 데이터, 메서드를 실행시켜서 값을 저장
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco
    
    
def connect_to_postgres():
    """상태 안 좋음. PostgresDB 클래스 사용 권장.
    Postgres 데이터베이스에 연결하고 커서를 반환합니다."""
    from dotenv import load_dotenv
    from langchain_community.utilities import SQLDatabase

    load_dotenv()
    
    dbname = os.getenv('Postgres_dbname')
    user = os.getenv('Postgres_user')
    password = os.getenv('Postgres_password')
    host = os.getenv('Postgres_host')
    port = os.getenv('Postgres_port')

    return SQLDatabase.from_uri(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    )

if __name__ == "__main__":
    # input_path = r"C:\Users\mungc\Desktop\venv-Langs\aibuddy\inputs"
    # input_handler = InputHandler()
    # prompts = input_handler.read_files_from_subdir('Prompts')
    # schemas = input_handler.read_files_from_subdir('Schemas')
    # print(input_handler.files)
    # print(prompts)
    # print(schemas)

    from dotenv import load_dotenv
    load_dotenv()

    print(os.getenv("Postgres_user"))
    print(os.getenv("Postgres_port"))