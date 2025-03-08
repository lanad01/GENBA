"""Postgre 연결 및 스키마 조회 메서드 모음"""
import psycopg
import os
import pandas as pd
from dotenv import load_dotenv

#####################################
# SQLite로 docker postgre 연결 시
# f"postgresql+psycopg://pinetree:pinetree123@localhost:1989/postgres?options=-c%20search_path={schema_name}"
#####################################

__all__ = [
    'PostgresDB',
    'get_table_list', 
    'get_table_columns',
    'get_table_schema',
    'get_primary_key',
    'get_foreign_key'
]

def get_table_list(db, schema='biz') -> list[str]:
    """Postgre 데이터베이스에서 주어진 스키마에 속한 모든 테이블 이름을 반환하는 함수
    ex)
       ['transaction_history',
        'assets',
        'deposit',
        'product_history',
        'loan',
        'product',
        'consult_history',
        'customer',
        'customer_income']"""
    query_get_table_list = "SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}' AND table_type = 'BASE TABLE';"
    ret = db.run(query_get_table_list.format(schema=schema))
    tables = []
    for table in ret:
        tables.append(table[0])
    return tables

def get_table_columns(db, schema='biz') -> dict:
    """Postgre 데이터베이스에서 주어진 스키마에 속한 모든 테이블의 컬럼명을 반환하는 함수
    ex)
       {'transaction_history': ['고객id', '결제id', '결제일', '결제수단', '이용처', '이용금액', '기준년월'],
        'assets': ['고객id', '자산id', '자산구분', '자산규모', '자산상태', '등록일자', '갱신일자', '기준년월'],
        'deposit': ['거래id','고객id','상품id','계좌번호','거래일자','거래구분','거래금액','잔액','계좌상태','기준년월'],
        ...}"""
    query_get_table_columns = "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = '{schema}' ORDER BY table_name, ordinal_position;"
    ret = db.run(query_get_table_columns.format(schema=schema))
    tables = get_table_list(db, schema)
    columns = {}
    for table in tables:
        columns[table] = []
    for col in ret:
        columns[col[0]].append(col[1])
    return columns

def get_table_schema(db, schema='biz'):
    """Postgre 데이터베이스에서 주어진 스키마에 속한 모든 테이블의 테이블스키마를 반환하는 함수
    ex)
       [('schema', 'table', 'column', 'type', 'nullable'),
        ('biz', 'assets', '고객id', 'character varying', 'NO'),
        ('biz', 'assets', '자산id', 'character varying', 'NO'),
        ('biz', 'assets', '자산구분', 'character varying', 'YES'),
        ('biz', 'assets', '자산규모', 'integer', 'YES'),
        ('biz', 'assets', '자산상태', 'character varying', 'YES'),
        ('biz', 'assets', '등록일자', 'character varying', 'YES'),
        ('biz', 'assets', '갱신일자', 'character varying', 'YES'),
        ('biz', 'assets', '기준년월', 'character varying', 'YES'),
        ('biz', 'consult_history', '고객id', 'character varying', 'NO')
        ...]"""
    query_get_table_schema = "SELECT table_schema AS schema_name, table_name, column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema NOT IN ('pg_catalog', 'information_schema') ORDER BY table_schema, table_name, ordinal_position;"
    tableschema = db.run(query_get_table_schema.format(schema=schema))
    tableschema.insert(0, ('schema','table','column','type','nullable'))
    # tableschema = pd.DataFrame(columns=['schema','table','column','type','nullable'])
    # for col in raw_tableschema:
    #     tableschema.loc[len(tableschema)] = [col[0], col[1], col[2], col[3], col[4]]
    return tableschema

def get_primary_key(db, schema='biz'):
    """Postgre 데이터베이스에서 주어진 스키마에 속한 테이블의 기본키를 반환하는 함수
    ex)
       [['schema', 'table', 'pk'],
        ('biz', 'customer_income', '고객id'),
        ('biz', 'customer_income', '소득id'),
        ('biz', 'product', '상품id'),
        ('biz', 'assets', '고객id'),
        ('biz', 'assets', '자산id'),
        ('biz', 'transaction_history', '고객id'),
        ...]"""
    query_get_primary_key = """
    SELECT 
        tc.table_schema AS schema_name,
        tc.table_name,
        kcu.column_name
    FROM 
        information_schema.table_constraints AS tc
    JOIN 
        information_schema.key_column_usage AS kcu
    ON 
        tc.constraint_name = kcu.constraint_name
    WHERE 
        tc.constraint_type = 'PRIMARY KEY'
    AND 
        tc.table_schema NOT IN ('pg_catalog', 'information_schema');
    """
    primary_keys = db.run(query_get_primary_key.format(schema=schema))
    primary_keys.insert(0, ['schema','table','column'])
    # primary_keys = pd.DataFrame(columns=['schema','table','column'])
    # for col in raw_tableschema:
    #     primary_keys.loc[len(primary_keys)] = [col[0], col[1], col[2]]
    return primary_keys

def get_foreign_key(db, schema='biz'):
    """Postgre 데이터베이스에서 주어진 스키마에 속한 테이블의 외래키를 반환하는 함수
    ex)
       [['schema','table','column','foreign_table_schema','foreign_table_name','foreign_column_name'],
        ('biz', 'customer_income', '고객id', 'biz', 'customer', '고객id'),
        ('biz', 'assets', '고객id', 'biz', 'customer', '고객id'),
        ('biz', 'transaction_history', '고객id', 'biz', 'customer', '고객id'),
        ('biz', 'consult_history', '고객id', 'biz', 'customer', '고객id'),
        ('biz', 'product_history', '고객id', 'biz', 'customer', '고객id'),
        ...]"""
    query_get_foreign_key = """
    SELECT 
        tc.table_schema AS schema_name,
        tc.table_name AS table_name,
        kcu.column_name AS column_name,
        ccu.table_schema AS foreign_table_schema,
        ccu.table_name AS foreign_table_name,
        ccu.column_name AS foreign_column_name
    FROM 
        information_schema.table_constraints AS tc
    JOIN 
        information_schema.key_column_usage AS kcu
    ON 
        tc.constraint_name = kcu.constraint_name
    JOIN 
        information_schema.constraint_column_usage AS ccu
    ON 
        ccu.constraint_name = tc.constraint_name
    WHERE 
        tc.constraint_type = 'FOREIGN KEY';
    """
    foreign_key = db.run(query_get_foreign_key.format(schema=schema))
    foreign_key.insert(0, ['foreign_schema','foreign_table','foreign_column','schema','table','column'])
    # foreign_keys = pd.DataFrame(columns=['schema','table','column','foreign_table_schema','foreign_table_name','foreign_column_name'])
    # for col in raw_foreign_key:
    #     foreign_keys.loc[len(foreign_keys)] = [col[0], col[1], col[2], col[3], col[4], col[5]]
    return foreign_key

class PostgresDB:
    def __init__(self, timeout=20, docker=False):
        """timeout은 초단위로 받고, self.connect_to_postgres, self.connect_to_postgres_docker 내부에서 *1000을 해준다."""
        load_dotenv()
        if docker:
            self.connection = self.connect_to_postgres_docker(timeout=timeout)
        else:
            self.connection = self.connect_to_postgres(timeout=timeout)
        self.cursor = self.connection.cursor()

    def _run(self, query):
        try:
            self.cursor.execute(query)
        except Exception as e:
            self.connection.rollback()
            raise e
        
    
    def run(self, query):
        """fetchall 메서드를 통해 쿼리 결과를 반환하는 메서드"""
        try:
            self._run(query)
            return True, self.cursor.fetchall()
        except Exception as e:
            self.connection.rollback()
            return False, str(e)
        
    def is_syntax_correct(self, query: str):
        """쿼리 문법 검사. 맞으면 True와 빈 스트링, 틀리면 False와 오류 메세지를 반환한다."""
        query = "EXPLAIN (COSTS FALSE) " + query
        try:
            self.run(query)
            return True, ''
        except Exception as e:
            self.connection.rollback()
            return False, str(e)
        
    def rollback(self):
        self.connection.rollback()
    
    def limited_run(self, query, num=1000):
        """fetchmany 메서드를 통해 쿼리 결과를 반환하는 메서드"""
        try:
            self._run(query)
            return True, self.cursor.fetchmany(num)
        except Exception as e:
            self.connection.rollback()
            return False, str(e)

    def connect_to_postgres(self, timeout):
        """timeout은 초단위로 받습니다. 2초는 그냥 2로 넣으면 됩니다."""
        timeout *= 1000
        try:           
            return psycopg.connect(
                dbname = os.getenv('Postgres_dbname'),
                user = os.getenv('Postgres_user'),
                password = os.getenv('Postgres_password'),
                host = os.getenv('Postgres_host'),
                port = os.getenv('Postgres_port'),
                options = f'-c statement_timeout={timeout}'
            )
        except (Exception, psycopg.DatabaseError) as error:
            raise f"Error: {error}"
        
    def connect_to_postgres_docker(self, timeout):
        """timeout은 초단위로 받습니다. 2초는 그냥 2로 넣으면 됩니다."""
        timeout *= 1000
        try:
            return psycopg.connect(
                dbname = os.getenv('Postgres_dbname_docker'),
                user = os.getenv('Postgres_user_docker'),
                password = os.getenv('Postgres_password_docker'),
                host = os.getenv('Postgres_host_docker'),
                port = os.getenv('Postgres_port_docker'),
                options = f'-c statement_timeout={timeout}'
            )
        except (Exception, psycopg.DatabaseError) as error:
            raise f"Error: {error}"
        
    def __del__(self):
        if self.connection:
            self.connection.close()
        if self.cursor:
            self.cursor.close()