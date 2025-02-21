from langchain_core.prompts import ChatPromptTemplate
import sys, os

home_dir = "C:/Users/권상우/GENBA-main/src"
sys.path.append(os.path.abspath(home_dir))

from utils.get_inputs import InputHandler_file_only

class PromptAgency:
    def __init__(self):
        self.input_handler = InputHandler_file_only()

    def get_data_extraction_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.input_handler.read_file('02_prompt_data_extraction.txt')),
            ("user", "### Messages:\n{messages}")
            ("user", "### recommended_schema:\n{recommended_schema}")
        ])

    def get_supervisor_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.input_handler.read_file('prompt_supervisor.txt')),
            ("user", "### Messages:\n{messages}")
        ])

    def get_general_query_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", "이 질문은 데이터 분석과 직접적으로 관련이 없습니다. 간단한 일반 답변을 제공하세요."),
            ("user", "### Messages:\n{messages}")
        ])

    def get_sql_builder_prompt(self, schema_name):
        return ChatPromptTemplate.from_messages([
            ("system", self.input_handler.read_file('prompt_sqlbuilder.txt').format(schema_name=schema_name)),
            ("user", "### Messages:\n{messages}"),
            ("user", "### Documents:\n{documents}")
        ])

    def get_sql_rebuilder_prompt(self):
        return ChatPromptTemplate.from_template("""
        Correct the following SQL query based on the error message provided.
        
        ### SQL Query:
        {sql_query}
        
        ### Error Message:
        {error_message}
        """)

    def get_insight_builder_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.input_handler.read_file('prompt_insightbuilder.txt')),
            ("user", (
                "### Question:\n{question}\n\n"
                "### Dataset:\n{dataframe}"
            ))
        ])

    def get_chart_decision_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.input_handler.read_file('prompt_chart_needed.txt')),
            ("user", (
                "### Question:\n{question}\n\n"
                "### Dataset:\n{dataframe}\n\n"
                "### Insights:\n{insights}"
            ))
        ])

    def get_error_fix_prompt(self):
        return ChatPromptTemplate.from_template(
            "다음 코드 실행 중 오류가 발생했습니다:\n{error_message}\n\n"
            "이 오류를 수정한 코드를 반환하세요:\n\n"
            "{original_code}\n\n"
            "수정된 코드를 ```python``` 블록으로 감사여 제공하세요."
        )

    def get_chart_builder_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.input_handler.read_file('prompt_chartbuilder.txt')),
            ("user", (
                "### Question:\n{question}\n\n"
                "### Dataset:\n{dataframe}\n\n"
                "### Insights:\n{insights}"
            ))
        ])

    def get_report_builder_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.input_handler.read_file('prompt_reportbuilder.txt')),
            ("user", (
                "### User Question:\n{question}\n\n"
                "### Dataset Provided:\n{dataframe}\n\n"
                "### Insights Derived:\n{insights}\n\n"
                "### Chart Filename:\n{chart_filename}"
            ))
        ])

    def get_planner_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.input_handler.read_file('prompt_planner.txt')),
        ])

    def get_general_query_handler_prompt(self):
        """일반 질의 처리를 위한 프롬프트 반환"""
        return ChatPromptTemplate.from_messages([
            ("system", """당신은 데이터 분석 전문가입니다. 
            사용자의 질문에 친절하게 답변해주세요.
            데이터 분석과 관련이 없는 질문이라도 최선을 다해 답변해주세요."""),
            ("human", "{messages}")
        ])
