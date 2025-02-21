import os
from datetime import datetime

# 글로벌 딕셔너리로 작업 시작 시간을 관리


class LogManager:
    def __init__(self, dir_root, job_nm):
        """
        LogManager 초기화.

        Args:
            dir_root (str): 로그 파일의 루트 디렉토리 경로.
        """
        
        self.dir_log = os.path.join(dir_root, "log")
        self.job_nm = job_nm
        self.job_start_times = {}

        # 디렉토리가 없으면 생성
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)
            
    def format_elapsed_time(self, elapsed_seconds):
        """
        초 단위를 hh:mm:ss 형식으로 변환합니다.
        
        Args:
            elapsed_seconds (int): 소요 시간(초)
        
        Returns:
            str: hh:mm:ss 형식의 문자열
        """
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"


    def list_log(self, job_name, flag = None):
        """
        특정 작업의 로그를 기록하며 시작, 끝 플래그와 소요 시간을 기록합니다.
        
        Args:
            program_name (str): 로그 대상 프로그램 이름
            job_name (str): 작업 이름
            flag (str): 'S'(시작) 또는 'E'(끝)
        
        Returns:
            None
        """
        global job_start_times

        # 로그 파일 디렉토리 및 파일 경로 설정
        log_file_path = os.path.join(self.dir_log, f"{self.job_nm}.log")

        # 디렉토리가 없으면 생성
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)

        # 현재 시간 가져오기
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y.%m.%d %H:%M:%S")

        # 로그 메시지 초기화
        formatted_log = ""

        # 시작 플래그 처리
        if flag == 'S':
            # 시작 시간 저장
            self.job_start_times[job_name] = current_time
            formatted_log = f"[LOG] job : {self.job_nm} | 작업 : {job_name} | 작업 시작 | {timestamp}\n"

        # 끝 플래그 처리
        elif flag == 'E':
            # 시작 시간이 존재하는지 확인
            if job_name in self.job_start_times:
                start_time = self.job_start_times.pop(job_name)
                elapsed_time = (current_time - start_time).total_seconds()

                # 소요 시간을 hh:mm:ss 형식으로 변환
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                elapsed_hms = f"{hours:02}:{minutes:02}:{seconds:02}"
                
                elapsed_hms = self.format_elapsed_time(int(elapsed_time))
                
                formatted_log = f"[LOG] job : {self.job_nm} | 작업 : {job_name} | 작업 종료 | {timestamp} | 소요시간 : {elapsed_hms}\n"
            else:
                # 시작 시간 정보가 없는 경우
                formatted_log = f"[LOG] job : {self.job_nm} | 작업 : {job_name} | 작업 종료 | {timestamp} | 시작 시간 없음\n"
        else :
            formatted_log = f"[LOG] job : {self.job_nm} | 작업 : {job_name} | {timestamp}\n"

        # 로그 파일에 텍스트 추가
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(formatted_log)

        print(f" {formatted_log.strip()}")
