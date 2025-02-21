
########################################################################################################################
# 라이브러리 선언                                                                                   
########################################################################################################################
import pandas as pd
import sqlite3
import os
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

########################################################################################################################
# DB연결                                                                                            
########################################################################################################################
# DB Directory
db_dir = f"{os.getcwd()}/data/dev/source/pine.db"

# 일반연결
conn = sqlite3.connect(db_dir, isolation_level=None)
cur = conn.cursor()

# SQLAlchemy 연결
engine = create_engine(f"sqlite:///{db_dir}")
conn2 = engine.connect()

########################################################################################################################
# 작업로그 함수                                                                                     
########################################################################################################################
def func_log(
          subJob_nm   # 세부작업명
        , flag        # 작업구분 - S : 시작, E : 종료, U : 업로드 
    ):
    
    ##########################################################################################
    # 초기설정                                                                         
    ##########################################################################################
    # 객체 글로벌 선언
    global list_log
    
    # list_log가 존재하지 않을 경우 객체 생성
    try: 
        list_log
            
        # list_log가 df인 경우 list로 변환
        if isinstance(list_log, pd.core.frame.DataFrame):
            list_log = list_log.values.tolist()        
    except:
        list_log = []
        print(f"[LOG] {btch_nm} > {job_nm} > {subJob_nm} 시작, 로그 객체 미존재로 인해 객체 생성")
    
    ##########################################################################################
    # 작업구분 = 'S' 
    # - 세부작업명의 시작시간 생성     
    ##########################################################################################    
    if flag.upper() == 'S':   
        # 시작시간 생성
        st_tm = str(datetime.now()).split(' ')[1].split('.')[0]
        
        # 리스트 생성 및 적재
        tmp_log = [baseYm, execYmd, btch_nm, job_nm, subJob_nm, st_tm, None, None, 'N', None]                   
        list_log.append(tmp_log)
        
        print(f"[LOG] {btch_nm} > {job_nm} > {subJob_nm} 시작, 기준년월 = {baseYm}, 시작시간 = {st_tm}")
    
    ##########################################################################################
    # 작업구분 = 'E'     
    # - 세부작업명의 종료시간, 소요시간 생성    
    ##########################################################################################      
    elif flag.upper() == 'E':
        # df 변환
        df_log = pd.DataFrame(
              list_log 
            , columns = [
                    "기준년월"
                  , "수행일자"
                  , "배치명"
                  , "작업명"
                  , "세부작업명"
                  , "시작시간"
                  , "종료시간"
                  , "소요시간"
                  , "정상여부"
                  , "에러메세지"
              ]            
        )
        
        # df 정렬
        df_log = df_log.sort_values(by = ["시작시간"], ascending = False)
        
        # df 중복제거
        df_log = df_log.drop_duplicates(subset = ["배치명", "작업명", "세부작업명"], keep = 'last', ignore_index=True)
        
        # 종료시간, 소요시간 생성
        # 시작부분 존재 시
        try:
            st_tm = df_log[df_log["세부작업명"] == subJob_nm]["시작시간"].values[0]
        # 시작부분 미존재 시
        except:
            st_tm = str(datetime.now()).split(' ')[1].split('.')[0]
            print(f"[LOG] {btch_nm} > {job_nm} > {subJob_nm} 종료, 세부작업의 시작부분 미존재")
        
        ed_tm = str(datetime.now()).split(' ')[1].split('.')[0]
        el_tm = datetime.strptime(ed_tm, '%H:%M:%S') - datetime.strptime(st_tm, '%H:%M:%S')
        el_tm = str(el_tm).split('.')[0].zfill(8)
        
        # 배치구동이 자정을 넘어가서 "종료시간 < 소요시간" 인 경우 "-1 day," 이후만 추출 
        if ',' in el_tm:
            el_tm = el_tm.split(', ')[1].zfill(8)
        
        # 리스트 생성 및 기존데이터 삭제 후 적재
        if 'error' in globals():
            # 에러에서 Quotation 제거
            error_msg = str(error).replace("'", '').replace('"', '')
        
            tmp_log = [baseYm, execYmd, btch_nm, job_nm, subJob_nm, st_tm, ed_tm, el_tm, 'N', error_msg]  
            
            print(f"[LOG] {btch_nm} > {job_nm} > {subJob_nm} 에러, 기준년월 = {baseYm}, 시작시간 = {st_tm}, 종료시간 = {ed_tm}, 소요시간 = {el_tm}, 에러메세지 = {error}")
        else :
            tmp_log = [baseYm, execYmd, btch_nm, job_nm, subJob_nm, st_tm, ed_tm, el_tm, 'Y', None]   
             
        list_log = df_log[df_log["세부작업명"] != subJob_nm].values.tolist()
        list_log.append(tmp_log)
        
        print(f"[LOG] {btch_nm} > {job_nm} > {subJob_nm} 종료, 기준년월 = {baseYm}, 시작시간 = {st_tm}, 종료시간 = {ed_tm}, 소요시간 = {el_tm}")
        
    ##########################################################################################
    # 작업구분 = 'U'     
    # - 작업로그 업로드   
    ##########################################################################################         
    elif flag.upper() == 'U':
        # df 변환
        df_log = pd.DataFrame(
              list_log 
            , columns = [
                    "기준년월"
                  , "수행일자"
                  , "배치명"
                  , "작업명"
                  , "세부작업명"
                  , "시작시간"
                  , "종료시간"
                  , "소요시간"
                  , "정상여부"
                  , "에러메세지"
              ]            
        )
        
        # df 정렬
        df_log = df_log.sort_values(by = ["시작시간"], ascending = False)        
        
        # df 중복제거
        df_log = df_log.drop_duplicates(subset = ["배치명", "작업명", "세부작업명"], keep = 'last', ignore_index=True)

        # df 중 종료시간이 없는 행 삭제
        df_log = df_log[df_log["종료시간"].notnull()]
        
        # JobLog 테이블 미존재 시 테이블 생성
        cur.execute(
        """
            create table if not exists PINE_JOBLOG (
                  기준년월              text
                , 수행일자              text
                , 배치명                text
                , 작업명                text
                , 세부작업명            text
                , 시작시간              text
                , 종료시간              text
                , 소요시간              text
                , 정상여부              text
                , 에러메세지            text
                , PRIMARY KEY(기준년월, 수행일자, 배치명, 작업명, 세부작업명)
            )
        """)
        
        # 기존데이터 삭제
        cur.execute(
        f"""
            delete from PINE_JOBLOG
            where 
                    기준년월 = '{baseYm}'
                and 수행일자 = '{execYmd}'
                and 배치명   = '{btch_nm}'
                and 작업명   = '{job_nm}'
        """
        )
        
        # 작업로그 데이터 업로드
        
        df_log.to_sql(
              name      = f'PINE_JOBLOG'
            , con       = engine
            , if_exists = 'append'
            , index     = False
            , method    = "multi"
            , chunksize = 10000
        )        
        
        list_log = df_log
             
    return list_log
  
########################################################################################################################
# 함수 테스트                                                                                       
########################################################################################################################  
# try:
#     list_log = func_log(subJob_nm = '마트1', flag = 'S')
#     list_log = func_log(subJob_nm = '마트2', flag = 'S')
# 
#     #rc = pd.dataframe(1)
# 
#     list_log = func_log(subJob_nm = '마트2', flag = 'E')
#     list_log = func_log(subJob_nm = '마트3', flag = 'S')
#     list_log = func_log(subJob_nm = '마트3', flag = 'E')
#     list_log = func_log(subJob_nm = '종료' , flag = 'U')
# except Exception as error:
#     list_log = func_log(subJob_nm = '마트2', flag = 'E')
#     list_log = func_log(subJob_nm = '종료' , flag = 'U')