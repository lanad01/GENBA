# ✅ 기본 라이브러리
import os
import json
import pandas as pd
from glob import glob
from pathlib import Path
import streamlit as st

def get_page_state(page_name, key, default=None):
    """페이지별 세션 상태 값 가져오기"""
    full_key = f"{page_name}_{key}"
    return st.session_state.get(full_key, default)

def set_page_state(page_name, key, value):
    """페이지별 세션 상태 값 설정하기"""
    full_key = f"{page_name}_{key}"
    # print(f"🔢 [ set_page_state ] st.session_state['{full_key}']: {value}")
    st.session_state[full_key] = value

def get_available_marts():
    """data 디렉토리에서 사용 가능한 pkl 파일 목록을 가져옴"""
    data_dir = Path('../data')
    if not data_dir.exists():
        return []
    
    # pkl 파일 확장자만 확인
    mart_files = [
        f.stem for f in data_dir.iterdir() 
        if f.is_file() and f.name.endswith('.pkl')
    ]
            
    return sorted(mart_files)  # 정렬된 목록 반환

def load_selected_mart(mart_name):
    """선택된 마트를 실제로 로드"""
    try:
        data_path = Path(f'../data/{mart_name}.pkl')
        if data_path.exists():
            return pd.read_pickle(data_path)
        else:
            print(f"파일을 찾을 수 없음: {mart_name}")
            return None
    except Exception as e:
        print(f"파일 로드 오류 {mart_name}: {str(e)}")
        return None
    

