import boto3
import json
import pandas as pd
from io import StringIO
import config  # config.py에서 설정 가져오기
from datetime import datetime

class S3Handler:
    def __init__(self):
        """
        S3DataFrameHandler 초기화.
        
        Args:
            aws_access_key_id (str): AWS 액세스 키 ID.
            aws_secret_access_key (str): AWS 시크릿 액세스 키.
            region_name (str): AWS 리전 이름.
        """
                
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.ACCESS_KEY_ID,
            aws_secret_access_key=config.ACCESS_SECRET_KEY,
            region_name=config.REGION_NAME
        )

    def upload_to_bucket(self, df, bucket_name, bas_dt, ):
        """
        Pandas DataFrame을 S3에 업로드.
        
        Args:
            df (pd.DataFrame): 업로드할 Pandas DataFrame.
            bucket_name (str): S3 버킷 이름.
            key (str): S3에 저장할 객체 키 (경로 포함).
        """
        try:
            # S3 객체 키 생성
            object_key = f"s3/{bas_dt}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"S3 버킷 '{bucket_name}'의 '{object_key}'에 DataFrame을 업로드 시작합니다.")

            # JSON 데이터를 문자열로 변환
            json_data = df.to_json(orient='records', force_ascii=False)

            # S3에 JSON 업로드
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=json_data,
                ContentType='application/json'
            )
            print(f"JSON 데이터가 S3 버킷 '{bucket_name}'의 '{object_key}'에 성공적으로 업로드 되었습니다.")
            return object_key
        except Exception as e:
            print(f"DataFrame 업로드 실패: {e}")

    def download_from_bucket(self, bucket_name, key):
        """
        S3에서 Pandas DataFrame 다운로드.
        
        Args:
            bucket_name (str): S3 버킷 이름.
            key (str): S3 객체 키 (경로 포함).
        
        Returns:
            pd.DataFrame: 다운로드한 Pandas DataFrame.
        """
        try:
            print(f"S3 버킷 '{bucket_name}'의 '{key}'에서 DataFrame을 다운로드를 시작합니다.")
            response = self.s3_client.get_object(Bucket=bucket_name, Key=key, )
                        
            # Body 읽기
            body = response['Body'].read()  # 바이너리 데이터

            # 디코딩 및 JSON 파싱
            data = json.loads(body.decode('utf-8'))

            print(f"S3 버킷 '{bucket_name}'의 '{key}'에서 DataFrame을 성공적으로 다운로드 되었습니다.")
            print(f"---------------------------------------------------------------------------------------------")

            return data
        except Exception as e:
            print(f"DataFrame 다운로드 실패: {e}")
            return None
        
        
    def get_list_objects(self, bucket_name, prefixes=None, ):
        """
        S3 객체를 여러 조건(Prefix/Substrings)으로 필터링.

        Args:
            bucket_name (str): S3 버킷 이름.
            prefixes (list, optional): Prefix 조건 목록.

        Returns:
            list: 조건에 맞는 객체 키 목록.
        """
        response = self.s3_client.list_objects(Bucket=bucket_name)

        # 객체 목록 가져오기
        if 'Contents' not in response:
            return []

        objects = response['Contents']
        filtered_keys = []

        # 여러 조건으로 필터링
        for obj in objects:
            key = obj['Key']
            # Prefix 조건 확인
            if prefixes and not any(key.startswith(prefix) for prefix in prefixes):
                continue
            filtered_keys.append(key)

        return filtered_keys