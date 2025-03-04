################
### Library Import
################

### Built-in Modules
import os
import argparse
import pandas as pd
from datetime import datetime
import time
from pytz import timezone
import warnings
warnings.filterwarnings('ignore')

### Third-party Libraries
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from datasets import Dataset, load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainingArguments
)
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from trl import DPOTrainer, DPOConfig
from huggingface_hub import login

################
### Function Definitions
################

################
### Arguments Parsing
################

def get_args():
    """
    sh 파일에서 입력받은 arguments parsing
    Args:
        num_epochs: 모델 학습을 반복할 총 에포크 수
        beta: DPO 로스 계산 시 사용하는 하이퍼파라미터 (trade-off factor)
        datapath: 데이터셋 파일의 경로
        model_name_or_path: 사용할 사전학습된 모델의 이름 또는 경로
        learning_rate: 옵티마이저의 학습률 (learning rate)
        lr_scheduler_type: 학습률 스케줄러의 유형 (예: linear, cosine)
        warmup_steps: 학습 초반에 학습률을 점진적으로 증가시키는 단계 수
        weight_decay: 옵티마이저의 가중치 감쇄 계수
        optimizer_type: 사용할 옵티마이저 유형 (예: paged_adamw_32bit)
        per_device_train_batch_size: 학습 시 GPU/CPU 장치당 배치 크기
        per_device_eval_batch_size: 평가 시 GPU/CPU 장치당 배치 크기
        gradient_accumulation_steps: 그래디언트를 누적할 스텝 수
        gradient_checkpointing: 그래디언트 체크포인팅 활성화 여부
        lora_alpha: LoRA의 스케일링 팩터 (fine-tuning에 사용)
        lora_dropout: LoRA 적용 시 드롭아웃 확률
        lora_r: LoRA rank (압축 비율)
        max_prompt_length: 입력 프롬프트의 최대 길이
        max_length: 전체 입력 시퀀스의 최대 길이
        max_steps: 총 학습 스텝 수
        logging_steps: 학습 로그를 출력할 스텝 간격
        save_steps: 체크포인트 저장 간격 (스텝 기준)
        eval_steps: 평가를 수행할 간격 (스텝 기준)
        output_dir: 모델 결과 및 체크포인트를 저장할 디렉토리
        log_freq: 학습 로그 출력 빈도
        sanity_check: 실행이 올바른지 확인하는 플래그
        report_to: 학습 로깅에 사용할 툴 (예: wandb, tensorboard)
        ignore_bias_buffers: LoRA 사용 시 편향값 (bias)의 버퍼를 무시할지 여부
        score_QA: 질의응답 점수 리스트
        lora_target_modules: LoRA 적용 대상 모듈 리스트
    Returns:
        parses_args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--datapath", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--optimizer_type", type=str, default="paged_adamw_32bit")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_step", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--sanity_check", type=bool, default=False)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--ignore_bias_buffers", type=bool, default=False)
    parser.add_argument("--lora_target_modules", type=list, default=[
        'embed_tokens', 'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'down_proj'
    ])
    parser.add_argument("--score_QA", type=int, default=8)

    return parser.parse_args()

def main():
    # 현재 시간(Asia/Seoul 시간대 기준)을 가져와 로그 시작
    start_time = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    print()
    print(f"[LOG] [{start_time}] [DPO] Fine-tuning training START")
    
    # 실행 인자 파싱
    args = get_args()
    
    # 학습 파라미터 로그 출력
    print(
        f"######## Training Arguments ########\n"
        f"model_name_or_path: {args.model_name_or_path}\n"
        f"datapath: {args.datapath}\n"
        f"output_dir: {args.output_dir}\n"
        f"per_device_train_batch_size: {args.per_device_train_batch_size}\n"
        f"per_device_eval_batch_size: {args.per_device_eval_batch_size}\n"
        f"per_device_train_batch_size: {args.per_device_train_batch_size}＼n"
        f"per_device_eval_batch_size: {args.per_device_eval_batch_size}＼n"
        f"num_epochs: {args.num_epochs}＼n"
        f"max_step: {args.max_step}＼n"
        f"learning_rate: {args.learning_rate}＼n"
        f"cutoff_len(max_length): {args.max_length}＼n"
        f"lora_r: {args.lora_r}＼n"
        f"lora_alpha: {args.lora_alpha}＼n"
        f"lora_dropout: {args.lora_dropout}＼n"
        f"lora_target_modules: {args.lora_target_modules}＼n"
        f"max_prompt_length: {args.max_prompt_length}＼n"
        f"#############################################＼n"
    )

    # Base model 로드
    print(f"Loading base model: {args.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **device_arg
    )

    # 토크나이저 로드 및 설정
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 만약 토크나이저가 확장되도록 변한다면 토큰 크기 조정 필요
    # base_model.resize_token_embeddings(len(tokenizer))

    # PEFT 모델 로드 및 병합
    print(f"Loading PEFT: {args.peft_model_path}")
    model = PeftModel.from_pretrained(base_model, args.peft_model_path, **device_arg)
    print(f"Running merge and unload")
    model = model.merge_and_unload()

    # 모델 및 토크나이저 저장
    model.save_pretrained(f"{args.output_dir}")
    tokenizer.save_pretrained(f"{args.output_dir}")
    print(f"Model saved to {args.output_dir}")

    # 완료 시간 기록
    finish_time = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    total_time = str(datetime.strptime(finish_time, '%Y-%m-%d %H:%M:%S') - datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'))
    print(f"[LOG] [{finish_time}] Fine-tuning Merge FINISH / Duration of Time: {total_time}")

if __name__ == "__main__":
    main()
