### Library Import

### Built-in Modules
import os
import argparse
import pandas as pd
from datetime import datetime
import time
from pytz import timezone
import warnings
warnings.filterwarnings('ignore')

### Third-party Library
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from datasets import Dataset, load_dataset, DatasetDict
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, BitsAndBytesConfig
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from trl import DPOTrainer, DPOConfig
from huggingface_hub import login

########

### Function
######

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


def paired_data_preparation(
    data_dir: str = "",  # 데이터 파일 디렉토리
    sanity_check: bool = False,  # 샘플 데이터를 사용할지 여부
    cache_dir: str = None,  # 캐시 디렉토리
    split_criteria: str = "train",  # 데이터 분할 기준
    num_proc: int = 24  # 멀티프로세싱 사용 개수
) -> Dataset:
    """
    주어진 JSON 데이터를 읽어, 딕셔너리 형태의 데이터셋으로 변환하는 함수입니다.
    반환 데이터 형식:
    {
        "prompt": List[str],
        "chosen": List[str],
        "rejected": List[str]
    }

    prompt의 구조는 다음과 같습니다:
    "### 질문:\n<prompt>\n\n### 답변:"

    Args:
        data_dir (str): 데이터 파일 경로
        sanity_check (bool): 10개 데이터만 사용하여 검증할지 여부
        cache_dir (str): 캐시 파일 경로
        split_criteria (str): 데이터 분할 기준 (예: "train", "test")
        num_proc (int): 병렬 작업에 사용할 프로세스 수

    Returns:
        Dataset: 처리된 데이터셋 객체
    """
    args = get_args()


    # 입력 데이터 로드 (log.xlsx)
    tmp_df = pd.read_excel(data_dir)
    tmp_df = tmp_df[tmp_df['score'] >= args.score_QA].reset_index(drop=True)

    # 로그 데이터프레임 초기화
    log_df = pd.DataFrame()
    log_df['question'] = tmp_df['question']
    log_df["response_j"] = tmp_df.apply(
        lambda row: row['answer1'].split('＼n＼n 참조 문서 ')[0]
        if row["select_answer"] == 1 else row['answer2'].split('＼n＼n 참조 문서 ')[0],
        axis=1
    )
    log_df["response_k"] = tmp_df.apply(
        lambda row: row['answer2'].split('＼n＼n 참조 문서 ')[0]
        if row["select_answer"] == 1 else row['answer1'].split('＼n＼n 참조 문서 ')[0],
        axis=1
    )
    log_df = log_df.dropna().reset_index(drop=True)

    # 데이터셋으로 변환
    dataset = Dataset.from_pandas(log_df)
    original_columns = dataset.column_names

    # 샘플 데이터만 사용 (sanity_check가 True인 경우)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    # 데이터 포맷을 매핑하는 함수 정의
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": [
                "###질문:＼n" + question + "＼n＼n###답변:＼n"
                for question in samples["question"]
            ],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    # 데이터셋 매핑 및 반환
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def main():
    # 시작 시간 출력
    start_time = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    print()
    print(f"[LOG] [{start_time}] [DPO] Fine-tuning training START")

    # 실행 인자 가져오기
    args = get_args()

    # 학습 파라미터 출력
    print(
        "#### Training Arguments #########\n"
        f"model_name_or_path: {args.model_name_or_path}＼n"
        f"datapath: {args.datapath}＼n"
        f"output_dir: {args.output_dir}＼n"
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
        "#################################\n"
    )


    # SFT MODEL
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    # 캐시 비활성화
    model.config.use_cache = False

    # 병렬 처리 설정
    model.is_parallelizable = True
    model.model_parallel = True

    # 모델 최대 입력 임베딩 길이 설정
    model.config.max_position_embeddings = args.max_prompt_length
    print("model's max position embeddings:", model.config.max_position_embeddings)

    # Boolean 타입 버퍼 무시 설정 (분산 학습)
    if args.ignore_bias_buffers:
        model.ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # 토크나이저 로드 및 pad_token_id 설정
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 학습 데이터 준비
    train_dataset = paired_data_preparation(
        data_dir=args.datapath,
        split_criteria='train',
        sanity_check=args.sanity_check
    )


    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= args.max_length
    )

    # 검증 데이터 준비
    eval_dataset = paired_data_preparation(
        data_dir=args.datapath,
        split_criteria="validation",
        sanity_check=True
    )

    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= args.max_length
    )

    # 학습 파라미터 설정
    training_args = DPOConfig(
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps"
    )
