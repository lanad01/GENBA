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

    # 1. 모델/토크나이저 로딩
    # 사전 학습 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    # 캐시 비활성화
    model.config.use_cache = False
    # 병렬 처리 가능 여부
    model.is_parallelizable = True
    # 병렬 처리 활성화
    model.model_parallel = True
    # 모델 최대 입력 임베딩 길이 설정
    model.config.max_position_embeddings = args.max_prompt_length


    print(f"Model's max position embeddings: {model.config.max_position_embeddings}")

    # 분산 학습을 위해 boolean 타입 버퍼 무시하도록 설정
    if args.ignore_bias_buffers:
        model.ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # 토크나이저 로드 및 pad_token_id 설정
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 입력 데이터 로드(log.xlsx)
    log_df = pd.read_excel(args.datapath)

    # 특정 점수 이상만 필터링
    log_df = log_df[log_df['score'] >= args.score_QA].reset_index(drop=True)

    # 각 데이터 행 전처리 함수 정의
    def preprocess_row(row):
        """
        각 행에서 question, answer1, answer2, 참조 문서(ref_doc),
        WTG model의 답변 점수를 포함한 프롬프트 생성
        """

        # WTG model의 답변 포함한 질문 텍스트 추출
        instruction = row["question"]

        # 정답 추출
        correct_answer = row[f"answer{row['select_answer']}"].split("＼n＼n 참조 문서 1")[0]

        # 사용자 정답
        true_answer = row["true_answer"]

        return {
            "instruction": instruction,
            "input": true_answer,
            "output": correct_answer
        }

    # 데이터프레임에 전처리 적용
    processed_data = log_df.apply(preprocess_row, axis=1)
    processed_df = pd.DataFrame(list(processed_data))

    #
    # 2. DataFrame Dataset 변환 및 train/eval 분할 로직
    #
    num_rows = len(processed_df)

    # Hugging Face 데이터셋 형식으로 변환
    dataset = Dataset.from_pandas(processed_df)

    if num_rows < 10:
        # 10건 미만이면 eval 없이 진행
        dataset = DatasetDict({"train": dataset})
        train_dataset = dataset["train"]
        eval_dataset = None
    else:
        # 10건 이상이면 90% train, 10% eval
        train_size = int(num_rows * 0.9)
        dataset = DatasetDict({"all": dataset})
        train_dataset = dataset["all"].select(range(train_size))
        eval_dataset = dataset["all"].select(range(train_size, num_rows))


    #
    # 3. 데이터 전처리 함수 정의
    #
    def format_instruction(example):
        """
        학습 및 평가에 사용할 프롬프트 생성
        """
        # input 값을 안전하게 처리하기 위해 get으로 가져온 뒤, 문자열이 아니면 빈 문자열로 변환
        input_value = example.get("input", "")
        if not isinstance(input_value, str):
            input_value = ""
        
        if input_value.strip():
            prompt = (
                f"Below is an instruction and an input.\n"
                f"Follow the instruction to produce a helpful answer.\n"
                f"### Instruction:\n{example['instruction']}\n"
                f"### Input:\n{input_value}\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"Below is an instruction.\n"
                f"Follow the instruction to produce a helpful answer.\n"
                f"### Instruction:\n{example['instruction']}\n"
                f"### Response:\n"
            )
        return prompt, example["output"]


    # 학습, 평가에 사용할 프롬프트 생성
    def preprocess_function(examples):
        prompts = []
        targets = []
        for i in range(len(examples["instruction"])):
            prompt, output = format_instruction({
                "instruction": examples["instruction"][i],
                "input": examples["input"][i],
                "output": examples["output"][i]
            })
            full_text = prompt + output + tokenizer.eos_token
            tokenized = tokenizer(full_text, truncation=True, max_length=512)
            prompts.append(tokenized["input_ids"])
            targets.append(tokenized["input_ids"])
        return {
            "input_ids": prompts,
            "labels": targets
        }

    # 학습 데이터 전처리
    processed_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    # 평가 데이터 전처리 (있을 경우)
    processed_eval = None
    if eval_dataset is not None:
        processed_eval = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )


    # 4. 데이터 콜레이터 준비 (데이터 배치 생성)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,  # True일 경우 최대 길이를 배치 내에서 자동으로 맞춰줌
        return_tensors="pt"  # PyTorch 텐서 반환
    )

    # 5. 학습 아규먼트 설정
    training_args = TrainingArguments(
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps" if processed_eval is not None else "no",
        eval_steps=args.eval_steps if processed_eval is not None else None,
        output_dir=args.output_dir,
        save_total_limit=1,  # 최신 체크포인트 하나만 유지
        fp16=True,  # 혼합 정밀도 학습 활성화
        bf16=True,  # BF16 지원 활성화
        logging_dir=f"{args.output_dir}/logs",  # 로그 저장 경로
        report_to=args.report_to,  # 로깅 툴 설정
        push_to_hub=False,  # 허브에 모델 푸시 비활성화
        remove_unused_columns=False,  # 불필요한 열 제거 비활성화
        optim=args.optimizer_type  # 옵티마이저 타입
    )

    # LoRA 설정
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # PEFT 모델 생성
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    print("##### MODEL was Loaded in GPU #####")

    # 6. Trainer 생성 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_eval,
        data_collator=data_collator
    )

    print("#### Training Process is preparing now #######################")

    trainer.train()
    trainer.save_model(args.output_dir)

    # 최종 체크포인트 저장
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

    # 학습 종료 로그
    finish_time = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    total_time = str(datetime.strptime(finish_time, '%Y-%m-%d %H:%M:%S') - datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'))
    print(f"[LOG] [{finish_time}] [DPO] Fine-tuning training FINISH / Duration of Time: {total_time}")

if __name__ == "__main__":
    main()
