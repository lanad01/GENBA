# 기존 훈련 모델의 경로
BASIS='../model/huggingface/llm/Bllossom-llama-3.2-Korean-Bllossom-3B'

# 학습에 사용할 데이터 경로
DATA='./data/log_1.xlsx'

# DPO 진행 후 어댑터가 저장되는 폴더
output_directory='../model/finetuned/lora-llama3.2-output_adapter_instruction'

# 어댑터와 기존 SFT를 merge한 후 저장될 폴더
final_directory='../model/finetuned/lora-llama3.2-output_merged_instruction'

# 학습에 사용될 데이터 점수 기준 (SCORE 이상만 사용)
SCORE=5

# 훈련 실행
# 파라미터 설명
# - num_epochs: 학습 에포크 수 (값이 클수록 overfitting 가능성 증가)
# - lora_r: LoRA 행렬 크기 (16~64 범위 권장)
# - lora_alpha: 학습 속도를 결정하는 LoRA scaling factor (16 또는 32 추천)
# - lora_dropout: 과적합 방지를 위한 dropout 비율 (0.01~0.1 범위 권장)
# - per_device_train_batch_size: GPU당 학습 배치 크기
# - lr_scheduler_type: Learning rate 스케줄러 (cosine, linear, constant 중 선택)
# - gradient_accumulation_steps: 그래디언트 축적 단계 (메모리 사용량 절감)
# - eval_steps: 평가 주기 (값이 클수록 검증 빈도 감소)
# - max_prompt_length: 모델 최대 입력 길이
# - max_length: 모델 최대 출력 길이

python finetuning_Instruction.py \
	--model_name_or_path $BASIS \
	--output_dir $output_directory \
	--datapath $DATA \
	--num_epochs 1 \
	--lora_r 32 \
	--lora_alpha 16 \
	--lora_dropout 0.01 \
	--per_device_train_batch_size 64 \
	--lr_scheduler_type cosine \
	--gradient_accumulation_steps 4 \
	--eval_steps 10 \
	--max_prompt_length 8192 \
	--max_length 8192 \
	--score_QA $SCORE

# 훈련 완료 후, 훈련된 최종 adapter와 기존 모델 merge
python merge.py \
	--base_model_name_or_path $BASIS \
	--peft_model_path $output_directory \
	--output_dir $final_directory
