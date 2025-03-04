# 기존 훈련 모델의 레포
BASIS='../model/huggingface/llm/Bllossom-llama-3.2-Korean-Bllossom-3B'

# 저장한 데이터의 레포
DATA='../data/log_1.xlsx'

# DPO가 진행된 후 어댑터가 저장되는 폴더의 명칭
output_directory='../model/finetuned/lora-llama3.2-output_adapter_dpo'

# 훈련이 끝난 어댑터와, 기존의 SFT를 merge한 이후 저장되는 폴더 이름
final_dir='../model/finetuned/lora-llama3.2-output_merged_dpo'

# 학습시킬 피드백 데이터 스코어 기준 (>= SCORE)
SCORE=5

# 훈련 실행
### 파라미터 설명
# num_epochs : 데이터셋 몇 번 학습할 지 설정(값이 클수록 overfitting 가능성 증가)
# lora_r : LoRA 행렬값 설정 (일반적으로 16 ~ 64, 값이 클수록 학습 가능한 파라미터 수 및 메모리 사용량 증가)
# lora_alpha : (일반적으로 16 or 32, 값이 클수록 학습 속도 증가, 그러나 학습 안정성 감소 가능성)
# lora_dropout : 과적합 방지 (일반적으로 0.01 ~ 0.1, 값이 클수록 과적합 방지 효과 증가, 학습 속도 감소)
# per_device_train_batch_size : GPU당 배치 크기 (일반적으로 8 ~ 64, 값이 클수록 학습 속도 증가, 그러나 메모리 사용량 급증)
# lr_scheduler_type : learning rate 스케줄러 지정 (cosine, linear, constant 중 선택 가능, 학습 속도와 성능에 영향을 미침)
# gradient_accumulation_steps : 배치의 그래디언트를 축적하는 단계 수 (일반적으로 2 ~ 8, 값이 클수록 메모리 사용량 감소, 학습 속도 감소)
# eval_step : 데이터 평가 빈도 (데이터 크기 및 학습 속도에 따라 조정 가능, 값이 클수록 디버깅이 어려울 수 있음)
# max_prompt_length : 모델이 처리할 수 있는 최대 입력(prompt) 길이
# max_length : 모델이 처리할 수 있는 최대 출력 시퀀스 길이

python finetuning_DPO.py \
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
	--eval_step 10 \
	--max_prompt_length 8192 \
	--max_length 8192 \
	--score_QA $SCORE

# 훈련 완료 후, 최종 adapter와 기존 모델을 merge
python merge.py \
	--base_model_name_or_path $BASIS \
	--peft_model_path $output_directory \
	--output_dir $final_dir
