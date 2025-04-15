#!/bin/bash

deepspeed --hostfile hostfile.txt \
  --master_addr asc250 \
  --master_port 12345 \
  src/train.py \
  --deepspeed config/ds_config.json \
  --stage sft \
  --do_train \
  --model_name_or_path deepseek-ai/deepseek-llm-7b-base \
  --finetuning_type lora \
  --template default \
  --dataset question_generation_squad \
  --dataset_dir data \
  --output_dir ./output \
  --overwrite_cache \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --save_steps 1000 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --bf16 True
