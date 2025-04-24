#!/bin/bash

cd ~/LLaMA-Factory
python document_segmentation.py

llamafactory-cli train \
    --stage sft \
    --model_name_or_path ~/LLaMA-Factory/ds_lora_llama_8B \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bnb \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset temp \
    --cutoff_len 1024 \
    --max_samples 1000000 \
    --per_device_eval_batch_size 32 \
    --predict_with_generate True \
    --max_new_tokens 512 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir temp \
    --trust_remote_code True \
    --do_predict True