python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/BigCreation/SFT_result/ds_lora_llama_8B \
  --tokenizer /root/autodl-tmp/BigCreation/SFT_result/ds_lora_llama_8B \
  --trust-remote-code \
  --served-model-name deepseek-chat \
  --port 8000
