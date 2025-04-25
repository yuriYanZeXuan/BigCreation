import json
import re

# 读取原始的 jsonl 文件
input_file = 'saves/DeepSeek-R1-8B-Distill/lora/eval_2025-04-19-01-36-12-DeepSeek-R1-Distill-Llama-8B/generated_predictions.jsonl'
output_file = 'output.jsonl'

# 正则表达式：匹配并提取 "Question:" 后面的内容
question_pattern = re.compile(r'Question:(.*?)(?=\s*Answer:|$)')

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 解析每一行 JSON
        data = json.loads(line)
        
        # 提取 label 和 predict 中的 question 部分
        label_question_match = re.search(question_pattern, data['label'])
        predict_question_match = re.search(question_pattern, data['predict'])

        if label_question_match:
            data['label'] = label_question_match.group(1).strip()  # 删除 "Question:"，只保留内容
        if predict_question_match:
            data['predict'] = predict_question_match.group(1).strip()  # 删除 "Question:"，只保留内容

        # 将处理后的数据写入新的 jsonl 文件
        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"处理完成，输出文件：{output_file}")
