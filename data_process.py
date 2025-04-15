import json

# 加载原始 SQuAD v2 JSON 文件
with open("dataset/dev-v2.0.json", "r", encoding="utf-8") as f:
    squad = json.load(f)

formatted_data = []

# 遍历结构化数据
for article in squad["data"]:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"].strip()
        for qa in paragraph["qas"]:
            if qa.get("is_impossible", False):
                continue  # 过滤掉不可回答的问题
            question = qa["question"].strip()
            formatted_data.append({
                "instruction": "Please generate a relevant question based on the following paragraph.",
                "input": context,
                "output": question
            })

# 保存为适用于 LLaMA-Factory WebUI 的格式
with open("data/squad_v2_qgen_dev_filtered.json", "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)

print(f"共生成 {len(formatted_data)} 条样本，保存在 data/squad_v2_qgen_dev_filtered.json 中。")
