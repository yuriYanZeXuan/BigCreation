import json

# 加载原始 SQuAD v2 JSON 文件
with open("dataset/dev-v2.0.json", "r", encoding="utf-8") as f:
    squad = json.load(f)

formatted_data = []

# 遍历结构化数据
for article in squad["data"]:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"].strip()

        # 检查 context 是否为空
        if not context:
            print(f"Warning: Empty context found, skipping this paragraph.")
            continue

        for qa in paragraph["qas"]:
            if qa.get("is_impossible", False):
                continue  # 过滤掉不可回答的问题

            question = qa["question"].strip()

            # 获取答案并检查是否为空
            answer = qa["answers"][0]["text"].strip() if qa["answers"] else ""
            if not answer:
                print(f"Warning: Empty answer found for question '{question}', skipping this QA pair.")
                continue

            # 将有效的样本加入 formatted_data
            output_str = f'"Question:{question} Answer: {answer}"'

            formatted_data.append({
                "instruction": "You will receive a text. Based on this text, generate a clear and concise question that is directly related to the most important and meaningful information in the content. The question should reflect the core message or key details of the text, and can be answered using information within the text. After generating the question, provide the corresponding answer based on the context. For example:\n\nInput:\nBeyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles 'Crazy in Love' and 'Baby Boy'.\n\nOutput:\nQuestion:When did Beyonce start becoming popular? Answer: In the late 1990s as lead singer of Destiny's Child.",
                "input": context,
                "output": output_str  # 使用字符串形式的 output
            })

# 过滤无效数据：确保每个条目都有有效的 context、question 和 answer
formatted_data = [example for example in formatted_data if example['input'] and example['output']]

# 保存为适用于 LLaMA-Factory WebUI 的格式
output_file = "data/squad_v2_qgen_dev_filtered_with_answer.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)

print(f"共生成 {len(formatted_data)} 条有效样本，保存在 {output_file} 中。")
