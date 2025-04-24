import json

# 加载原始 SQuAD v2 JSON 文件
with open("dataset/train-v2.0.json", "r", encoding="utf-8") as f:
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
                "instruction": "You will receive a text. Based on this text, generate a single question that is directly related to the most important and meaningful information in the content. The question should be clear, concise, and reflect the core message or key details of the text. Avoid questions that are too broad or trivial, and ensure that the question can be answered using information within the provided text. For example:\
Input:\
Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles 'Crazy in Love' and 'Baby Boy'.\
Output:\
When did Beyonce start becoming popular?",
                "input": context,
                "output": question
            })

# 保存为适用于 LLaMA-Factory WebUI 的格式
with open("data/squad_v2_qgen_filtered.json", "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)

print(f"共生成 {len(formatted_data)} 条样本，保存在 data/squad_v2_qgen_filtered.json 中。")
