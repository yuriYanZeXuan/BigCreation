import json

def check_dataset(file_path):
    """
    检查数据集是否符合预期格式，并输出报告。
    
    :param file_path: 数据集文件的路径
    """
    try:
        # 读取并解析 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # 检查数据集是否为列表格式
        if not isinstance(dataset, list):
            print("数据集格式错误：数据集应该是一个包含多个样本的列表。")
            return
        
        valid_samples = 0
        invalid_samples = 0

        # 遍历每个样本，检查字段是否符合要求
        for idx, sample in enumerate(dataset):
            # 确保每个样本有 "instruction", "input" 和 "output" 字段
            if not all(key in sample for key in ["instruction", "input", "output"]):
                print(f"样本 {idx} 格式错误：缺少必要字段 ('instruction', 'input', 'output')")
                invalid_samples += 1
                continue

            # 检查字段是否为空
            if any(not sample[key].strip() for key in ["instruction", "input", "output"]):
                print(f"样本 {idx} 存在空字段 ('instruction', 'input', 'output') 之一")
                invalid_samples += 1
                continue
            
            valid_samples += 1
        
        # 输出结果
        print(f"\n检查完成：")
        print(f"有效样本数量：{valid_samples}")
        print(f"无效样本数量：{invalid_samples}")
        if invalid_samples > 0:
            print(f"请检查以上无效样本并修复格式问题。")

    except json.JSONDecodeError:
        print("文件格式错误：无法解析 JSON 文件。")
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"发生错误：{e}")

# 调用检查函数，传入数据集文件路径
file_path = "data/squad_v2_qgen_filtered_with_answer.json"
check_dataset(file_path)
