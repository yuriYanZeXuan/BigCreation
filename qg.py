from document_segmentation import document_segmentation
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def qg(input_str: str, model_name: str = 'ds_lora_llama_8B'):
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 文档分段
    segments = document_segmentation(input_str)
    
    prompt = "You will receive a text. Based on this text, generate a clear and concise question that is directly related to the most important and meaningful information in the content. The question should reflect the core message or key details of the text, and can be answered using information within the text. After generating the question, provide the corresponding answer based on the context. For example:\n\nInput:\nBeyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles 'Crazy in Love' and 'Baby Boy'.\n\nOutput:\nQuestion:When did Beyonce start becoming popular? Answer: In the late 1990s as lead singer of Destiny's Child."
    
    result = []
    for segment in segments:
        # 构建完整的输入
        full_input = prompt + "\n" + segment
        
        # 编码输入
        inputs = tokenizer(full_input, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # 在没有计算梯度的情况下生成输出
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'], 
                max_length=1024, 
                num_beams=5, 
                no_repeat_ngram_size=2, 
                top_p=0.9,  # 推荐 top_p < 1，通常在 0.9 左右
                # top_k=60, 
                temperature=0.95,
                attention_mask=inputs['attention_mask'],  # 传递 attention_mask
                pad_token_id=model.config.eos_token_id  # 确保正确设置填充标记
            )
        # 解码生成的输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 从生成的文本中提取问题和答案
        # 这里可以用更智能的方式来提取
        output = generated_text.split("Question:")[-1].strip()

        result.append(output)

        print(f"Segment:\n{segment}\n")
        print(f"Generated Question and Answer:\n{output}\n")

    return result

if __name__ == "__main__":
    input_str = '''The Saint Alexander Nevsky Church was established in 1936 by Archbishop Vitaly (Maximenko) () on a tract of land donated by Yulia Martinovna Plavskaya. The initial chapel, dedicated to the memory of the great prince St. Alexander Nevsky (1220–1263), was blessed in May, 1936. The church building was subsequently expanded three times. In 1987, ground was cleared for the construction of the new church and on September 12, 1989, on the Feast Day of St. Alexander Nevsky, the cornerstone was laid and the relics of St. Herman of Alaska placed in the foundation. The imposing edifice, completed in 1997, is the work of Nikolaus Karsanov, architect and Protopresbyter Valery Lukianov, engineer. Funds were raised through donations. The Great blessing of the cathedral took place on October 18, 1997 with seven bishops, headed by Metropolitan Vitaly Ustinov, and 36 priests and deacons officiating, some 800 faithful attended the festivity. The old church was rededicated to Our Lady of Tikhvin. Metropolitan Hilarion (Kapral) announced, that cathedral will officially become the episcopal See of the Ruling Bishop of the Eastern American Diocese and the administrative center of the Diocese on September 12, 2014. At present the parish serves the spiritual needs of 300 members. The parochial school instructs over 90 boys and girls in religion, Russian language and history. The school meets every Saturday. The choir is directed by Andrew Burbelo. The sisterhood attends to the needs of the church and a church council acts in the administration of the community. The cathedral is decorated by frescoes in the Byzantine style. The iconography project was fulfilled by Father Andrew Erastov and his students from 1995 until 2001.'''
    
    qg(input_str)
