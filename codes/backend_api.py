# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import json
def init_openai():  
    with open("config/apikey.json", "r") as f:
        api_key = json.load(f)["api_key"]

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return client

def convert_to_txt(file):
    """
    根据文件类型解析文件内容并返回文本
    支持 pdf、doc/docx、txt 格式
    """
    import os
    from pathlib import Path
    
    file_ext = Path(file.name).suffix.lower()
    
    if file_ext == '.txt':
        # 读取txt文件内容
        try:
            with open(file.name, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # 如果UTF-8解码失败,尝试其他编码
            try:
                with open(file.name, 'r', encoding='gbk') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file.name, 'r', encoding='gb2312') as f:
                    text = f.read()
        return text
        
    elif file_ext == '.pdf':
        # 使用pdfplumber解析PDF
        import pdfplumber
        
        with pdfplumber.open(file) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() + '\n'
        return text
        
    elif file_ext in ['.docx']:
        # 使用python-docx解析Word文档
        from docx import Document
        doc = Document(file)
        text = ''
        for para in doc.paragraphs:
            text += para.text + '\n'
        return text
    elif file_ext in ['.doc']:
        import win32com.client

        word = win32com.client.Dispatch("Word.Application")
        doc = word.Documents.Open(file)
        text = doc.Content.Text
        doc.Close()
        word.Quit()
        return text
    else:
        raise ValueError(f'不支持的文件格式: {file_ext}')

def test_api():
    client = init_openai()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )
    print(response.choices[0].message.content)

def process_file(file, question_type):
    print(file)
    print(question_type)
    # return f"已选择的题型: {question_type},文件名: {file.name}"
    text = convert_to_txt(file)
    client = init_openai()
    prompt=f"""
    请根据以下内容生成{question_type}类型的题目，题目要求：
    1. 题目数量：3道
    2. 题目难度：中等
    3. 题目类型：{question_type}
    4. 题目内容：{text}
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": prompt}],
        stream=False
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    test_api()
