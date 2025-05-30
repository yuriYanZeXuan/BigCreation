import gradio as gr
import os
from codes.backend_api import process_file


os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'


# 启动应用
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_run', action='store_true', help='是否使用本地模型推理')
    args = parser.parse_args()
    local_run = args.local_run
    # 创建 Gradio 接口
    iface = gr.Interface(
        fn=lambda file, question_type: process_file(file, question_type, local_run=local_run),
        inputs=[
            gr.File(label="上传文档文件"),
            gr.Radio(['多选题', '填空题', '简答题'], label="选择题型"),
        ],
        outputs=gr.Markdown(label="输出"),
        title="文档题型生成器"
    )
    iface.launch(share=True,server_name="127.0.0.1")
