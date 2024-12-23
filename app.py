import gradio as gr
import os
from codes.backend_api import process_file


os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'

# 创建 Gradio 接口
iface = gr.Interface(
    fn=process_file,
    inputs=[
        gr.File(label="上传文档文件"),
        gr.Radio(['多选题', '填空题', '简答题'], label="选择题型")
    ],
    outputs=gr.Textbox(label="输出"),
    title="文档题型生成器"
)

# 启动应用
if __name__ == "__main__":
    iface.launch(share=True,server_name="127.0.0.1")
