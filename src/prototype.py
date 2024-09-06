import gradio as gr
from openai import OpenAI
import re,os,uuid
from dotenv import load_dotenv

# 目录名称
dir_name = "generated_htmls"

# 创建一个存放生成HTML文件的目录
os.makedirs(dir_name, exist_ok=True)

# 加载.env文件中的环境变量
load_dotenv()

# 从环境变量获取API密钥
api_key = os.getenv('DEEPSEEK_API_KEY')

# 设置 DeepSeek API 密钥和基础 URL
client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/beta",
    )

model_name = "deepseek-coder"

# client = OpenAI(
#         api_key="Replace with your own API key",
#         base_url="https://openrouter.ai/api/v1",
#     )

# model_name = "openai/gpt-4o-mini"


def extract_code(text):
    pattern = r"```html\n((?:.*\n)*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    code =  "\n\n".join(code_blocks)
    # 去掉code中的“```html”
    code = code.replace("```html\n", "")
    return code

def save_code_to_file(code, filename=None):
    if filename is None:
        # 生成一个唯一的文件名
        filename = str(uuid.uuid4()) + ".html"
        
    filepath = os.path.join("generated_htmls", filename)
    
    if len(code) == 0:
        return ""

    # 将HTML代码写入文件
    with open(filepath, "w") as f:
        f.write(code)
    
    return filepath

def merge_response(text_a, text_b):
  """
  合并两个文本字符串，并去除它们之间的重复部分。
  
  Args:
      text_a (str): 需要合并的第一个文本字符串。
      text_b (str): 需要合并的第二个文本字符串。
  
  Returns:
      str: 合并后的文本字符串，已去除text_a和text_b之间的重复部分。
  
  """
  
  # 去掉text_a结尾的空格和换行
  text_a = text_a.rstrip()
  # 去掉text_b开头的空格和换行
  text_b = text_b.lstrip()
  
  # 找到最长重复部分
  overlap = ''
  for i in range(len(text_a) - 1, 0, -1):
    if text_b.startswith(text_a[i:]):
      overlap = text_a[i:]
      break

  # 合并文本，去除重复部分
  merged_text = text_a + text_b[len(overlap):]

  return merged_text
        
sys_prompt = """
你是一名优秀的前端工程师，你会使用HTML + CSS + JavaScript来构建一个网页，你会使用外部的一些前端库，例如jQuery来进行一些展示上的优化。你将根据我的需求，为我写出一个HTML文件。请注意，只需要输出代码，不需要在代码之前回答任何“我可以”“当然可以”之类的回复。
在代码后，也不需要对代码进行任何解释。注意，需求中涉及到了表格，不要生成空白的表格，生成一些演示数据，以方便进行操作。示例中的文字尽量使用中文。
"""


def deepseek_chat(message, history):
    """
    使用DeepSeek API进行对话，并返回生成的回复文本。
    
    Args:
        message (str): 用户输入的消息文本。
        history (list[tuple[str, str]]): 历史对话记录，包含多个元组，每个元组包含用户消息和助手消息。
    
    Returns:
        str: DeepSeek API生成的回复文本。
    
    """
    
    # 将历史消息和当前消息一起传递给 DeepSeek API
    messages = [{"role": "system", "content": sys_prompt}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    
    response = ""
    
    while True:
        result = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=8192,
            temperature=0.7,
            stream=False
        )
        
        print(result.choices[0].finish_reason)
        
        message = result.choices[0].message.content
        
        # 如果是第二次及以后的内容，去掉message开头的```html\n
        if len(response) > 0:
            if message.startswith("```html\n"):
                message = message[len("```html\n"):]
        
        response = merge_response(response, message)
        
        if result.choices[0].finish_reason == "stop":
            # 模型认为生成已完成
            break
        else:
            # 响应不完整,将已生成的内容添加到prompt,继续生成
            messages.append({"role": "assistant", "content": message})
            messages.append({"role": "user", "content": "请继续"})
    
    
    response_text = response
    
    # 提取代码并保存到文件
    code = extract_code(response_text)
    filepath = save_code_to_file(code)
    
    if (len(filepath) > 0):
        file_url = f"file={filepath}"
    else:
        file_url = ""
    
    return response_text, file_url, code


def summarize(request_msg, last_code):
    summary_prompt = f"""
    请根据以下的对话历史和最后生成的HTML代码，对整个需求进行总结：

    对话历史：
    {request_msg}

    最后生成的HTML代码：
    {last_code}

    请提供一个简洁的总结，包括主要需求点和实现的功能。请注意，由于对话历史是由多轮对话组成的，其中有一些是修正生成代码的一些问题的，请注意进行识别提取最终的要求。
    """

    messages = [
        {"role": "system", "content": "你是一个优秀的需求分析师，善于总结和提炼关键信息。"},
        {"role": "user", "content": summary_prompt}
    ]

    result = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=8192,
        temperature=0.7,
        stream=False
    )

    summary = result.choices[0].message.content
    return summary

with gr.Blocks(fill_height=True) as demo_chatbot:
    htmlcode = "<H2>示例</H2>"
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            title = gr.Markdown("""
                                # AI生成原型工具
                                - 你可以完整的描述需要的内容和交互的动作，他将会生成一个HTML文件，注意打开文件时，会新建窗口或tab打开。
                                """)
            chatbot = gr.Chatbot()
            submit_button = gr.Button("提交")
            msg = gr.Textbox(label="需求")
        with gr.Column(scale=2, min_width=600):
            with gr.Tab("预览"):
                html = gr.HTML(htmlcode)
            with gr.Tab("源码"):
                source = gr.TextArea(label="Source Code",lines=26,max_lines=26,show_copy_button=True)
    with gr.Accordion("清除和配置在下面", open=False):
        request_msg = gr.Textbox(label="需求过程")
        summary_button = gr.Button("进行需求总结")
        summary_msg = gr.Textbox(label="需求总结")
        clear = gr.ClearButton([msg, chatbot, html, source])
    
    real_history = []
    
    def clear_real_history():
        real_history.clear()

    def respond(message, chat_history):
        response_text, file_url, code = deepseek_chat(message, real_history)
        
        if len(file_url) > 0:
            response_text_to_chatbot = gr.HTML(f"""
                            <a href='{file_url}' target="_blank">点击这里查看</a>
                            """)
        else:
            response_text_to_chatbot = response_text
        
        chat_history.append((message, response_text_to_chatbot))
        real_history.append((message, response_text))
        
        # 更新 request_msg
        updated_request_msg = "\n".join([f" {user}" for user, assistant in real_history])
        
        return "", chat_history, f'<iframe src="{file_url}" width="100%" height="600"></iframe>', code, updated_request_msg

    submit_button.click(respond, [msg, chatbot], [msg, chatbot, html, source, request_msg])
    clear.click(clear_real_history)
    
    def generate_summary(request_msg, last_code):
        summary = summarize(request_msg, last_code)
        return summary

    summary_button.click(generate_summary, inputs=[request_msg, source], outputs=summary_msg)

if __name__ == "__main__":
    demo_chatbot.queue(max_size=2).launch(allowed_paths=["generated_htmls"], share=False)
