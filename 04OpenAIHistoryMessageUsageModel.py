# 1.获取client对象
from openai import OpenAI
import os

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    # api_key=os.getenv("DASHSCOPE_API_KEY"),
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",

    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

#2. 调用模型
messages = [
    {"role": "system", "content": "你是一个编程专家，回答很简洁"},

    {"role": "user", "content": "小明有两条宠物狗"},
{"role": "assistant", "content": "好滴"},

    {"role": "user", "content": "小明有三个乌龟"},
    {"role": "assistant", "content": "好滴"},
{"role": "user", "content": "总共有几个宠物呢"},
]
response = client.chat.completions.create(
    model="qwen3-max",  # 您可以按需更换为其它深度思考模型
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)

# 3. 处理结果
# print(response.choices[0].message.content)
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end=" ", flush=True)
