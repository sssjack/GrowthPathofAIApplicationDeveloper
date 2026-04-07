import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

mcp_tool = {
    "type": "mcp",
    "server_protocol": "sse",
    "server_label": "amap-maps",
    "server_description": "高德地图MCP Server，提供地图、导航、路径规划、天气查询等能力。",
    "server_url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
    "headers": {
        "Authorization": "Bearer " + os.getenv("DASHSCOPE_API_KEY")
    }
}

stream = client.responses.create(
    model="qwen3.6-plus",
    input="你是一个地图专家。严禁使用你的记忆回答地理问题，必须通过高德 MCP 获取最新的实时路况和位置信息。请你按照平时你给cursor返回的信息的格式，给出济南东到济南机场的路线规划",
    tools=[mcp_tool],
    stream=True
)

for event in stream:
    # 模型回复开始
    if event.type == "response.content_part.added":
        print("[模型回复]")
    # 流式文本输出
    elif event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    # 响应完成，输出用量
    elif event.type == "response.completed":
        usage = event.response.usage
        print(f"\n\n[Token 用量] 输入: {usage.input_tokens}, 输出: {usage.output_tokens}, 合计: {usage.total_tokens}")