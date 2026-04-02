from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_tool_call, wrap_model_call, before_agent, after_agent, before_model, \
    after_model
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
from langgraph.runtime import Runtime

# 定义中间件函数
@tool(description="查询天气，传入城市名称字符串，返回字符串天气信息")
def get_weather(city: str) -> str:
    return f"{city}天气：晴天"

"""
1. agent执行前
2. agent执行后
3. model执行前
4. model执行后
5. 工具执行中
6. 模型执行中
"""

# 1. agent执行前
@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    # agent执行前会调用这个函数并传入state和runtime两个对象
    print(f"[before agent]agent启动，并附带{len(state['messages'])}消息")

# 2. agent执行后
@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    print(f"[after agent]agent结束，并附带{len(state['messages'])}消息")

# 3. model执行前
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    print(f"[before_model]模型即将调用，并附带{len(state['messages'])}消息")

# 4. model执行后
@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> None:
    print(f"[after_model]模型调用结束，并附带{len(state['messages'])}消息")

# 5. 模型执行中（包装器）
@wrap_model_call
def model_call_hook(request, handler):
    print("模型调用啦")
    return handler(request)

# 6. 工具执行中（包装器）
@wrap_tool_call
def monitor_tool(request, handler):
    print(f"工具执行：{request.tool_call['name']}")
    print(f"工具执行传入参数：{request.tool_call['args']}")
    return handler(request)

# 构建Agent（需将中间件传入）
agent = create_agent(
    model=ChatTongyi(model="qwen3-max"),
    tools=[get_weather],
    system_prompt="你是一个智能助手，可以回答用户问题！",
    # 若框架支持中间件参数，需添加如下配置
    middleware=[log_before_agent, log_after_agent, log_before_model, log_after_model, model_call_hook, monitor_tool]
)



# 测试Agent调用
if __name__ == "__main__":
    result = agent.invoke(input={"messages": [{"role": "user", "content": "深圳今天天气如何？如何穿衣"}]})
    print(result["messages"][-1].content)