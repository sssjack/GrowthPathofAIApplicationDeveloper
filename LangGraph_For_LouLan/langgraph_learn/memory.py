from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver # 内存检查点，用于存放短期记忆

# 1. 定义工具 (Tools)
def get_weather(city: str) -> str:
    """获取某个城市的天气"""
    # 模拟返回，实际开发中可对接天气 API
    return f"{city}的天气一直都是晴天！"

tools = [get_weather]

# 2. 初始化模型和内存存储
model = ChatTongyi(model="qwen3-max")
memory = MemorySaver() # 这是实现记忆的核心组件

# 3. 创建 Agent
# 对应图中红圈 1：传入模型、工具和检查点
agent_executor = create_agent(
    model,
    tools=tools,
    checkpointer=memory
)

# --- 第一轮对话 ---
# 对应图中红圈2：设置 thread_id
config = {"configurable": {"thread_id": "user_session_001"}}

print("--- 第一轮：提问长沙天气 ---")
first_input = {"messages": [("user", "长沙天气怎么样？")]}
for event in agent_executor.stream(first_input, config):
    for value in event.values():
        print(value["messages"][-1].content)



# --- 第二轮对话（体现记忆） ---
print("\n--- 第二轮：追问北京（不提“天气”二字） ---")
# 注意：我们依然使用同一个 config (相同的 thread_id)
second_input = {"messages": [("user", "北京呢？")]}
for event in agent_executor.stream(second_input, config):
    for value in event.values():
        print(value["messages"][-1].content)


# --- 第二轮对话（体现记忆） ---
print("\n--- 第二轮：追问别的 ---")
# 注意：我们依然使用同一个 config (相同的 thread_id)
three_input = {"messages": [("user", "这几个城市都在什么维度")]}
for event in agent_executor.stream(three_input, config):
    for value in event.values():
        print(value["messages"][-1].content)