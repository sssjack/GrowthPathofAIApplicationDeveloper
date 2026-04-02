from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver  # 对应图中 InMemorySaver
from langgraph.types import interrupt, Command  # 对应图中 Command
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool


# 1. 定义工具：在工具内部通过 interrupt 挂起逻辑
@tool(return_direct=True)  # return_direct 确保工具结果直接返回给用户
def book_hotel(hotel_name: str):
    """预定宾馆"""
    # 对应图 2：执行到这里，Agent 会将状态存入 checkpointer 并暂停
    response = interrupt(
        f"正准备执行 'book_hotel' 工具预定宾馆，相关参数名：{{'hotel_name': {hotel_name}}}。"
        "请选择 OK 表示同意，或者选择 edit 提出补充意见。"
    )

    # 对应图 2：根据人工输入的 Command 逻辑分支处理
    if response["type"] == "OK":
        pass  # 保持原有 hotel_name
    elif response["type"] == "edit":
        # 从 response 中获取编辑后的参数
        hotel_name = response.get("hotel_name", hotel_name)
    else:
        raise ValueError(f"Unknown response type: {response['type']}")

    return f"成功在 {hotel_name} 预定了一个房间。"


# 2. 初始化与配置
model = ChatTongyi(model="qwen3-max")
checkpointer = InMemorySaver()
agent = create_react_agent(model, tools=[book_hotel], checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}  # 对应图中的 thread_id

# --- 第一阶段：触发中断 ---
print("--- 步骤 1: 发起预定请求 ---")
input_data = {"messages": [("user", "帮我预定图灵宾馆")]}

for chunk in agent.stream(input_data, config):
    # 当输出包含 __interrupt__ 时，说明此时代码已经执行到了工具里的 interrupt() 这一行
    if "__interrupt__" in chunk:
        print(f"\n[系统中断] 等待用户确认: {chunk['__interrupt__'][0].value}")



# --- 第二阶段：提交 Command 恢复执行 ---
# 对应图 1：使用 Command(resume=...) 发送信号
print("\n--- 步骤 2: 模拟人工点击 OK 按钮 ---")

# 构造恢复命令
resume_command = Command(resume={"type": "edit","hotel_name": "三号宾馆"})

# 关键：调用 stream 时第一个参数传 Command 对象，config 保持一致
for chunk in agent.stream(resume_command, config):
    # 此时 Agent 会从 book_hotel 的 interrupt 处向下执行
    if 'tools' in chunk:
        print(f"工具执行结果: {chunk['tools']['messages'][-1].content}")