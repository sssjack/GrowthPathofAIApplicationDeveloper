# 1. 导入核心模块
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver

# 2. 构建大模型客户端（需替换为你的 API Key）
llm = ChatTongyi(
    model="qwen-plus",
    # api_key="你的-百炼-API-Key",
)

# 3. 定义节点：调用大模型
def call_model(state: MessagesState):
    # 传入历史消息，调用大模型
    response = llm.invoke(state["messages"])
    # 返回新消息，更新状态
    return {"messages": response}

# 4. 构建图并配置检查点（记忆）
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)  # 给节点命名，更规范
builder.add_edge(START, "call_model")

# 初始化内存检查点（用于保存对话状态）
checkpointer = InMemorySaver()

# 编译图，开启检查点功能
graph = builder.compile(checkpointer=checkpointer)

# 5. 配置会话 ID
# 通过 thread_id 标识一个独立的对话会话，实现多轮对话隔离
config = {
    "configurable": {
        "thread_id": "1"
    }
}

# 6. 第一轮对话：湖南
print("=== 第一轮：湖南 ===")
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "湖南的省会是哪里？"}]},
    config=config,
    stream_mode="values"
):
    # 打印本轮最后一条消息（即模型回答）
    chunk["messages"][-1].pretty_print()

# 7. 第二轮对话：湖北（利用记忆）
print("\n=== 第二轮：湖北 ===")
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "湖北呢？"}]},  # 省略上下文，模型会自动回忆
    config=config,
    stream_mode="values"
):
    chunk["messages"][-1].pretty_print()