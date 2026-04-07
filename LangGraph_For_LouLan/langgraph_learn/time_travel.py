# ====================== 1. 导入依赖 ======================
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver  # 内存记忆（必须）

# ====================== 2. 初始化模型 ======================
# 换成你自己的大模型（GPT、DeepSeek、Qwen 都可以）
llm = ChatTongyi(model="qwen-plus")

# ====================== 3. 定义节点（调用大模型） ======================
def call_model(state: MessagesState):
    """节点：调用大模型，返回新消息"""
    response = llm.invoke(state["messages"])
    return {"messages": response}

# ====================== 4. 构建图（带记忆存储） ======================
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

# ✅ 关键：开启记忆检查点（必须！否则无法回溯）
checkpointer = InMemorySaver()

# 编译图
graph = builder.compile(checkpointer=checkpointer)

# ====================== 5. 定义会话 ID（用于区分对话） ======================
# 同一个 thread_id 会共享记忆
config = {"configurable": {"thread_id": "session_101"}}

# ====================== 6. 第一轮对话 ======================
print("=" * 50)
print("【第一轮对话】问：湖南的省会是？")
graph.invoke(
    {"messages": [("user", "湖南的省会是？")]},
    config=config
)

# ====================== 7. 第二轮对话（自动带记忆） ======================
print("\n" + "=" * 50)
print("【第二轮对话】问：那湖北呢？")
graph.invoke(
    {"messages": [("user", "那湖北呢？")]},
    config=config
)

# ====================== ✨ 8. 时间回溯：回到第 1 轮对话结束的状态 ======================
print("\n" + "=" * 50)
print("🔄 【时间回溯】回到第一轮对话结束后！")

# 回溯核心代码：获取历史检查点
# 使用 get_tuple 方法获取完整的检查点信息
history = list(checkpointer.list(config))
print(f"\n历史检查点数量：{len(history)}")

# 打印所有检查点信息以便调试
for i, cp in enumerate(history):
    print(f"\n检查点 {i}:")
    print(f"  类型：{type(cp)}")
    if hasattr(cp, '_asdict'):
        # 如果是 namedtuple
        print(f"  内容：{cp._asdict()}")
    elif isinstance(cp, dict):
        print(f"  内容：{cp}")
    else:
        print(f"  内容：{cp}")

# 获取第一个对话后的检查点（索引 1）
if len(history) > 1:
    first_checkpoint = history[1]
    
    # 正确提取 checkpoint_id
    # InMemorySaver 返回的是 CheckpointTuple 命名元组
    # 包含：(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, metadata)
    if hasattr(first_checkpoint, 'checkpoint_id'):
        checkpoint_id = first_checkpoint.checkpoint_id
    elif hasattr(first_checkpoint, 'id'):
        checkpoint_id = first_checkpoint.id
    elif isinstance(first_checkpoint, (list, tuple)) and len(first_checkpoint) >= 3:
        # 如果是元组，第三个元素通常是 checkpoint_id
        checkpoint_id = first_checkpoint[2]
    else:
        raise ValueError(f"无法从检查点对象中提取 checkpoint_id: {first_checkpoint}")
    
    print(f"\n使用 checkpoint_id: {checkpoint_id}")
    
    # 构建回溯配置
    rollback_config = {
        "configurable": {
            "thread_id": "session_101",
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": ""  # 添加命名空间（可选）
        }
    }
else:
    raise ValueError("历史检查点数量不足，无法回溯")

# ====================== 9. 在回溯后的状态继续对话 ======================
print("\n【回溯后新对话】问：那江西呢？（不会记得湖北）")
try:
    result = graph.invoke(
        {"messages": [("user", "那江西呢？")]},
        config=rollback_config  # ✅ 使用回溯配置
    )
    
    # 打印最终回答
    print("\n【最终 AI 回答】：", result["messages"][-1].content)
except Exception as e:
    print(f"\n❌ 回溯失败：{e}")
    print("\n尝试使用简化配置...")
    # 如果复杂配置失败，尝试最简单的 checkpoint_id
    simple_config = {"configurable": {"thread_id": "session_101"}}
    result = graph.invoke(
        {"messages": [("user", "那江西呢？")]},
        config=simple_config
    )
    print("\n【简化配置结果】：", result["messages"][-1].content)