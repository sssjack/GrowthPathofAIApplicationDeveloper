# 子图与主图使用相同State
from operator import add
from typing import TypedDict, Annotated

from langgraph.constants import END
from langgraph.graph import StateGraph, START

# 1. 定义全局状态
class State(TypedDict):
    # Annotated[list[str], add] 表示：messages 是列表，更新时用 add 做追加操作（不是覆盖）
    messages: Annotated[list[str], add]

# --------------------------
# 2. 定义子图（Subgraph）
# --------------------------
def sub_node_1(state: State):
    # 子图节点：向 messages 追加一条响应
    return {"messages": ["response from subgraph"]}

# 构建子图
subgraph_builder = StateGraph(State)
subgraph_builder.add_node("sub_node_1", sub_node_1)
subgraph_builder.add_edge(START, "sub_node_1")
subgraph_builder.add_edge("sub_node_1", END)

# 编译子图（关键：必须编译后才能作为节点加入主图）
subgraph = subgraph_builder.compile()

# --------------------------
# 3. 定义主图（Parent graph）
# --------------------------
builder = StateGraph(State)
# 将编译好的子图作为一个节点加入主图
builder.add_node("subgraph_node", subgraph)
# 主图流程：START → 子图节点 → END
builder.add_edge(START, "subgraph_node")
builder.add_edge("subgraph_node", END)

# 编译主图
graph = builder.compile()

# --------------------------
# 4. 调用并打印结果
# --------------------------
result = graph.invoke({"messages": ["hello subgraph"]})
print(result)