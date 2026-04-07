from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from rich.jupyter import display


# 1. 定义状态结构
class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

# 2. 定义节点函数
def node_1(state: InputState) -> OverallState:
    return {"foo": state["user_input"] + "->学院"}

def node_2(state: OverallState) -> PrivateState:
    return {"bar": state["foo"] + "->非常"}

def node_3(state: PrivateState) -> OutputState:
    return {"graph_output": state["bar"] + "->靠谱"}

# 3. 构建图 ———— 这里修复了！！！
builder = StateGraph(
    OverallState,
    input_schema=InputState,    # 旧：input
    output_schema=OutputState  # 旧：output
)

# 添加节点
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 定义边
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

# 4. 编译并运行
graph = builder.compile()
result = graph.invoke({"user_input": "图灵"})
print(result)

from IPython.display import Image, display

# draw_mermaid方法可以打印出Graph的mermaid代码。
display(Image(graph.get_graph().draw_mermaid_png()))
graph.get_graph().draw_mermaid_png(output_file_path="workflow.png")
print("流程图已保存为 workflow.png")

