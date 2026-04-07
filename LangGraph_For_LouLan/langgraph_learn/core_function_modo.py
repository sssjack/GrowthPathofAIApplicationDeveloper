from typing import TypedDict, Annotated, Literal, Optional

class TicketState(TypedDict):
    ticket_id: str
    raw_text: str
    extracted_info: dict
    classification: str
    human_decision: Optional[str]
    response: str
    retry_count: Annotated[int, lambda x: min(x, 3)]
    thread_id: str
    checkpoint_id: Optional[str]


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 子图状态
class ExtractInfoState(TypedDict):
    text: str
    order_id: str
    issue_type: str

# 提取信息节点
def extract_order_id(state: ExtractInfoState) -> ExtractInfoState:
    # 模拟从文本中提取订单号
    state["order_id"] = "ORDER123" if "order" in state["text"] else "N/A"
    return state

def extract_issue_type(state: ExtractInfoState) -> ExtractInfoState:
    # 模拟从文本中提取问题类型
    if "refund" in state["text"]:
        state["issue_type"] = "refund"
    elif "support" in state["text"]:
        state["issue_type"] = "support"
    else:
        state["issue_type"] = "inquiry"
    return state

# 构建子图
extract_info_subgraph = StateGraph(ExtractInfoState)
extract_info_subgraph.add_node("extract_order_id", extract_order_id)
extract_info_subgraph.add_node("extract_issue_type", extract_issue_type)
extract_info_subgraph.add_edge("extract_order_id", "extract_issue_type")
extract_info_subgraph.set_entry_point("extract_order_id")
extract_info_subgraph.set_finish_point("extract_issue_type")
extract_info_agent = extract_info_subgraph.compile()


# 主图节点
def ocr_scan(state: TicketState) -> TicketState:
    # 模拟 OCR 提取全文
    state["raw_text"] = "Dear Support, I need a refund for my order ORDER123. Regards, John Doe"
    return state

def extract_info(state: TicketState) -> TicketState:
    # 调用子图提取信息
    result = extract_info_agent.invoke({"text": state["raw_text"]})
    state["extracted_info"] = {"order_id": result["order_id"], "issue_type": result["issue_type"]}
    return state

def classify_ticket(state: TicketState) -> TicketState:
    # 模拟分类
    state["classification"] = state["extracted_info"]["issue_type"]
    return state

def await_human_decision(state: TicketState) -> TicketState:
    # 此节点将被中断
    return state

def generate_response(state: TicketState) -> TicketState:
    # 根据分类生成回复
    if state["classification"] == "refund":
        state["response"] = "We are processing your refund request for order ORDER123."
    elif state["classification"] == "support":
        state["response"] = "Our support team will contact you shortly regarding your issue."
    else:
        state["response"] = "Thank you for your inquiry. We will get back to you soon."
    return state

# 构建主图
workflow = StateGraph(TicketState)

workflow.add_node("ocr_scan", ocr_scan)
workflow.add_node("extract_info", extract_info)
workflow.add_node("classify_ticket", classify_ticket)
workflow.add_node("await_human_decision", await_human_decision)
workflow.add_node("generate_response", generate_response)

# 连接主流程
workflow.add_edge(START, "ocr_scan")
workflow.add_edge("ocr_scan", "extract_info")
workflow.add_edge("extract_info", "classify_ticket")
workflow.add_edge("classify_ticket", "await_human_decision")
workflow.add_edge("await_human_decision", "generate_response")
workflow.add_edge("generate_response", END)

# 条件边：提取信息失败则重试 OCR
def should_retry_extract(state: TicketState) -> Literal["retry_ocr", "__end__"]:
    if state["extracted_info"]["order_id"] == "N/A" and state["retry_count"] < 3:
        return "retry_ocr"
    return "__end__"

workflow.add_conditional_edges(
    "extract_info",
    should_retry_extract,
    {
        "retry_ocr": "ocr_scan",  # 循环回 OCR
        "__end__": "classify_ticket"
    }
)

# 设置人工干预点
workflow.set_interrupt("await_human_decision")

# 编译（启用检查点）
app = workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    print("-ticket_processing_agent Demo —— 客户支持工单处理\n")

    # 初始化工单
    initial_state = TicketState(
        ticket_id="TICKET-2026-001",
        raw_text="",
        extracted_info={},
        classification="",
        human_decision=None,
        response="",
        retry_count=0,
        thread_id="ticket_20260403_001",
        checkpoint_id=None
    )

    print("➡️ STEP 1: 启动工单处理（自动 OCR + 提取信息 + 初步分类）...")
    for event in app.stream(initial_state, config={"configurable": {"thread_id": "ticket_20260403_001"}},
                            stream_mode="events"):
        if event.get("event") == "checkpoint":
            print(f"✅ Checkpoint saved: {event['checkpoint_id']}")
        elif event.get("event") == "interrupt":
            print(f"🛑 INTERRUPTED at 'await_human_decision' —— 请人工审核！")
            break

    # 获取当前状态（AI 已初步分类）
    snapshot = app.get_state(config={"configurable": {"thread_id": "ticket_20260403_001"}})
    print(f"\n🔍 AI 初步分类结果:")
    print(f"   • 分类: {snapshot.values['classification']}")

    # 模拟人工输入（真实中为前端表单提交）
    print("\n💬 人工输入: \"APPROVE\"")
    app.update_state(
        config={"configurable": {"thread_id": "ticket_20260403_001"}},
        values={"human_decision": "APPROVE"},
        as_node="await_human_decision"
    )

    # 继续执行生成回复
    print("\n➡️ STEP 2: 生成并发送回复...")
    final = app.invoke(None, config={"configurable": {"thread_id": "ticket_20260403_001"}})
    print(f"🎉 回复完成！回复内容: {final['response']}")

    # 展示回溯能力：客户改需求后重审
    print("\n🔄 回溯演示：客户要求重新分类...")
    # 加载上一个 checkpoint（AI 初判后、人工决策前）
    last_cp = snapshot.config["configurable"]["checkpoint_id"]
    revised_state = app.get_state(
        config={"configurable": {"thread_id": "ticket_20260403_001", "checkpoint_id": last_cp}})

    # 注入新分类（模拟客户邮件）
    new_classification = "support"
    app.update_state(
        config={"configurable": {"thread_id": "ticket_20260403_001", "checkpoint_id": last_cp}},
        values={"classification": new_classification},
        as_node="classify_ticket"
    )

    print("✅ 已注入新分类，正在重跑处理流程...")
    final_revised = app.invoke(None, config={"configurable": {"thread_id": "ticket_20260403_001"}})
    print(f"🎯 重审结果: {final_revised['response']} → 全部 OK！")

    print("\n✨ ticket_processing_agent —— 让客服专注决策，而非重复劳动。")
