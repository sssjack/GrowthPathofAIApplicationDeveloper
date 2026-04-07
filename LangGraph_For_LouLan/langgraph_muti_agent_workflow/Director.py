import asyncio
from operator import add
from typing import TypedDict, Annotated

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_redis import RedisConfig, RedisVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph
import os
from langgraph.prebuilt import create_react_agent


nodes = ["supervisor", "travel", "couplet", "joke", "other"]

llm=ChatTongyi(
    model="qwen-plus"
)

class State(TypedDict):
    messages: Annotated[list[AnyMessage],add]
    type: str

def other_node(state: State):
    print(">>> other_node")
    writer = get_stream_writer()
    writer({"node":">>> other_node"})
    return {"messages": [HumanMessage(content="我暂时无法回答这个问题")], "type":"other"}

def supervisor_node(state: State):
    print(">>> supervisor_node")
    writer = get_stream_writer()
    writer({"node": ">>> supervisor_node"})
    # 根据用户的问题，对问题进行分类。分类结果保存到type当中
    prompt = """你是一个专业的客服助手，负责对用户的问题进行分类，并将任务分给其他Agent执行。
    如果用户的问题是和旅游路线规划相关的，那就返回 travel 。
    如果用户的问题是希望讲一个笑话，那就返回 joke 。
    如果用户的问题是希望对一个对联，那就返回 couplet 。
    如果是其他的问题，返回 other 。
    除了这几个选项外，不要返回任何其他的内容。
    """

    # 处理消息格式：可能是 HumanMessage 对象或字符串
    user_message = state["messages"][0]
    user_content = user_message.content if isinstance(user_message, HumanMessage) else user_message
    
    prompts = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content}
    ]

    # 如果已经有type属性了，表示问题已经交由其他节点处理完成了，就可以直接返回
    if "type" in state:
        writer({"supervisor_step", f"已获得{state['type']} 智能体处理结果"})
        return {"type":END}
    else:
        response = llm.invoke(prompts)
        typeRes = response.content
        writer({"supervisor_step", f"问题分类结果: {typeRes}"})
        if typeRes in nodes:
            return {"type":typeRes}
        else:
            raise ValueError("type is not in (travel,joke,other,couplet)")

async def travel_node(state: State):
    print(">>> travel_node")
    writer = get_stream_writer()
    writer({"node": ">>> travel_node"})
    system_prompt = "你是一个专业的旅行规划助手，根据用户的问题，生成一个旅行路线规划，按照你平时给cursor返回的数据提示语，用中文回答，包含路线，耗时，注意时间，出行建议等，并返回一个不超过300字的规划结果"

    # 处理消息格式：可能是 HumanMessage 对象或字符串
    user_message = state["messages"][0]
    user_content = user_message.content if isinstance(user_message, HumanMessage) else user_message
    
    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    client = MultiServerMCPClient(
        {
            "amap-maps": {
                "url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp",
                "headers": {
                    "Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}"
                },
                "transport": "streamable_http"
            }
        }
    )
    tools = await client.get_tools()  # 注意加 await
    agent = create_react_agent(model=llm, tools=tools)
    response = await agent.ainvoke({"messages": prompts})

    last_message = response["messages"][-1]
    writer({"travel_result": last_message.content})
    return {"messages": [HumanMessage(content=last_message.content)], "type": "travel"}

def joke_node(state: State):
    print(">>> joke_node")
    writer = get_stream_writer()
    writer({"node": ">>>> joke_node"})

    system_prompt = "你是一个笑话大师，根据用户的问题，写一个不超过100个字的笑话。"

    # 处理消息格式：可能是 HumanMessage 对象或字符串
    user_message = state["messages"][0]
    user_content = user_message.content if isinstance(user_message, HumanMessage) else user_message
    
    prompts=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    response = llm.invoke(prompts)

    return {"messages": [HumanMessage(content = response.content)], "type": "joke"}


def couplet_node(state: State):
    print(">>> couplet_node")
    writer = get_stream_writer()
    writer({"node": ">>> couplet_node"})

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
你是一个专业的对联大师，你的任务是根据用户给出的上联，设计一个下联。
回答时，可以参考下面的参考对联。
参考对联:
{samples}
请用中文回答问题
"""),
        ("user", "{text}")
    ])

    query = state["messages"][0].content if isinstance(state["messages"][0], HumanMessage) else state["messages"][0]



    embedding_model = DashScopeEmbeddings(model="text-embedding-v1")
    redis_url = "redis://localhost:6379"
    config = RedisConfig(index_name="couplet", redis_url=redis_url)
    vector_store = RedisVectorStore(embedding_model, config=config)

    samples = []
    scored_results = vector_store.similarity_search_with_score(query, k=10)
    for doc, score in scored_results:
        samples.append(doc.page_content)

    prompt = prompt_template.invoke({"samples": samples, "text": query})
    writer({"couplet_prompt": prompt})

    response = llm.invoke(prompt)
    return {"messages": [HumanMessage(content=response.content)], "type": "couplet"}

# 条件路由
def routing_func(state: State):
    if state["type"] == "travel":
        return "travel_node"
    elif state["type"] == "joke":
        return "joke_node"
    elif state["type"] == "couplet":
        return "couplet_node"
    elif state["type"] == END:
        return END
    else:
        return "other_node"

# 构建图
builder = StateGraph(State)
# 添加节点
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("travel_node",travel_node)
builder.add_node("joke_node",joke_node)
builder.add_node("couplet_node",couplet_node)
builder.add_node("other_node",other_node)
# 添加Edge
builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node",routing_func, ["travel_node","joke_node","couplet_node","other_node",END])
builder.add_edge(start_key="travel_node", end_key="supervisor_node")
builder.add_edge(start_key="joke_node", end_key="supervisor_node")
builder.add_edge(start_key="couplet_node", end_key="supervisor_node")
builder.add_edge(start_key="other_node", end_key="supervisor_node")

# 构建Graph
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer = checkpointer)


# if __name__ == "__main__":
#     config = {
#         "configurable": {
#             "thread_id": "1"
#         }
#     }
#
#     async def main():
#         res = await graph.ainvoke(
#             {"messages": [HumanMessage(content="给我对一个对联，上联是金榜题名时")]},
#             config
#         )
#         print(res["messages"][-1].content)
#
#     asyncio.run(main())

# 执行任务的测试代码
if __name__ == "__main__":
    # 1. 配置会话 ID（用于共享记忆）
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    # 2. 流式调用图
    # 注意：使用 graph.stream 时，input 必须是字典格式
    for chunk in graph.stream(
        input={"messages": ["给我对一个对联，上联是：金榜题名时"]},
        config=config,
        stream_mode="custom"
    ):
        # 打印每个流式数据块
        print(chunk)
