import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.chat_models import ChatTongyi
from langgraph.prebuilt import create_react_agent

# 开启调试模式（使用环境变量方式）
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"


async def run_mcp_example():
    # 1. 使用本地 MCP 服务（更稳定可靠）
    mcp_config = {
        "math": {
            "command": "python",
            "args": ["langgraph_learn/examples/math_server.py"],
            "transport": "stdio",
        }
    }

    # 2. 初始化客户端（不使用上下文管理器）
    client = MultiServerMCPClient(mcp_config)
    
    # 3. 获取并打印工具列表
    tools = await client.get_tools()
    print(f"\n=== 成功加载 {len(tools)} 个工具 ===")
    for i, tool in enumerate(tools, 1):
        print(f"  {i}. {tool.name}: {tool.description}")

        # 4. 初始化模型
        llm = ChatTongyi(model="qwen-plus")
        
        # 5. 创建 ReAct Agent
        agent = create_react_agent(model=llm, tools=tools)
        
        # 6. 执行数学计算问题
        query = "计算 (3 + 5) × 12 的结果是多少？"
        
        print(f"\n=== 开始调用 ===")
        print(f"问题：{query}\n")
                
        async for event in agent.astream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values"
        ):
            last_message = event["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                print(f"[模型思考] 调用工具：{last_message.tool_calls[0]['name']}")
            elif last_message.type == "tool":
                print(f"[工具返回] 收到计算结果")
        
        # 最终回答
        final_answer = (await agent.ainvoke({"messages": [{"role": "user", "content": query}]}))["messages"][-1].content
        print("\n=== 最终回答 ===")
        print(final_answer)


if __name__ == "__main__":
    try:
        asyncio.run(run_mcp_example())
    except KeyboardInterrupt:
        pass