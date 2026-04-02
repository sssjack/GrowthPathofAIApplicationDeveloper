from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

def print_prompt(full_prompt):
    print("="*20, full_prompt.to_string(), "="*20)
    return full_prompt

model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template(
    "你需要根据对话历史回应用户问题。对话历史:\n{chat_history}。用户当前输入: {input}， 请给出回应"
)

base_chain = prompt | print_prompt | model | StrOutputParser()

chat_history_store = {}  # 存放多个会话ID所对应的历史会话记录

def get_history(session_id):
    if session_id not in chat_history_store:
        # 存入新的实例
        chat_history_store[session_id] = InMemoryChatMessageHistory()
    return chat_history_store[session_id]

# 通过RunnableWithMessageHistory获取一个新的带有历史记录功能的chain
conversation_chain = RunnableWithMessageHistory(
    base_chain,          # 被附加历史消息的Runnable，通常是chain
    get_history,         # 获取历史会话的函数
    input_messages_key="input",  # 声明用户输入消息在模板中的占位符
    history_messages_key="chat_history"  # 声明历史消息在模板中的占位符
)

if __name__ == '__main__':
    # 如下固定格式，配置当前会话的ID
    session_config = {"configurable": {"session_id": "user_001"}}

    print(conversation_chain.invoke({"input": "小明有一只猫"}, session_config))
    print(conversation_chain.invoke({"input": "小刚有两只狗"}, session_config))
    print(conversation_chain.invoke({"input": "共有几只宠物？"}, session_config))