from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models import ChatTongyi

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗："),
        MessagesPlaceholder("history"),
        ("human", "再来一首唐诗"),
    ]
)

history_data = [
    ("human", "你来作一首唐诗？"),
    ("assistant", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "请继续"),
    ("assistant", "白日依山尽，黄河入海流。欲穷千里目，更上一层楼"),

]

# prompt = chat_prompt_template.invoke({"history": history_data}).to_string()
# print(prompt)

model = ChatTongyi(model="qwen3-max")

# 组成一个完整的对话
chain = chat_prompt_template | model

# res = chain.invoke({"history": history_data})
# print(res.content)

for res in chain.stream({"history": history_data}):
    print(res.content, end="", flush=True)