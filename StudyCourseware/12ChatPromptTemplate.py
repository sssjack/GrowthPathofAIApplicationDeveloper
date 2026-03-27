from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from openai import models

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗："),
        MessagesPlaceholder("history"),
        ("human", "再来一首唐诗"),
    ])

history_data = [
    ("human", "你来作一首唐诗？"),
    ("assistant", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "请继续"),
    ("assistant", "白日依山尽，黄河入海流。欲穷千里目，更上一层楼"),

]

prompt_text = chat_template.invoke({"history": history_data}).to_string()

model = ChatTongyi(model="qwen3-max")
res = model.invoke(input = prompt_text)
print(res.content,type( res))