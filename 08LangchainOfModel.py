from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

### 实例化模型
model = ChatTongyi(Model = "qwen-max",)

messages = [
    ("system", "你是一个边塞诗人"),
    SystemMessage(content="你是一个边塞诗人"),
    HumanMessage(content="帮我写一首边塞诗，主题是边塞"),
    AIMessage(content="锄禾日当午，汗滴血流土。谁是边塞诗人？粒粒皆辛苦"),
    HumanMessage(content="请按照上一个回复的格式，再写一首唐诗"),
]



res = model.stream(input = "你谁呀，能做啥?")

for chunk in model.stream(input = messages):
    print(chunk.content, end="", flush=True)
