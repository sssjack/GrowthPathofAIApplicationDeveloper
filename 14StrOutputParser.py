from asyncio import Runner

from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
#创建所需的解析器
parser = StrOutputParser()
json_parser = JsonOutputParser()

model = ChatTongyi(Model = "qwen3-max")

#第一个提示词模板
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了一个{firstname}，并封装为json格式返回，要求key是name，value就是你的名字，严格按照格式返回"
)

#第二个提示词模板
second_prompt = PromptTemplate.from_template(

    "我邻居的姓名是{name}，请你帮我解释一下，"
)



#构建链
first_chain = first_prompt | model | json_parser | second_prompt | model | parser

res = first_chain.stream({"lastname": "张三", "firstname": "女儿"})

for chunk in res:
    print(chunk, end="", flush=True)


res_function = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})

res = first_prompt | model | res_function | second_prompt | model | parser

for chunk in res.stream({"lastname": "张三", "firstname": "女儿"}):
    print(chunk, end="", flush=True)
