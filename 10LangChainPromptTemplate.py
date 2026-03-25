from statistics import mode

from langchain_classic.chains.summarize.map_reduce_prompt import prompt_template
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "我的朋友是{friend}，他在{school}上学，请你给他一些 建议。"
)

model = Tongyi(Model = "qwen-max")
chain = prompt_template | model

print(chain.invoke(input= { "friend":"小王", "school":"东京大学"}))
