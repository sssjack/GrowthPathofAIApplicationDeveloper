from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import FewShotPromptTemplate,PromptTemplate


example_template = PromptTemplate.from_template("单词：{input}，翻译：{output}")

example_data = [
    {"input": "正", "output": "反"},
    {"input": "黑", "output": "白"},
    {"input": "内", "output": "外"},

]



few_shot_prompt = FewShotPromptTemplate(
    examples=example_data,
    example_prompt=example_template,
    prefix="请给出给定次的反义，有如下示例：",

    suffix="基于示例告诉我：{word_input}，反例是？",
    input_variables=["word_input"],

)
prompt_text = few_shot_prompt.invoke(input= {"word_input":"优秀"}).to_string()
print(prompt_text)

model = Tongyi(Model = "qwen-max")

res = model.invoke(input = prompt_text)
print(res)
