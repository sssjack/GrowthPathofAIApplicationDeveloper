from langchain_community.chat_models import ChatTongyi
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 初始化模型与向量库
model = ChatTongyi(model="qwen3-max")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料:{context}。"),
        ("user", "用户提问:{input}")
    ]
)

vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(model="text-embedding-v4"))

# 准备向量库数据
vector_store.add_texts(
    ["减肥就是要少吃多练", "在减脂期间吃东西很重要，清淡少油控制卡路里摄入并运动起来", "跑步是很好的运动哦"]
)

input_text = "怎么减肥？"

# 初始化检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 文档格式化函数
def format_func(docs: list):
    if not docs:
        return "无相关参考资料"
    formatted_str = "["
    for doc in docs:
        formatted_str += doc.page_content
    formatted_str += "]"
    return formatted_str

# 打印提示词函数
def print_prompt(prompt):
    print(prompt.to_string())
    print("="*20)
    return prompt

# 构建RAG链
chain = (
    {"input": RunnablePassthrough(), "context": retriever | format_func}
    | prompt
    | print_prompt
    | model
    | StrOutputParser()
)

# 执行并输出结果
res = chain.invoke(input_text)
print(res)