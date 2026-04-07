import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# --------------------------
# 1. 初始化配置与查询
# --------------------------
query = "帮我对个对联，上联是：瑞雪兆丰年"

# 设置百炼API密钥环境变量

# 初始化通义千问嵌入模型
embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

# --------------------------
# 2. 加载FAISS向量数据库
# --------------------------
vector_store_path = "./faiss_couplet_index"
try:
    vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
    print("已加载 FAISS 向量库")
except Exception as e:
    print(f"错误：无法加载向量库，请先运行 CoupletLoader.py 创建向量库。错误信息：{e}")
    exit(1)

# --------------------------
# 3. 向量检索（召回相似对联）
# --------------------------
samples = []
# 相似度检索，取Top10最相关的参考对联
scored_results = vector_store.similarity_search_with_score(query, k=10)
for doc, score in scored_results:
    # print(f"{doc.page_content} - {score}")
    samples.append(doc.page_content)

# --------------------------
# 4. 构建RAG提示词模板
# --------------------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的对联大师，你的任务是根据用户给出的上联，设计一个下联。
回答时，可以参考下面的参考对联。
参考对联:
    {samples}
请用中文回答问题
"""),
    ("user", "{text}")
])

# 填充模板，生成最终提示词
prompt = prompt_template.invoke({"samples": samples, "text": query})
print(prompt)

# --------------------------
# 5. 调用大模型生成下联
# --------------------------
llm = ChatTongyi(
    model="qwen-plus",

)

# 执行推理并打印结果
print(llm.invoke(prompt))