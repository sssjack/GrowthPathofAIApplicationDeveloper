import os
import redis
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore

# --------------------------
# 1. 初始化嵌入模型与环境
# --------------------------
# 设置百炼API密钥环境变量


# 初始化通义千问嵌入模型 (text-embedding-v1)
embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

# --------------------------
# 2. 连接Redis向量数据库
# --------------------------
redis_url = "redis://localhost:6379"
redis_client = redis.from_url(redis_url)
print(redis_client.ping())  # 测试连接：返回True表示成功

# 配置Redis向量索引
config = RedisConfig(
    index_name="couplet",
    redis_url=redis_url
)

# 初始化向量存储实例
vector_store = RedisVectorStore(embedding_model, config=config)
# --------------------------
# 4. 加载CSV文件数据并存入向量库
# --------------------------
lines = []
# 打开对联测试CSV文件 (相对路径：../resource/couplettest.csv)
with open("../resource/train.csv", "r", encoding="utf-8") as file:
    for line in file:
        print(line)       # 打印每一行 (可用于调试)
        lines.append(line)# 收集文本到列表

# 将CSV文本数据添加到Redis向量数据库
vector_store.add_texts(lines)

# --------------------------
# 5. RAG 提示词模板 (截图中显示的Prompt)
# --------------------------
promptTemplate = """你是一个专业的对联大师。
你的任务是根据给定的参考对联，给用户的问题设计一个下联。
参考对联：
{context}  # {context} 是检索出来的参考文档
用户问题：
{question} # {question} 是用户的问题
请用中文回答用户问题。
"""

# 格式化Prompt (示例：需传入context和question变量)
# formatted_prompt = promptTemplate.format(context="参考对联内容", question="用户的上联")