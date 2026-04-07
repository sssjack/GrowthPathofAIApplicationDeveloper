import random

from Director import graph
from langchain_core.messages import HumanMessage

config = {
    "configurable":{
        "thread_id": random.randint(1, 10000)
    }
}

query="请给我讲一个冯巩的笑话  "
res = graph.invoke({"messages": [HumanMessage(content=query)]}
    , config)
print(res["messages"][-1].content)