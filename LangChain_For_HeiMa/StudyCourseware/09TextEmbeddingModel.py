from langchain_community.embeddings import DashScopeEmbeddings

model = DashScopeEmbeddings()

print(model.embed_query("I Love You"))
print(model.embed_documents( ["I Love You", "I Love You Too", "I Love You Too Too"]))
