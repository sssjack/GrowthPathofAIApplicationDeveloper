from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self, source_filter=None):
        """返回向量检索器，方便加入chain"""
        search_kwargs = {"k": config.similarity_threshold}
        if source_filter:
            search_kwargs["filter"] = {"source": source_filter}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def list_sources(self):
        """返回当前知识库中已存在的文档来源列表"""
        data = self.vector_store.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        sources = sorted(
            {
                metadata.get("source")
                for metadata in metadatas
                if metadata and metadata.get("source")
            }
        )
        return sources

if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res = retriever.invoke("我的体重180斤，尺码推荐")
    print(res)
