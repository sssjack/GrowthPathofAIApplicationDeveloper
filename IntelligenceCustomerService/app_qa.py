import time
from rag import RagService
import streamlit as st
import config_data as config
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings

# 标题
st.title("智能客服")
st.divider()  # 分隔符

if "vector_service" not in st.session_state:
    st.session_state["vector_service"] = VectorStoreService(
        embedding=DashScopeEmbeddings(model=config.embedding_model_name)
    )

source_options = ["全部文档"] + st.session_state["vector_service"].list_sources()
selected_source_label = st.sidebar.selectbox("参考文档范围", source_options)
selected_source = None if selected_source_label == "全部文档" else selected_source_label

if st.sidebar.button("清空当前会话历史"):
    get_history(config.session_config["configurable"]["session_id"]).clear()
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]
    st.rerun()

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

if st.session_state.get("selected_source") != selected_source or "rag" not in st.session_state:
    st.session_state["selected_source"] = selected_source
    st.session_state["rag"] = RagService(source_filter=selected_source)
    st.session_state["message"] = [
        {
            "role": "assistant",
            "content": "你好，有什么可以帮助你？"
            if not selected_source
            else f"你好，我会优先参考《{selected_source}》来回答你的问题。",
        }
    ]

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 在页面最下方提供用户输入栏
prompt = st.chat_input()

if prompt:
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    with st.spinner("AI 思考中..."):
        res = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)
        response_content = st.chat_message("assistant").write_stream(res)
        # 往 session_state 中添加 response_content
        time.sleep(1)
        st.session_state["message"].append({"role": "assistant", "content": response_content})
