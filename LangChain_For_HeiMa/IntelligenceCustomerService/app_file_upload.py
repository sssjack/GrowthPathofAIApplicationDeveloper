import time

import streamlit as st
from knowledge_base import KnowledgeBaseService

# 设置页面标题
st.title("知识库更新服务")


# file_uploader
uploader_file = st.file_uploader(
    "请上传 TXT / PDF / DOCX 文件",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=False,
)

# session_state就是一个字典
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    # 提取文件的信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024  # KB

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")

    with st.spinner("正在处理中..."):
        time.sleep(1)
        try:
            result = st.session_state["service"].upload_by_bytes(uploader_file.getvalue(), file_name)
            st.success(result)
        except Exception as exc:
            st.error(f"[失败]文件导入异常: {exc}")


