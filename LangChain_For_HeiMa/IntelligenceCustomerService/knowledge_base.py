import hashlib
import io
import os
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import config_data as config
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader



def check_md5(md5_str: str):
    """检查传入的md5字符串是否已经被处理过了
    return False(md5未处理过) True(已经处理过, 已有记录)
    """
    if not os.path.exists(config.md5_path):
        # if进入表示文件不存在，那肯定没有处理过这个md5了
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()  # 处理字符串前后的空格和回车
            if line == md5_str:
                return True  # 已处理过

        return False

def save_md5(md5_str: str):
    """将传入的md5字符串，记录到文件内保存"""
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')

def get_string_md5(input_str: str, encoding='utf-8'):
    """将传入的字符串转换为md5字符串"""

    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)

    # 创建md5对象
    md5_obj = hashlib.md5()  # 得到md5对象
    md5_obj.update(str_bytes)  # 更新内容（传入即将要转换的字节数组）
    md5_hex = md5_obj.hexdigest()  # 得到md5的十六进制字符串

    return md5_hex


def normalize_text(text: str) -> str:
    """清洗提取后的文本，去掉空白行，方便后续切分和检索"""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def decode_text_bytes(file_bytes: bytes) -> str:
    """按常见编码顺序解析文本文件"""
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("TXT 文件编码无法识别，请保存为 UTF-8 或 GB18030 后重试")


def extract_docx_text(file_bytes: bytes) -> str:
    """从 docx 文件中提取正文文本"""
    with ZipFile(io.BytesIO(file_bytes)) as zip_file:
        xml_content = zip_file.read("word/document.xml")

    root = ET.fromstring(xml_content)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []

    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
        paragraph_text = "".join(texts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    return "\n".join(paragraphs)


def extract_pdf_text(file_bytes: bytes) -> str:
    """从 PDF 文件中提取文本"""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page in reader.pages:
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(page_text)

    return "\n".join(pages)


class KnowledgeBaseService(object):
    def __init__(self):
        # 如果文件夹不存在则创建，如果存在则跳过
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.collection_name,  # 数据库的表名
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory,  # 数据库本地存储文件夹
        )  # 向量存储的实例 Chroma向量库对象

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,  # 分割后的文本段最大长度
            chunk_overlap=config.chunk_overlap,  # 连续文本段之间的字符重叠数量
            separators=config.separators,  # 自然段落划分的符号
            length_function=len,  # 使用Python自带的len函数做长度统计的依据
        )  # 文本分割器的对象

    def extract_text_from_file(self, file_bytes: bytes, filename: str) -> str:
        """根据文件类型提取纯文本"""
        suffix = Path(filename).suffix.lower()

        if suffix == ".txt":
            text = decode_text_bytes(file_bytes)
        elif suffix == ".docx":
            text = extract_docx_text(file_bytes)
        elif suffix == ".pdf":
            text = extract_pdf_text(file_bytes)
        else:
            raise ValueError(f"暂不支持的文件类型: {suffix}")

        text = normalize_text(text)
        if not text:
            raise ValueError("文件解析成功，但未提取到可用文本内容")

        return text

    def upload_by_str(self, data: str, filename):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        data = normalize_text(data)
        if not data:
            return "[失败]没有可入库的文本内容"

        # 先得到传入字符串的md5值
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        if len(data) > config.max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "小曹",
        }

        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        save_md5(md5_hex)

        return "[成功]内容已经成功载入向量库"

    def upload_by_bytes(self, file_bytes: bytes, filename: str):
        """将上传的文件内容转为文本后入库"""
        text = self.extract_text_from_file(file_bytes, filename)
        return self.upload_by_str(text, filename)

    def upload_by_path(self, file_path: str):
        """按本地文件路径直接导入知识库"""
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        return self.upload_by_bytes(path.read_bytes(), path.name)

if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str(data="周杰伦SSS", filename="testfile")
    print(r)
