from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sympy.physics.units import length

loader = TextLoader(
    "./stu.txt",
    encoding="utf-8"
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
    length_function=len,
)
splitter_docs = splitter.split_documents(docs)