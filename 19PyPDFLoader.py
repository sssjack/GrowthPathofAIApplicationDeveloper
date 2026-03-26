from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path= "./stu.pdf",
    mode="single"
)

i = 0
for page in loader.lazy_load():
    i += 1
    print(type(page), page)