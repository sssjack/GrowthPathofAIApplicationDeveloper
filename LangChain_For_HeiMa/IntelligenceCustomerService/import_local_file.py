import sys

from knowledge_base import KnowledgeBaseService


def main():
    if len(sys.argv) < 2:
        print("用法: python import_local_file.py <文件路径>")
        raise SystemExit(1)

    file_path = sys.argv[1]
    service = KnowledgeBaseService()
    result = service.upload_by_path(file_path)
    print(result)


if __name__ == "__main__":
    main()
