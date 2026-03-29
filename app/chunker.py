from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str) -> list[str]:
    """
    일본어/영어 혼합 문서용 청킹
    Semantic Chunking은 Colab 전용 (2GB 모델 로딩)
    서버에서는 RecursiveCharacterTextSplitter 사용
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=[
            "\n\n",
            "\n",
            "。",   # 일본어 문장 끝
            "．",   # 일본어 마침표
            ". ",   # 영어 문장 끝
            "! ",
            "? ",
            "、",   # 일본어 쉼표
            " ",
            "",
        ],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]
    return chunks
