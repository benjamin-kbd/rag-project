"""
Google Colab에서 실행:
  1. PDF/텍스트 파일을 청크로 분할
  2. /ingest 엔드포인트로 전송
"""

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter

API_URL = "https://rag-project-cf8p.onrender.com"  # Render 배포 URL로 변경

def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text: str, chunk_size=500, chunk_overlap=50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)

def ingest_to_api(texts: list[str], metadata: list[dict] = None):
    with httpx.Client(timeout=120) as client:
        response = client.post(
            f"{API_URL}/ingest",
            json={"texts": texts, "metadata": metadata or []},
        )
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    # 예시: 텍스트 파일 업로드
    raw_text = load_text_file("document.txt")
    chunks = split_text(raw_text)
    print(f"총 청크 수: {len(chunks)}")

    result = ingest_to_api(chunks)
    print(result)