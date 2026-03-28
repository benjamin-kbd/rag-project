import httpx
import sys
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

API_URL = "https://rag-api.onrender.com"  # 본인 URL로 변경

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path: str) -> str:
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text
    except ImportError:
        raise ImportError("pip install pdfplumber 필요")

def semantic_chunk(text: str) -> list[str]:
    print("BGE-M3 모델 로딩 중... (최초 1회 약 2GB 다운로드)")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda" if __import__("torch").cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("모델 로딩 완료, Semantic Chunking 시작...")

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )

    chunks = splitter.split_text(text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

    print(f"생성된 청크 수: {len(chunks)}")
    for i, c in enumerate(chunks[:3]):
        print(f"  [청크 {i+1}] {c[:80]}...")
    return chunks

def ingest_to_api(texts: list[str], metadata: list[dict] = None):
    print(f"\n{len(texts)}개 청크를 API로 전송 중...")
    res = httpx.post(
        f"{API_URL}/ingest",
        json={"texts": texts, "metadata": metadata or []},
        timeout=120,
    )
    res.raise_for_status()
    return res.json()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python ingest/ingest.py <파일경로>")
        print("예시: python ingest/ingest.py document.txt")
        print("예시: python ingest/ingest.py document.pdf")
        sys.exit(1)

    path = sys.argv[1]
    print(f"파일 로딩: {path}")

    if path.endswith(".pdf"):
        text = load_pdf(path)
    else:
        text = load_text(path)

    print(f"텍스트 길이: {len(text)} 글자")

    chunks = semantic_chunk(text)
    result = ingest_to_api(chunks)
    print(f"\n✅ 완료: {result}")
