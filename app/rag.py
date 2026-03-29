from app.embedder import get_embedding
from app.vectorstore import search_similar
from app.reranker import rerank
from app.llm import generate_answer

async def run_rag(question: str, top_k: int = 5) -> dict:
    """Semantic Search + Reranking RAG 파이프라인"""

    # 1. 질문 임베딩
    query_vector = await get_embedding(question)

    # 2. 벡터 유사도 검색 (넉넉하게 top_k * 2)
    search_results = search_similar(query_vector, top_k=top_k * 2)

    if not search_results:
        return {
            "answer": "관련 문서를 찾을 수 없습니다.",
            "sources": [],
            "question": question,
        }

    # 3. BGE Reranker로 정밀 재정렬
    documents = [r["text"] for r in search_results]
    reranked = await rerank(question, documents, top_k=3)

    # 4. 재랭킹된 상위 3개로 LLM 답변 생성
    contexts = [r["text"] for r in reranked]
    answer = await generate_answer(question, contexts)

    return {
        "answer": answer,
        "question": question,
        "sources": [
            {
                "text": r["text"][:200] + "...",
                "score": round(r["score"], 4),
                "reranked": True,
            }
            for r in reranked
        ],
    }
