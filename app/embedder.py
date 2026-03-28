embedder_content = ''import httpx
from app.config import settings

# 새 엔드포인트로 변경
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{settings.HF_EMBED_MODEL}/pipeline/feature-extraction"
HEADERS = {
    "Authorization": f"Bearer {settings.HF_API_KEY}",
    "Content-Type": "application/json",
}

async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": text, "options": {"wait_for_model": True}},
        )
        response.raise_for_status()
        result = response.json()

    if isinstance(result, list) and isinstance(result[0], list):
        return result[0]
    return result

async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": texts, "options": {"wait_for_model": True}},
        )
        response.raise_for_status()
        return response.json()
'''

BASE = "/content/drive/MyDrive/rag-project"
with open(f"{BASE}/app/embedder.py", "w") as f:
    f.write(embedder_content)

print("✅ embedder.py 수정 완료")