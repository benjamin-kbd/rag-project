import httpx
from app.config import settings

HF_LLM_URL = f"https://router.huggingface.co/hf-inference/models/{settings.HF_LLM_MODEL}/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {settings.HF_API_KEY}",
    "Content-Type": "application/json",
}

SYSTEM_PROMPT = """당신은 주어진 문서를 바탕으로 질문에 답변하는 AI 어시스턴트입니다.
반드시 제공된 컨텍스트만을 사용하여 답변하세요.
컨텍스트에 없는 정보는 '제공된 문서에서 해당 정보를 찾을 수 없습니다.'라고 답변하세요.
답변은 한국어로 작성하세요."""

async def generate_answer(question: str, contexts: list[str]) -> str:
    context_text = "\n\n---\n\n".join(
        [f"[문서 {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    )

    user_message = f"""다음 문서들을 참고하여 질문에 답변해주세요.

[참고 문서]
{context_text}

[질문]
{question}"""

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            HF_LLM_URL,
            headers=HEADERS,
            json={
                "model": settings.HF_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": 1024,
                "temperature": 0.1,
            },
        )
        response.raise_for_status()
        result = response.json()

    return result["choices"][0]["message"]["content"].strip()
