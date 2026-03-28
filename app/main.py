import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.rag import run_rag
from app.embedder import get_embeddings_batch
from app.vectorstore import ensure_collection, upsert_documents

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection()
    print("✅ RAG 서버 시작")
    yield

app = FastAPI(
    title="RAG API",
    description="BGE-M3 + Qdrant + Groq 기반 RAG 시스템",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class IngestRequest(BaseModel):
    texts: list[str]
    metadata: list[dict] | None = None

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API 서버가 실행 중입니다."}

@app.post("/query")
async def query(req: QueryRequest):
    try:
        result = await run_rag(req.question, top_k=req.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        embeddings = await get_embeddings_batch(req.texts)
        count = upsert_documents(req.texts, embeddings, req.metadata)
        return {"message": f"{count}개 문서가 저장되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/chat")
def chat_ui():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
