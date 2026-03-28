from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # HuggingFace
    HF_API_KEY: str
    HF_EMBED_MODEL: str = "BAAI/bge-m3"
    HF_LLM_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION: str = "rag_collection"


    class Config:
        env_file = ".env"

settings = Settings()