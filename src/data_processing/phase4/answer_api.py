import uvicorn
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from .settings import get_settings

settings = get_settings()

app = FastAPI(title="QuantConnect RAG – Answer API")

# instantiate heavy stuff once
from langchain_openai import OpenAIEmbeddings
from .retriever import HybridRetriever
from .cache import SemanticCache
from .answer_chain import AnswerChain
from typing import Dict

embedding_model = OpenAIEmbeddings(
    model=settings.embedding_model,
)
embedding_fn = embedding_model.embed_query


qdrant_client = QdrantClient(url=settings.qdrant_url)
retriever = HybridRetriever(bm25_endpoint=settings.bm25_endpoint,
                            qdrant_client=qdrant_client,
                            collection_name=settings.qdrant_collection)

cache = SemanticCache(redis_url=settings.redis_url)
chain = AnswerChain(retriever, cache, embedding_fn)

# middleware for latency


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    import time

    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(duration)
    return response


@app.post("/answer")
async def answer_endpoint(body: Dict[str, str], x_api_key: str | None = Header(default=None)):
    if settings.api_key_header and x_api_key != settings.api_key_header:
        raise HTTPException(status_code=401, detail="Invalid API key")
    query = body.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    try:
        response = chain(query)
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)

