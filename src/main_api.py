from fastapi import FastAPI, HTTPException
from rag_service import RAGService
from schemas import QueryRequest, QueryResponse

app = FastAPI(title="RAG Service")

rag = RAGService()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    try:
        return rag.answer(req.query, req.selected_papers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
