from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    selected_papers: Optional[List[str]] = None

class QueryResponse(BaseModel):
    answer: str
    retrieval_time: float
    generation_time: float
    total_time: float

