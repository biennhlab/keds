from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    rank: int = Field(ge=1)
    image_id: str
    score: float
    image_url: str


class SearchResponse(BaseModel):
    request_id: str
    model_version: str
    gallery_id: str
    gallery_version: str
    knowledge_version: str
    latency_ms: float
    results: list[SearchResult]
    details: dict

