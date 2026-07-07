import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str = Field(min_length=8)


class UserOut(BaseModel):
    id: uuid.UUID
    name: str
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PaperOut(BaseModel):
    id: uuid.UUID
    filename: str
    storage_url: str
    created_at: datetime

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    response_time_sec: float
    sources: list[SourceChunk]


class SummarizeRequest(BaseModel):
    filename: str
    length: str = "brief"


class SummarizeResponse(BaseModel):
    summary: str
    chunks_processed: int
    used_map_reduce: bool
    response_time_sec: float