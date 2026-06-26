from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag_service

app = FastAPI(title="Research IQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    answer: str
    response_time_sec: float
    sources: list[dict]


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "papers_loaded": rag_service.has_papers(),
        "paper_count": len(rag_service.list_papers()),
    }


@app.get("/api/papers")
def get_papers():
    return {"papers": rag_service.list_papers()}


@app.post("/api/papers/upload")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        saved_name = rag_service.save_paper(file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Research paper uploaded successfully.",
        "filename": saved_name,
        "papers": rag_service.list_papers(),
    }


@app.delete("/api/papers/{filename}")
def remove_paper(filename: str):
    if not rag_service.delete_paper(filename):
        raise HTTPException(status_code=404, detail="Paper not found.")
    return {"message": "Paper deleted.", "papers": rag_service.list_papers()}


@app.post("/api/query", response_model=QueryResponse)
def ask_question(payload: QueryRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag_service.query_papers(question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to process your question. Please try again.",
        ) from exc

    return result
