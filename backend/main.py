from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from . import models
from . import rag_service
from . import schemas
from . import security
from . database import get_db, init_db
from . deps import get_current_user

app = FastAPI(title="Research IQ API")

# Dev-friendly CORS so the Streamlit frontend (different port) can call this API.
# Tighten allow_origins to your actual frontend URL before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def _user_key(user: models.User) -> str:
    """Storage/vector-index folder key for a user. Uses the id, not the
    name/email, so it stays stable and unique."""
    return str(user.id)


@app.post("/auth/signup", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def signup(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    user = models.User(
        name=payload.name,
        email=payload.email,
        hashed_password=security.hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # OAuth2PasswordRequestForm's "username" field carries the email.
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token = security.create_access_token(subject=str(user.id))
    return schemas.Token(access_token=token)


@app.get("/papers", response_model=list[schemas.PaperOut])
def get_papers(db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    return rag_service.list_papers(db, user.id)


@app.post("/papers", response_model=schemas.PaperOut, status_code=status.HTTP_201_CREATED)
async def upload_paper(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    content = await file.read()
    try:
        paper = rag_service.save_paper(db, user.id, _user_key(user), file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return paper


@app.delete("/papers/{filename}")
def remove_paper(
    filename: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    try:
        deleted = rag_service.delete_paper(db, user.id, _user_key(user), filename)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return {"deleted": True}


@app.post("/query", response_model=schemas.QueryResponse)
def query(payload: schemas.QueryRequest, user: models.User = Depends(get_current_user)):
    try:
        return rag_service.query_papers(_user_key(user), payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/summarize", response_model=schemas.SummarizeResponse)
def summarize(payload: schemas.SummarizeRequest, user: models.User = Depends(get_current_user)):
    try:
        return rag_service.summarize_paper(_user_key(user), payload.filename, payload.length)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok"}