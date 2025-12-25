from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.chatbot import FAQChatbot
from app.config import FAQ_PATH, API_CONFIG

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title=API_CONFIG["title"],
    version=API_CONFIG["version"],
    description=API_CONFIG["description"],
)

# =========================
# STATIC + TEMPLATES (UI)
# =========================
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# =========================
# LOAD CHATBOT
# =========================
chatbot = FAQChatbot(str(FAQ_PATH), enable_rag=True)

# =========================
# SCHEMAS
# =========================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Pertanyaan dari user")

class ContextItem(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    source: str  # "faq" | "handbook" | "none"
    contexts: List[ContextItem] = Field(default_factory=list)

# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # tampilkan UI chat
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = chatbot.get_answer(req.message)

    contexts_raw = result.get("contexts", [])
    contexts = [
        ContextItem(
            id=str(c.get("id", "")),
            score=float(c.get("score", 0.0)),
            metadata=c.get("metadata", {}) or {},
        )
        for c in contexts_raw
        if isinstance(c, dict)
    ]

    return ChatResponse(
        answer=result.get("answer", ""),
        confidence=float(result.get("confidence", 0.0)),
        source=str(result.get("source", "none")),
        contexts=contexts,
    )

@app.get("/health")
def health():
    return {"status": "ok"}
