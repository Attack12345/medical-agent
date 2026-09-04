"""FastAPI 入口：REST + SSE + 前端静态托管 + 探针。

启动：uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8090
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.api.auth import router as auth_router  # noqa: E402
from app.api.chat import router as chat_router  # noqa: E402
from app.api.history import router as admin_router  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # medical-agent/

app = FastAPI(title="MedicalConsultationAgent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")


@app.get("/api/v1/health")
def health():
    return {"code": 0, "status": "ok"}


@app.get("/api/v1/ready")
def ready():
    from app.graph.repo import GraphRepo
    from app.retrieval.vector_db import VectorDb

    mysql = db_ready()
    neo4j = _neo4j_ready()
    qdrant = _qdrant_ready()
    return {"code": 0, "mysql": mysql, "neo4j": neo4j, "qdrant": qdrant}


def db_ready() -> bool:
    try:
        from app.services import db
        return db.is_ready()
    except Exception:
        return False


def _neo4j_ready() -> bool:
    try:
        from app.graph.repo import GraphRepo
        return GraphRepo().is_ready()
    except Exception:
        return False


def _qdrant_ready() -> bool:
    try:
        from app.retrieval.vector_db import VectorDb
        return VectorDb().client.collection_exists("entities")
    except Exception as e:  # 探针失败原因打印到 stderr（uvicorn.err）
        print(f"[ready] qdrant probe failed: {e}", flush=True)
        return False


@app.get("/")
def index():
    return RedirectResponse(url="/chat.html")


# 前端静态托管（§9）
_frontend_dir = PROJECT_ROOT / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
