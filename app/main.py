from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from dotenv import load_dotenv
import os

load_dotenv()

from app.database.connections import create_tables
create_tables()

app = FastAPI(
    title = "Triage AI",
    description = "AI powered support ticket automation using Langgraph and RAG",
    version = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=['*'],
    allow_headers=['*']
)

app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "ok", "app": "Triage AI"}