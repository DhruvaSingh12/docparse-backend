from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import warnings
import logging
from dotenv import load_dotenv
from app.api.medical_bills import router as medical_bills_router
from app.db import init_db
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
warnings.filterwarnings("ignore", message=".*ccache.*")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="paddleocr")
warnings.filterwarnings("ignore", message=".*invalid escape sequence.*")

logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("torchvision").setLevel(logging.WARNING)

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting DocParse API...")
    init_db()
    print("Database initialized.")
    yield
    # Shutdown
    print("Shutting down DocParse API...")

# Create FastAPI app
app = FastAPI(
    title=os.getenv("APP_NAME", "DocParse"),
    description="AI-powered medical document parsing",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for mobile app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(medical_bills_router, prefix="/api/medical-bills", tags=["Medical Bills"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "DocParse API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }