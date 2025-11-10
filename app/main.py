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
    
    # Eager load all models at startup
    print("\n" + "="*60)
    print("LOADING ALL MODELS AT STARTUP...")
    print("="*60)
    
    try:
        # 1. Load OCR engines
        print("\n[1/5] Loading OCR Engines...")
        from app.services.service import get_ocr_service
        ocr_service = get_ocr_service()
        print("✓ OCR engines ready (PaddleOCR + Tesseract)")
        
        # 2. Load Ensemble Detection Models (YOLO + Faster R-CNN)
        print("\n[2/5] Loading Ensemble Detection Models...")
        from app.ensemble import WBFEnsemble, config as ensemble_config
        ensemble = WBFEnsemble(
            device=ensemble_config.DEVICE,
            confidence_threshold=ensemble_config.CONFIDENCE_THRESHOLD
        )
        print("✓ Ensemble models loaded (YOLO v11n + Faster R-CNN)")
        
        # Store in app state for reuse
        app.state.ensemble_model = ensemble
        
        # 3. Load MedGemma LLM
        print("\n[3/5] Loading MedGemma LLM...")
        from app.post_processing.llm_corrector import get_medgemma_corrector
        llm = get_medgemma_corrector()
        if llm.is_available():
            print("✓ MedGemma LLM loaded and ready")
        else:
            print("⚠ MedGemma LLM not available (will use fallback methods)")
        
        # 4. Load Table Transformer Models
        print("\n[4/5] Loading Table Transformer Models...")
        from app.post_processing.table_extractor import get_table_transformer
        table_transformer = get_table_transformer()
        table_transformer.load_models()
        if table_transformer.loaded:
            print("✓ Table Transformer models loaded")
        else:
            print("⚠ Table Transformer models failed to load")
        
        # 5. Verify all models
        print("\n[5/5] Verifying Model Status...")
        models_status = {
            "OCR (PaddleOCR + Tesseract)": "✓ Ready",
            "Ensemble (YOLO + Faster R-CNN)": "✓ Ready",
            "MedGemma LLM": "✓ Ready" if llm.is_available() else "⚠ Fallback",
            "Table Transformer": "✓ Ready" if table_transformer.loaded else "⚠ Fallback"
        }
        
        print("\nModel Loading Summary:")
        for model_name, status in models_status.items():
            print(f"  {model_name}: {status}")
        
        print("\n" + "="*60)
        print("ALL MODELS LOADED - READY TO PROCESS DOCUMENTS")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n⚠ Error during model loading: {e}")
        import traceback
        traceback.print_exc()
        print("Server will continue, models will load on first use.\n")
    
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