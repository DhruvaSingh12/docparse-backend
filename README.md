# DocParse Backend

AI-powered medical bill processing system with OCR, entity extraction, and intelligent table parsing.

## Features

- **Ensemble Detection**: YOLO v11n + Faster R-CNN for 8 field types
- **Dual OCR**: PaddleOCR (primary) + Tesseract (fallback)
- **LLM Corrections**: MedGemma 4B for intelligent formatting
- **Table Extraction**: AI-powered parsing with column mapping
- **Validation Pipeline**: Domain-specific validators for medical bills
- **Database**: PostgreSQL with structured storage

## Quick Start

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (optional, for acceleration)
- Tesseract OCR: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### Installation

```bash
# Clone and navigate
cd docparse/backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo DATABASE_URL=postgresql://user:pass@host:port/db > .env
```

### Running

```bash
# Start server
python -m uvicorn app.main:app --port 8000 --host 127.0.0.1
```
Access at: http://localhost:8000/docs

## API Endpoints

**POST** `/api/medical-bills/process` - Upload and process medical bill
- Accepts: JPG, PNG, PDF
- Returns: Extracted fields + product table

**GET** `/api/medical-bills/{id}` - Get bill details

**GET** `/api/medical-bills/{id}/products` - Get product items

### Example

```bash
curl -X POST http://localhost:8000/api/medical-bills/process \
  -F "file=@bill.jpg"
```

## Architecture

```
Upload → Detection (YOLO+RCNN) → OCR (Paddle+Tess) → Validation → LLM Correction → Database
```

### Models
- **Detection**: YOLOv11n + Faster R-CNN + WBF ensemble
- **OCR**: PaddleOCR (GPU) with Tesseract fallback
- **LLM**: MedGemma 4B IT (4-bit quantized)
- **Table**: Microsoft Table Transformer

### Extracted Fields
- Invoice Number
- Date of Receipt
- Total Amount
- Store Name & Address
- GSTIN
- Mobile Number
- Product Table (product, quantity, pack, mrp, expiry, total_amount)

## Project Structure

```
app/
├── main.py                    # FastAPI app with eager model loading
├── db.py                      # Database session management
├── models.py                  # SQLModel schemas
├── api/
│   └── medical_bills.py      # API routes
├── services/
│   ├── medical_bill_service.py  # Main processing pipeline
│   ├── service.py               # OCR service
│   ├── engines.py               # OCR engines (Paddle/Tesseract)
│   └── preprocess.py            # Image preprocessing
├── ensemble/
│   ├── ensemble.py              # WBF ensemble detection
│   ├── config.py                # Model configurations
│   └── utils.py                 # Detection utilities
└── post_processing/
    ├── pipeline.py              # Post-processing orchestration
    ├── validators.py            # Field validators
    ├── correctors.py            # Field correctors
    ├── llm_corrector.py         # MedGemma integration
    └── table_extractor.py       # Table parsing
```

## Performance

- **Processing Time**: ~15-20 seconds per bill
- **Detection Accuracy**: ~95% (8 field types)
- **OCR Accuracy**: ~95% (printed text)
- **Table Extraction**: ~90% accuracy
- **Model Loading**: One-time at startup (~10-15s)

## Database Schema

**medical_bill**: invoice_no, date, amount, store info, GSTIN, mobile, status  
**product_item**: product details with foreign key to medical_bill  
**processing_job**: job tracking with status and errors

## Configuration

Environment variables in `.env`:
- `DATABASE_URL`: PostgreSQL connection string
- `ENABLE_LLM_CORRECTIONS`: Enable MedGemma (true/false)

## Development

Models are loaded eagerly at startup:
1. OCR engines (PaddleOCR + Tesseract)
2. Ensemble models (YOLO + Faster R-CNN)
3. MedGemma LLM (4-bit quantized)
4. Table Transformer models

First request is instant with no loading delay.