from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, ClassVar
from datetime import datetime, date
from enum import Enum

class DocumentStatus(str, Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    DETECTED = "detected"  # Ensemble detection complete
    OCR_COMPLETE = "ocr_complete"  # OCR extraction complete
    PARSED = "parsed"  # Data parsed and structured
    VALIDATED = "validated"
    FAILED = "failed"

class DocumentType(str, Enum):
    """Types of medical documents"""
    MEDICAL_BILL = "medical_bill"
    PRESCRIPTION = "prescription"
    DIAGNOSTIC_REPORT = "diagnostic_report"
    OTHER = "other"


class MedicalBill(SQLModel, table=True):
    """Main table for medical bill documents with 8 detected classes"""
    __tablename__: ClassVar[str] = "medical_bill"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # File Information
    filename: str = Field(index=True)
    original_path: str
    file_size: int
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_timestamp: Optional[datetime] = None
    status: DocumentStatus = DocumentStatus.UPLOADED
    
    # Ensemble Detection Metadata
    ensemble_model_used: Optional[str] = None  # "Faster-RCNN + YOLO11n WBF"
    detection_confidence: Optional[float] = None  # Average confidence across all detections
    
    # 1. Date of Receipt
    date_of_receipt: Optional[date] = None
    date_of_receipt_raw: Optional[str] = None  # Raw OCR text
    date_of_receipt_confidence: Optional[float] = None
    date_of_receipt_bbox: Optional[str] = None  # JSON: {"x1": 0, "y1": 0, "x2": 100, "y2": 50}
    
    # 2. GSTIN
    gstin: Optional[str] = Field(default=None, max_length=15)
    gstin_raw: Optional[str] = None
    gstin_confidence: Optional[float] = None
    gstin_bbox: Optional[str] = None
    
    # 3. Invoice Number
    invoice_no: Optional[str] = None
    invoice_no_raw: Optional[str] = None
    invoice_no_confidence: Optional[float] = None
    invoice_no_bbox: Optional[str] = None
    
    # 4. Mobile Number
    mobile_no: Optional[str] = Field(default=None, max_length=15)
    mobile_no_raw: Optional[str] = None
    mobile_no_confidence: Optional[float] = None
    mobile_no_bbox: Optional[str] = None
    
    # 5. Product Table (Foreign Key - One-to-Many relationship)
    product_table_bbox: Optional[str] = None  # Bounding box of entire product table
    product_table_confidence: Optional[float] = None
    
    # 6. Store Address
    store_address: Optional[str] = None
    store_address_raw: Optional[str] = None
    store_address_confidence: Optional[float] = None
    store_address_bbox: Optional[str] = None
    
    # 7. Store Name
    store_name: Optional[str] = None
    store_name_raw: Optional[str] = None
    store_name_confidence: Optional[float] = None
    store_name_bbox: Optional[str] = None
    
    # 8. Total Amount
    total_amount: Optional[float] = None
    total_amount_raw: Optional[str] = None
    total_amount_confidence: Optional[float] = None
    total_amount_bbox: Optional[str] = None
    
    # OCR Metadata
    ocr_engine_used: Optional[str] = None  # "paddleocr", "easyocr", "tesseract"
    full_ocr_text: Optional[str] = None  # Complete OCR text for reference
    
    # Validation & Compliance
    is_validated: bool = False
    validation_notes: Optional[str] = None
    
    # Relationships
    product_items: List["ProductItem"] = Relationship(back_populates="medical_bill")
    processing_jobs: List["ProcessingJob"] = Relationship(back_populates="medical_bill")


class ProductItem(SQLModel, table=True):
    """Separate table for products detected in the product_table region"""
    __tablename__: ClassVar[str] = "product_item"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    medical_bill_id: int = Field(foreign_key="medical_bill.id", index=True)
    
    # Product Information (6 columns as specified)
    product: Optional[str] = None  # Product/Medicine name
    quantity: Optional[str] = None  # Quantity
    pack: Optional[str] = None  # Pack size/type
    mrp: Optional[float] = None  # Maximum Retail Price
    expiry: Optional[str] = None  # Expiry date (can be date or string)
    total_amount: Optional[float] = None  # Total amount for this product
    
    # OCR Metadata for each field
    product_confidence: Optional[float] = None
    quantity_confidence: Optional[float] = None
    pack_confidence: Optional[float] = None
    mrp_confidence: Optional[float] = None
    expiry_confidence: Optional[float] = None
    total_amount_confidence: Optional[float] = None
    
    # Bounding box for this product row
    bbox: Optional[str] = None  # JSON: {"x1": 0, "y1": 0, "x2": 100, "y2": 50}
    
    # Row order in table
    row_index: Optional[int] = None  # To maintain order of products
    
    # Raw OCR text for entire row
    raw_text: Optional[str] = None
    
    # Validation
    is_validated: bool = False
    validation_notes: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    medical_bill: MedicalBill = Relationship(back_populates="product_items")


class ProcessingJob(SQLModel, table=True):
    """Track processing jobs for async operations"""
    __tablename__: ClassVar[str] = "processing_job"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    medical_bill_id: int = Field(foreign_key="medical_bill.id", index=True)
    
    job_type: str  # "ensemble_detection", "ocr_extraction", "data_parsing", "validation"
    status: DocumentStatus
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    error_message: Optional[str] = None
    result_data: Optional[str] = None  # JSON string with job results
    
    # Relationships
    medical_bill: MedicalBill = Relationship(back_populates="processing_jobs")


class MedicineDatabase(SQLModel, table=True):
    """Reference medicine database for validation"""
    __tablename__: ClassVar[str] = "medicine_database"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    medicine_name: str = Field(index=True)
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    dosage_form: Optional[str] = None  # tablet, syrup, injection
    strength: Optional[str] = None  # 500mg, 10ml
    
    mrp: Optional[float] = None
    manufacturer: Optional[str] = None
    
    # Search optimization
    search_keywords: Optional[str] = None  # Space-separated keywords for fuzzy search
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
