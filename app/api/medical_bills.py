from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Query
from sqlmodel import Session, select
import os
import tempfile
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.db import get_session
from app.models import MedicalBill, ProductItem, ProcessingJob, DocumentStatus
from app.services.medical_bill_service import get_medical_bill_processor

router = APIRouter()

# File upload configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg"]


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes"
        )
    
    extension = file.filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )


@router.post("/process", response_model=Dict[str, Any])
async def process_medical_bill(
    file: UploadFile = File(..., description="Medical bill image (PNG, JPG, JPEG)"),
    use_tesseract_fallback: bool = Form(True, description="Use Tesseract as fallback if PaddleOCR confidence is low"),
    session: Session = Depends(get_session)
) -> Dict[str, Any]:

    try:
        # Validate file
        validate_file(file)
        
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Create temporary file
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Get processor and run pipeline
            processor = get_medical_bill_processor()
            result = await processor.process_medical_bill(
                image_path=temp_path,
                filename=file.filename,
                file_size=len(content),
                session=session,
                use_tesseract_fallback=use_tesseract_fallback
            )
            
            return {
                "success": True,
                "message": "Medical bill processed successfully",
                "data": result
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@router.get("/{medical_bill_id}", response_model=Dict[str, Any])
async def get_medical_bill(
    medical_bill_id: int,
    session: Session = Depends(get_session)
) -> Dict[str, Any]:
    """Get medical bill by ID with all extracted fields"""
    medical_bill = session.get(MedicalBill, medical_bill_id)
    
    if not medical_bill:
        raise HTTPException(status_code=404, detail="Medical bill not found")
    
    # Get product items
    product_items = session.exec(
        select(ProductItem).where(ProductItem.medical_bill_id == medical_bill_id)
    ).all()
    
    return {
        "id": medical_bill.id,
        "filename": medical_bill.filename,
        "status": medical_bill.status,
        "upload_timestamp": medical_bill.upload_timestamp,
        "processing_timestamp": medical_bill.processing_timestamp,
        "ensemble_model_used": medical_bill.ensemble_model_used,
        "detection_confidence": medical_bill.detection_confidence,
        "fields": {
            "date_of_receipt": {
                "value": medical_bill.date_of_receipt,
                "raw": medical_bill.date_of_receipt_raw,
                "confidence": medical_bill.date_of_receipt_confidence,
                "bbox": medical_bill.date_of_receipt_bbox
            },
            "gstin": {
                "value": medical_bill.gstin,
                "raw": medical_bill.gstin_raw,
                "confidence": medical_bill.gstin_confidence,
                "bbox": medical_bill.gstin_bbox
            },
            "invoice_no": {
                "value": medical_bill.invoice_no,
                "raw": medical_bill.invoice_no_raw,
                "confidence": medical_bill.invoice_no_confidence,
                "bbox": medical_bill.invoice_no_bbox
            },
            "mobile_no": {
                "value": medical_bill.mobile_no,
                "raw": medical_bill.mobile_no_raw,
                "confidence": medical_bill.mobile_no_confidence,
                "bbox": medical_bill.mobile_no_bbox
            },
            "store_address": {
                "value": medical_bill.store_address,
                "raw": medical_bill.store_address_raw,
                "confidence": medical_bill.store_address_confidence,
                "bbox": medical_bill.store_address_bbox
            },
            "store_name": {
                "value": medical_bill.store_name,
                "raw": medical_bill.store_name_raw,
                "confidence": medical_bill.store_name_confidence,
                "bbox": medical_bill.store_name_bbox
            },
            "total_amount": {
                "value": medical_bill.total_amount,
                "raw": medical_bill.total_amount_raw,
                "confidence": medical_bill.total_amount_confidence,
                "bbox": medical_bill.total_amount_bbox
            }
        },
        "product_items": [
            {
                "id": item.id,
                "product": item.product,
                "quantity": item.quantity,
                "pack": item.pack,
                "mrp": item.mrp,
                "expiry": item.expiry,
                "total_amount": item.total_amount,
                "row_index": item.row_index
            }
            for item in product_items
        ],
        "is_validated": medical_bill.is_validated,
        "validation_notes": medical_bill.validation_notes
    }


@router.get("/", response_model=Dict[str, Any])
async def list_medical_bills(
    limit: int = 10,
    offset: int = 0,
    status: Optional[DocumentStatus] = None,
    session: Session = Depends(get_session)
) -> Dict[str, Any]:
    """List all medical bills with pagination and optional status filter"""
    query = select(MedicalBill)
    
    if status:
        query = query.where(MedicalBill.status == status)
    
    query = query.offset(offset).limit(limit)
    
    medical_bills = session.exec(query).all()
    
    # Get total count
    count_query = select(MedicalBill)
    if status:
        count_query = count_query.where(MedicalBill.status == status)
    total = len(session.exec(count_query).all())
    
    return {
        "medical_bills": [
            {
                "id": bill.id,
                "filename": bill.filename,
                "status": bill.status,
                "upload_timestamp": bill.upload_timestamp,
                "processing_timestamp": bill.processing_timestamp,
                "detection_confidence": bill.detection_confidence,
                "store_name": bill.store_name,
                "total_amount": bill.total_amount,
                "invoice_no": bill.invoice_no
            }
            for bill in medical_bills
        ],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/{medical_bill_id}/processing_jobs", response_model=Dict[str, Any])
async def get_processing_jobs(
    medical_bill_id: int,
    session: Session = Depends(get_session)
) -> Dict[str, Any]:
    """Get all processing jobs for a medical bill"""
    # Check if medical bill exists
    medical_bill = session.get(MedicalBill, medical_bill_id)
    if not medical_bill:
        raise HTTPException(status_code=404, detail="Medical bill not found")
    
    # Get processing jobs
    jobs = session.exec(
        select(ProcessingJob)
        .where(ProcessingJob.medical_bill_id == medical_bill_id)
    ).all()
    
    return {
        "medical_bill_id": medical_bill_id,
        "jobs": [
            {
                "id": job.id,
                "job_type": job.job_type,
                "status": job.status,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error_message": job.error_message,
                "result_data": job.result_data
            }
            for job in jobs
        ]
    }


@router.delete("/{medical_bill_id}", response_model=Dict[str, Any])
async def delete_medical_bill(
    medical_bill_id: int,
    session: Session = Depends(get_session)
) -> Dict[str, Any]:
    """Delete a medical bill and all related data"""
    medical_bill = session.get(MedicalBill, medical_bill_id)
    
    if not medical_bill:
        raise HTTPException(status_code=404, detail="Medical bill not found")
    
    # Delete related product items (cascade should handle this, but explicit is better)
    product_items = session.exec(
        select(ProductItem).where(ProductItem.medical_bill_id == medical_bill_id)
    ).all()
    for item in product_items:
        session.delete(item)
    
    # Delete processing jobs
    jobs = session.exec(
        select(ProcessingJob).where(ProcessingJob.medical_bill_id == medical_bill_id)
    ).all()
    for job in jobs:
        session.delete(job)
    
    # Delete medical bill
    session.delete(medical_bill)
    session.commit()
    
    return {
        "success": True,
        "message": f"Medical bill {medical_bill_id} deleted successfully"
    }
