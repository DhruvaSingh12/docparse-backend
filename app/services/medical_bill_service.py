import cv2
import json
import numpy as np
from typing import Dict, Any
from datetime import datetime
from sqlmodel import Session

from app.models import MedicalBill, ProcessingJob, DocumentStatus
from app.ensemble import WBFEnsemble, config as ensemble_config, utils as ensemble_utils
from app.services.ocr_service import get_ocr_service, OCREngine


class MedicalBillProcessor:
    
    def __init__(self):
        self.ensemble_model = None
        self.ocr_service = get_ocr_service()
        self._model_loaded = False
        
    def _load_ensemble_model(self):
        if not self._model_loaded:
            print("Loading ensemble model...")
            self.ensemble_model = WBFEnsemble(
                device=ensemble_config.DEVICE,
                confidence_threshold=ensemble_config.CONFIDENCE_THRESHOLD
            )
            self._model_loaded = True
            print("✓ Ensemble model loaded successfully")
    
    async def process_medical_bill(
        self,
        image_path: str,
        filename: str,
        file_size: int,
        session: Session,
        use_tesseract_fallback: bool = True
    ) -> Dict[str, Any]:
        # Step 1: Create medical bill record
        medical_bill = MedicalBill(
            filename=filename,
            original_path=image_path,
            file_size=file_size,
            status=DocumentStatus.UPLOADED
        )
        session.add(medical_bill)
        session.commit()
        session.refresh(medical_bill)
        
        try:
            # Step 2: Run ensemble detection
            if medical_bill.id is None:
                raise ValueError("Medical bill ID is None")
            
            detection_result = await self._run_ensemble_detection(
                image_path, 
                medical_bill.id,
                session
            )
            
            # Step 3: Run OCR on detected regions
            ocr_result = await self._run_ocr_on_regions(
                image_path,
                detection_result['predictions'],
                medical_bill.id,
                session,
                use_tesseract_fallback
            )
            
            # Step 4: Update medical bill with extracted data
            await self._update_medical_bill(
                medical_bill,
                detection_result,
                ocr_result,
                session
            )
            
            # Step 5: Mark as parsed
            medical_bill.status = DocumentStatus.PARSED
            medical_bill.processing_timestamp = datetime.utcnow()
            session.add(medical_bill)
            session.commit()
            
            # Prepare serializable detection summary (exclude numpy arrays)
            detection_summary = {}
            for class_name, details in detection_result['organized'].items():
                detection_summary[class_name] = {
                    'bbox_json': details['bbox_json'],
                    'confidence': details['confidence'],
                    'class_name': details['class_name'],
                    'db_field': details['db_field']
                }
            
            return {
                "success": True,
                "medical_bill_id": medical_bill.id,
                "filename": filename,
                "status": medical_bill.status,
                "detections": len(detection_result['predictions']['boxes']),
                "detection_summary": detection_summary,
                "ocr_results": ocr_result
            }
            
        except Exception as e:
            # Mark as failed
            medical_bill.status = DocumentStatus.FAILED
            session.add(medical_bill)
            session.commit()
            
            # Log error in processing job
            if medical_bill.id is not None:
                error_job = ProcessingJob(
                    medical_bill_id=medical_bill.id,
                    job_type="full_pipeline",
                    status=DocumentStatus.FAILED,
                    error_message=str(e)
                )
                session.add(error_job)
                session.commit()
            
            raise Exception(f"Medical bill processing failed: {str(e)}")
    
    async def _run_ensemble_detection(
        self,
        image_path: str,
        medical_bill_id: int,
        session: Session
    ) -> Dict[str, Any]:
        # Load ensemble model if not loaded
        self._load_ensemble_model()
        
        # Create processing job
        job = ProcessingJob(
            medical_bill_id=medical_bill_id,
            job_type="ensemble_detection",
            status=DocumentStatus.PROCESSING
        )
        session.add(job)
        session.commit()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Run ensemble detection
            print(f"Running ensemble detection on {image_path}...")
            if self.ensemble_model is None:
                raise RuntimeError("Ensemble model not loaded")
                
            predictions = self.ensemble_model.predict(
                image,
                iou_threshold=ensemble_config.IOU_THRESHOLD,
                skip_box_threshold=ensemble_config.SKIP_BOX_THRESHOLD
            )
            
            # Organize predictions by class
            organized = ensemble_utils.organize_predictions_by_class(predictions)
            
            # Calculate average confidence
            avg_confidence = ensemble_utils.calculate_average_confidence(predictions)
            
            # Mark job as complete
            job.status = DocumentStatus.DETECTED
            job.completed_at = datetime.utcnow()
            job.result_data = json.dumps({
                "num_detections": len(predictions['boxes']),
                "average_confidence": avg_confidence,
                "classes_detected": list(organized.keys())
            })
            session.add(job)
            session.commit()
            
            print(f"✓ Detected {len(predictions['boxes'])} regions")
            
            return {
                "predictions": predictions,
                "organized": organized,
                "average_confidence": avg_confidence
            }
            
        except Exception as e:
            job.status = DocumentStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            session.add(job)
            session.commit()
            raise
    
    async def _run_ocr_on_regions(
        self,
        image_path: str,
        predictions: Dict,
        medical_bill_id: int,
        session: Session,
        use_tesseract_fallback: bool = True
    ) -> Dict[str, Any]:
        # Create processing job
        job = ProcessingJob(
            medical_bill_id=medical_bill_id,
            job_type="ocr_extraction",
            status=DocumentStatus.PROCESSING
        )
        session.add(job)
        session.commit()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            ocr_results = {}
            
            # Process each detected region
            for box, label, score, class_name in zip(
                predictions['boxes'],
                predictions['labels'],
                predictions['scores'],
                predictions['class_names']
            ):
                print(f"Running OCR on {class_name}...")
                
                # Skip product_table for now (as requested)
                if class_name == 'product_table':
                    ocr_results[class_name] = {
                        "text": None,
                        "confidence": float(score),
                        "bbox": ensemble_utils.bbox_to_json(box),
                        "skipped": True,
                        "reason": "Product table processing not implemented yet"
                    }
                    continue
                
                # Extract region from image
                region = ensemble_utils.extract_region_from_image(image, box, padding=5)
                
                # Run PaddleOCR first
                try:
                    ocr_result = await self.ocr_service.process_image(
                        region,
                        engine=OCREngine.PADDLEOCR
                    )
                    
                    ocr_results[class_name] = {
                        "text": ocr_result.get("text", "").strip(),
                        "confidence": ocr_result.get("confidence", 0.0),
                        "detection_confidence": float(score),
                        "bbox": ensemble_utils.bbox_to_json(box),
                        "engine_used": "paddleocr"
                    }
                    
                    # Try Tesseract fallback if PaddleOCR confidence is low
                    if use_tesseract_fallback and ocr_result.get("confidence", 0) < 0.6:
                        print(f"  → PaddleOCR confidence low, trying Tesseract fallback...")
                        tesseract_result = await self.ocr_service.process_image(
                            region,
                            engine=OCREngine.TESSERACT
                        )
                        
                        # Use Tesseract if it has higher confidence
                        if tesseract_result.get("confidence", 0) > ocr_result.get("confidence", 0):
                            ocr_results[class_name]["text"] = tesseract_result.get("text", "").strip()
                            ocr_results[class_name]["confidence"] = tesseract_result.get("confidence", 0.0)
                            ocr_results[class_name]["engine_used"] = "tesseract"
                            print(f"  → Using Tesseract result (better confidence)")
                
                except Exception as ocr_error:
                    print(f"  ⚠ OCR failed for {class_name}: {ocr_error}")
                    ocr_results[class_name] = {
                        "text": None,
                        "confidence": 0.0,
                        "detection_confidence": float(score),
                        "bbox": ensemble_utils.bbox_to_json(box),
                        "error": str(ocr_error),
                        "engine_used": "none"
                    }
            
            # Mark job as complete
            job.status = DocumentStatus.OCR_COMPLETE
            job.completed_at = datetime.utcnow()
            job.result_data = json.dumps({
                "regions_processed": len(ocr_results),
                "successful": sum(1 for r in ocr_results.values() if r.get("text"))
            })
            session.add(job)
            session.commit()
            
            print(f"✓ OCR completed on {len(ocr_results)} regions")
            
            return ocr_results
            
        except Exception as e:
            job.status = DocumentStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            session.add(job)
            session.commit()
            raise
    
    async def _update_medical_bill(
        self,
        medical_bill: MedicalBill,
        detection_result: Dict,
        ocr_result: Dict,
        session: Session
    ):
        # Update ensemble metadata
        medical_bill.ensemble_model_used = "Faster-RCNN + YOLO11n WBF"
        medical_bill.detection_confidence = detection_result['average_confidence']
        medical_bill.ocr_engine_used = "paddleocr_with_tesseract_fallback"
        
        # Update each field based on OCR results
        for class_name, ocr_data in ocr_result.items():
            # Skip if no text extracted
            if not ocr_data.get("text"):
                continue
            
            # Get database field name
            db_field = ensemble_utils.get_db_field_name(class_name)
            
            # Skip product_table (handled separately)
            if class_name == 'product_table':
                medical_bill.product_table_bbox = ocr_data.get("bbox")
                medical_bill.product_table_confidence = ocr_data.get("detection_confidence")
                continue
            
            # Set main value, raw value, confidence, and bbox for each field
            text = ocr_data.get("text", "")
            
            # Date of receipt
            if db_field == 'date_of_receipt':
                medical_bill.date_of_receipt_raw = text
                medical_bill.date_of_receipt_confidence = ocr_data.get("confidence")
                medical_bill.date_of_receipt_bbox = ocr_data.get("bbox")
                # TODO: Parse date string to actual date
            
            # GSTIN
            elif db_field == 'gstin':
                medical_bill.gstin = text
                medical_bill.gstin_raw = text
                medical_bill.gstin_confidence = ocr_data.get("confidence")
                medical_bill.gstin_bbox = ocr_data.get("bbox")
            
            # Invoice Number
            elif db_field == 'invoice_no':
                medical_bill.invoice_no = text
                medical_bill.invoice_no_raw = text
                medical_bill.invoice_no_confidence = ocr_data.get("confidence")
                medical_bill.invoice_no_bbox = ocr_data.get("bbox")
            
            # Mobile Number
            elif db_field == 'mobile_no':
                medical_bill.mobile_no = text
                medical_bill.mobile_no_raw = text
                medical_bill.mobile_no_confidence = ocr_data.get("confidence")
                medical_bill.mobile_no_bbox = ocr_data.get("bbox")
            
            # Store Address
            elif db_field == 'store_address':
                medical_bill.store_address = text
                medical_bill.store_address_raw = text
                medical_bill.store_address_confidence = ocr_data.get("confidence")
                medical_bill.store_address_bbox = ocr_data.get("bbox")
            
            # Store Name
            elif db_field == 'store_name':
                medical_bill.store_name = text
                medical_bill.store_name_raw = text
                medical_bill.store_name_confidence = ocr_data.get("confidence")
                medical_bill.store_name_bbox = ocr_data.get("bbox")
            
            # Total Amount
            elif db_field == 'total_amount':
                medical_bill.total_amount_raw = text
                medical_bill.total_amount_confidence = ocr_data.get("confidence")
                medical_bill.total_amount_bbox = ocr_data.get("bbox")
                # Try to parse as float
                try:
                    # Remove currency symbols and commas
                    clean_text = text.replace('₹', '').replace(',', '').strip()
                    medical_bill.total_amount = float(clean_text)
                except (ValueError, AttributeError):
                    medical_bill.total_amount = None
        
        session.add(medical_bill)
        session.commit()
        print(f"✓ Medical bill {medical_bill.id} updated with extracted data")


# Singleton instance
_processor_instance = None

def get_medical_bill_processor() -> MedicalBillProcessor:
    """Get singleton instance of medical bill processor"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = MedicalBillProcessor()
    return _processor_instance
