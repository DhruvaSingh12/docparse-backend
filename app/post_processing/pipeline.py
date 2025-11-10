import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from sqlmodel import Session

from app.models import MedicalBill, ProductItem
from .correctors import get_field_corrector
from .table_extractor import get_table_transformer
from .llm_corrector import get_medgemma_corrector
from app.services.service import get_ocr_service

class PostProcessingPipeline:
    def __init__(self):
        self.table_transformer = get_table_transformer()
        self.ocr_service = get_ocr_service()
        self.llm_corrector = get_medgemma_corrector()
    
    async def process_medical_bill(
        self,
        medical_bill: MedicalBill,
        ocr_results: Dict[str, Any],
        image_path: str,
        session: Session
    ) -> Dict[str, Any]:
        print("\n=== Starting Post-Processing Pipeline ===")
        
        results = {
            'fields_corrected': 0,
            'fields_validated': 0,
            'llm_corrections': 0,
            'product_items_extracted': 0,
            'errors': []
        }
        
        try:
            return await self._process_internal(medical_bill, ocr_results, image_path, session, results)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"\n✗ Post-processing failed with error: {str(e)}")
            print(f"Traceback:\n{error_trace}")
            results['errors'].append({
                'field': 'pipeline',
                'errors': [f'Pipeline error: {str(e)}']
            })
            return results
    
    async def _process_internal(
        self,
        medical_bill: MedicalBill,
        ocr_results: Dict[str, Any],
        image_path: str,
        session: Session,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        # Step 1: Process each field
        for field_name, ocr_data in ocr_results.items():
            if field_name == 'product_table':
                continue  # Handle separately
            
            if not ocr_data or ocr_data.get('skipped'):
                continue
            
            raw_text = ocr_data.get('text')
            ocr_confidence = ocr_data.get('confidence', 0.0)
            bbox = ocr_data.get('bbox')
            
            if not raw_text:
                continue
            
            print(f"\nProcessing {field_name}...")
            print(f"  Raw OCR: '{raw_text}' (confidence: {ocr_confidence:.2f})")
            
            # Get appropriate corrector
            corrector = get_field_corrector(field_name)
            correction_result = corrector.correct(raw_text, ocr_confidence, bbox)
            
            corrected_value = correction_result['corrected_value']
            final_confidence = correction_result['confidence']
            method = correction_result['correction_method']
            
            print(f"  Corrected: '{corrected_value}' (confidence: {final_confidence:.2f}, method: {method})")
            
            if correction_result['errors']:
                print(f"  ⚠ Errors: {', '.join(correction_result['errors'])}")
                results['errors'].append({
                    'field': field_name,
                    'errors': correction_result['errors']
                })
            
            # Update medical_bill object
            try:
                self._update_field(medical_bill, field_name, corrected_value, final_confidence)
            except Exception as e:
                error_msg = f"Failed to update {field_name}: {str(e)}"
                print(f"  ✗ {error_msg}")
                results['errors'].append({
                    'field': field_name,
                    'errors': [error_msg]
                })
                continue
            
            results['fields_corrected'] += 1
            if method == 'llm':
                results['llm_corrections'] += 1
            
            results['fields_validated'] += 1
        
        # Step 2: Process product table if detected
        if 'product_table' in ocr_results and not ocr_results['product_table'].get('skipped'):
            print("\n=== Processing Product Table ===")
            
            product_table_bbox = ocr_results['product_table'].get('bbox')
            
            if product_table_bbox and medical_bill.id is not None:
                try:
                    product_items = await self._extract_product_table(
                        image_path,
                        product_table_bbox,
                        medical_bill.id,
                        session
                    )
                    
                    results['product_items_extracted'] = len(product_items)
                    print(f"✓ Extracted {len(product_items)} product items")
                    
                except Exception as e:
                    error_msg = f"Product table extraction failed: {str(e)}"
                    print(f"✗ {error_msg}")
                    results['errors'].append({
                        'field': 'product_table',
                        'errors': [error_msg]
                    })
        
        # Save changes (with error handling for connection issues)
        try:
            session.add(medical_bill)
            session.commit()
        except Exception as e:
            print(f"  ⚠ Database commit failed, attempting rollback and retry: {e}")
            session.rollback()
            # Refresh the session and try again
            session.refresh(medical_bill)
            session.add(medical_bill)
            session.commit()
        
        print("\n=== Post-Processing Complete ===")
        print(f"Fields corrected: {results['fields_corrected']}")
        print(f"LLM corrections: {results['llm_corrections']}")
        print(f"Product items: {results['product_items_extracted']}")
        
        return results
    
    def _update_field(self, medical_bill: MedicalBill, field_name: str, value: Any, confidence: float):
        """Update medical_bill field with corrected value"""
        
        # Field name mapping (handle both correct and typo spellings)
        field_mapping = {
            'gstin': 'gstin',
            'mobile_no': 'mobile_no',
            'date_of_receipt': 'date_of_receipt',
            'date_of_reciept': 'date_of_receipt',  # Handle typo
            'invoice_no': 'invoice_no',
            'total_amount': 'total_amount',
            'store_name': 'store_name',
            'store_address': 'store_address',
        }
        
        db_field = field_mapping.get(field_name)
        if not db_field:
            return
        
        # Skip if value is None (don't overwrite with None)
        if value is None:
            print(f"  ⚠ Skipping update for {field_name} (value is None)")
            return
        
        # Special handling for date fields
        if db_field == 'date_of_receipt':
            # For date fields, we need to parse the string to a date object
            from datetime import datetime
            if isinstance(value, str) and value:
                try:
                    # Parse YYYY-MM-DD format
                    parsed_date = datetime.strptime(value, '%Y-%m-%d').date()
                    setattr(medical_bill, db_field, parsed_date)
                except ValueError:
                    # If parsing fails, leave as None
                    print(f"  ⚠ Could not parse date: {value}")
                    setattr(medical_bill, db_field, None)
            elif value is None:
                setattr(medical_bill, db_field, None)
        else:
            # Update main value for other fields
            setattr(medical_bill, db_field, value)
        
        # Update confidence
        confidence_field = f"{db_field}_confidence"
        if hasattr(medical_bill, confidence_field):
            # Combine OCR confidence with validation confidence
            setattr(medical_bill, confidence_field, confidence)
    
    async def _extract_product_table(
        self,
        image_path: str,
        table_bbox_json: str,
        medical_bill_id: int,
        session: Session
    ) -> List[ProductItem]:
        import json
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Parse bbox
        table_bbox_dict = json.loads(table_bbox_json)
        table_bbox = [
            int(table_bbox_dict['x1']),
            int(table_bbox_dict['y1']),
            int(table_bbox_dict['x2']),
            int(table_bbox_dict['y2'])
        ]
        
        # Use new LLM-based extraction method
        print("  Using intelligent LLM-based table extraction...")
        products = await self.table_transformer.extract_table_with_llm(
            image,
            table_bbox,
            self.ocr_service,
            self.llm_corrector
        )
        
        if not products:
            print("  ⚠ No products extracted from table")
            return []
        
        # Convert to ProductItem objects and save to database
        product_items = []
        
        for idx, product_data in enumerate(products, start=1):
            try:
                # Parse amounts
                mrp = self._parse_amount(product_data.get('mrp'))
                total_amount = self._parse_amount(product_data.get('total_amount'))
                
                # Clean up text fields
                product_name = product_data.get('product', '').strip() if product_data.get('product') else None
                quantity = product_data.get('quantity', '').strip() if product_data.get('quantity') else None
                pack = product_data.get('pack', '').strip() if product_data.get('pack') else None
                expiry = product_data.get('expiry', '').strip() if product_data.get('expiry') else None
                
                # Skip if no product name
                if not product_name:
                    continue
                
                product_item = ProductItem(
                    medical_bill_id=medical_bill_id,
                    product=product_name,
                    quantity=quantity,
                    pack=pack,
                    mrp=mrp,
                    expiry=expiry,
                    total_amount=total_amount,
                    row_index=idx
                )
                
                session.add(product_item)
                product_items.append(product_item)
                
                print(f"  ✓ Product {idx}: {product_name} | Qty: {quantity} | Total: {total_amount}")
                
            except Exception as e:
                print(f"  ⚠ Failed to process product {idx}: {e}")
                continue
        
        # Commit all product items
        if product_items:
            session.commit()
            print(f"\n  ✓ Saved {len(product_items)} products to database")
        
        return product_items
    
    def _parse_amount(self, text: Optional[str]) -> Optional[float]:
        """Parse amount from text"""
        if not text:
            return None
        
        try:
            import re
            # Remove non-numeric characters except decimal point
            cleaned = re.sub(r'[^\d.]', '', text)
            return float(cleaned)
        except:
            return None


# Singleton instance
_post_processing_pipeline = None

def get_post_processing_pipeline() -> PostProcessingPipeline:
    """Get or create post-processing pipeline instance"""
    global _post_processing_pipeline
    if _post_processing_pipeline is None:
        _post_processing_pipeline = PostProcessingPipeline()
    return _post_processing_pipeline
