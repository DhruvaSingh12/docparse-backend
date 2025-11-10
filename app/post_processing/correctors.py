from typing import Dict, Any, Optional
from .validators import (
    GST_Validator,
    MobileNumberValidator,
    DateValidator,
    InvoiceNumberValidator,
    AmountValidator
)
from .llm_corrector import get_medgemma_corrector

class FieldCorrector:  
    def __init__(self):
        self.llm_corrector = get_medgemma_corrector()
        self.llm_available = self.llm_corrector.is_available()
        
        if self.llm_available:
            print("✓ MedGemma LLM available for corrections")
        else:
            print("⚠ MedGemma LLM not available, using rule-based corrections only")
    
    def correct(self, raw_text: str, ocr_confidence: float, bbox: Optional[Dict] = None) -> Dict[str, Any]:
        raise NotImplementedError

class GSTINCorrector(FieldCorrector):   
    def correct(self, raw_text: str, ocr_confidence: float, bbox: Optional[Dict] = None) -> Dict[str, Any]:
        
        # Step 1: Validate with rule-based validator
        validation = GST_Validator.validate(raw_text)
        
        # If valid, return immediately
        if validation['is_valid']:
            return {
                'corrected_value': validation['corrected_value'],
                'confidence': min(validation['confidence'], ocr_confidence),
                'validation_result': validation,
                'correction_method': 'validator',
                'errors': []
            }
        
        # Step 2: Try LLM correction if needed and available
        if validation['needs_llm_correction'] and self.llm_available:
            print(f"  → Attempting LLM correction for GSTIN...")
            llm_result = self.llm_corrector.correct_gstin(raw_text, bbox)
            
            if llm_result['success']:
                # Validate LLM output
                llm_validation = GST_Validator.validate(llm_result['corrected_value'])
                
                if llm_validation['is_valid']:
                    return {
                        'corrected_value': llm_validation['corrected_value'],
                        'confidence': llm_result['confidence'] * 0.9,  # Slight penalty for LLM correction
                        'validation_result': llm_validation,
                        'correction_method': 'llm',
                        'errors': []
                    }
        
        # Step 3: Return best effort (original validation result)
        return {
            'corrected_value': validation.get('corrected_value'),
            'confidence': validation['confidence'] * ocr_confidence,
            'validation_result': validation,
            'correction_method': 'validator_failed',
            'errors': validation['errors']
        }

class MobileNumberCorrector(FieldCorrector):
    
    def correct(self, raw_text: str, ocr_confidence: float, bbox: Optional[Dict] = None) -> Dict[str, Any]:
        
        # Step 1: Validate
        validation = MobileNumberValidator.validate(raw_text)
        
        if validation['is_valid']:
            return {
                'corrected_value': validation['corrected_value'],
                'confidence': min(validation['confidence'], ocr_confidence),
                'validation_result': validation,
                'correction_method': 'validator',
                'errors': []
            }
        
        # Step 2: Try LLM correction if needed
        if validation['needs_llm_correction'] and self.llm_available:
            print(f"  → Attempting LLM correction for mobile number...")
            llm_result = self.llm_corrector.correct_mobile_number(raw_text)
            
            if llm_result['success']:
                llm_validation = MobileNumberValidator.validate(llm_result['corrected_value'])
                
                if llm_validation['is_valid']:
                    return {
                        'corrected_value': llm_validation['corrected_value'],
                        'confidence': llm_result['confidence'] * 0.9,
                        'validation_result': llm_validation,
                        'correction_method': 'llm',
                        'errors': []
                    }
        
        return {
            'corrected_value': validation.get('corrected_value'),
            'confidence': validation['confidence'] * ocr_confidence,
            'validation_result': validation,
            'correction_method': 'validator_failed',
            'errors': validation['errors']
        }

class DateCorrector(FieldCorrector):    
    def correct(self, raw_text: str, ocr_confidence: float, bbox: Optional[Dict] = None) -> Dict[str, Any]:
        
        validation = DateValidator.validate(raw_text)
        
        if validation['is_valid']:
            return {
                'corrected_value': validation['corrected_value'],
                'date_object': validation['date_object'],
                'confidence': min(validation['confidence'], ocr_confidence),
                'validation_result': validation,
                'correction_method': 'validator',
                'errors': []
            }
        
        # For dates, LLM might not be as reliable as rule-based parsing
        return {
            'corrected_value': validation.get('corrected_value'),
            'date_object': validation.get('date_object'),
            'confidence': validation['confidence'] * ocr_confidence,
            'validation_result': validation,
            'correction_method': 'validator_failed',
            'errors': validation['errors']
        }

class InvoiceNumberCorrector(FieldCorrector):
    
    def correct(self, raw_text: str, ocr_confidence: float, bbox: Optional[Dict] = None) -> Dict[str, Any]:
        
        validation = InvoiceNumberValidator.validate(raw_text)
        
        return {
            'corrected_value': validation.get('corrected_value'),
            'confidence': min(validation['confidence'], ocr_confidence),
            'validation_result': validation,
            'correction_method': 'validator',
            'errors': validation['errors']
        }

class AmountCorrector(FieldCorrector):
    
    def correct(self, raw_text: str, ocr_confidence: float, bbox: Optional[Dict] = None) -> Dict[str, Any]:
        
        validation = AmountValidator.validate(raw_text)
        
        return {
            'corrected_value': validation.get('corrected_value'),
            'confidence': min(validation['confidence'], ocr_confidence),
            'validation_result': validation,
            'correction_method': 'validator',
            'errors': validation['errors']
        }

class StoreNameCorrector(FieldCorrector):
    
    def correct(self, raw_text: str, ocr_confidence: float, bbox: Optional[Dict] = None) -> Dict[str, Any]:
        
        # Basic cleaning first
        cleaned = raw_text.strip()
        
        # Always try LLM correction if available (for formatting, spelling, completion)
        if self.llm_available:
            print(f"  → Using LLM to format and correct store name...")
            llm_result = self.llm_corrector.correct_store_name(raw_text)
            
            if llm_result['success']:
                return {
                    'corrected_value': llm_result['corrected_value'],
                    'confidence': min(llm_result['confidence'], ocr_confidence),
                    'correction_method': 'llm',
                    'errors': []
                }
        
        # Fallback to basic cleaning if LLM not available or failed
        return {
            'corrected_value': cleaned,
            'confidence': ocr_confidence,
            'correction_method': 'basic_cleaning',
            'errors': []
        }

class StoreAddressCorrector(FieldCorrector):
    def correct(self, raw_text: str, ocr_confidence: float, bbox: Optional[Dict] = None) -> Dict[str, Any]:
        # Basic cleaning first
        cleaned = raw_text.strip()
        
        # Always try LLM correction if available (for formatting, spacing, commas)
        if self.llm_available:
            print(f"  → Using LLM to format and correct address...")
            llm_result = self.llm_corrector.correct_store_address(raw_text)
            
            if llm_result['success']:
                return {
                    'corrected_value': llm_result['corrected_value'],
                    'confidence': min(llm_result['confidence'], ocr_confidence),
                    'correction_method': 'llm',
                    'errors': []
                }
        
        # Fallback to basic cleaning if LLM not available or failed
        return {
            'corrected_value': cleaned,
            'confidence': ocr_confidence,
            'correction_method': 'basic_cleaning',
            'errors': []
        }

def get_field_corrector(field_name: str) -> FieldCorrector:
    correctors = {
        'gstin': GSTINCorrector,
        'mobile_no': MobileNumberCorrector,
        'date_of_receipt': DateCorrector,
        'date_of_reciept': DateCorrector, 
        'invoice_no': InvoiceNumberCorrector,
        'total_amount': AmountCorrector,
        'store_name': StoreNameCorrector,
        'store_address': StoreAddressCorrector,
    }
    
    corrector_class = correctors.get(field_name, FieldCorrector)
    return corrector_class()