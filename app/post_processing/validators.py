import re
from typing import Optional, Dict, Any
from datetime import datetime

class GST_Validator:
    """
    GSTIN Format: 15 characters
    Pattern: 22AAAAA0000A1Z5
    - First 2 chars: State code (01-37)
    - Next 10 chars: PAN number
    - 13th char: Entity number
    - 14th char: 'Z' by default
    - 15th char: Checksum digit
    """
    
    @staticmethod
    def validate(text: Optional[str]) -> Dict[str, Any]:
        if not text:
            return {
                'is_valid': False,
                'corrected_value': None,
                'confidence': 0.0,
                'errors': ['No GSTIN provided']
            }
        
        # Clean text: remove spaces, convert to uppercase
        cleaned = re.sub(r'\s+', '', text.upper())
        errors = []
        confidence = 1.0
        
        # Check length
        if len(cleaned) != 15:
            errors.append(f'Invalid length: {len(cleaned)} (expected 15)')
            confidence *= 0.3
            
            # Attempt correction if close
            if len(cleaned) == 12:
                # Might be missing state code + checksum
                errors.append('Attempting to correct short GSTIN')
                confidence *= 0.5
        
        # Validate format using regex
        gstin_pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$'
        if not re.match(gstin_pattern, cleaned):
            errors.append('GSTIN format does not match standard pattern')
            confidence *= 0.5
        
        # Validate state code (01-37)
        if len(cleaned) >= 2:
            state_code = cleaned[:2]
            if not state_code.isdigit() or not (1 <= int(state_code) <= 37):
                errors.append(f'Invalid state code: {state_code}')
                confidence *= 0.7
        
        is_valid = len(errors) == 0 and len(cleaned) == 15
        
        return {
            'is_valid': is_valid,
            'corrected_value': cleaned if is_valid else None,
            'confidence': confidence,
            'errors': errors,
            'original': text,
            'needs_llm_correction': not is_valid and confidence < 0.5
        }


class MobileNumberValidator:    
    @staticmethod
    def validate(text: Optional[str]) -> Dict[str, Any]:
        """Validate and correct mobile number"""
        if not text:
            return {
                'is_valid': False,
                'corrected_value': None,
                'confidence': 0.0,
                'errors': ['No mobile number provided']
            }
        
        # Extract only digits
        all_digits = re.sub(r'\D', '', text)
        errors = []
        confidence = 1.0
        
        if len(all_digits) > 12:
            matches = re.findall(r'\d{10}', text)
            if matches:
                # Take the first valid-looking number (starts with 6-9)
                for match in matches:
                    if match[0] in '6789':
                        all_digits = match
                        errors.append('Multiple numbers detected, using first valid one')
                        confidence *= 0.9
                        break
            else:
                # Just take first 10 digits
                all_digits = all_digits[:10]
                errors.append('Truncated to first 10 digits')
                confidence *= 0.8
        
        digits = all_digits
        
        # Remove country code if present
        if len(digits) == 12 and digits.startswith('91'):
            digits = digits[2:]
            errors.append('Removed country code +91')
        elif len(digits) == 11 and digits.startswith('0'):
            digits = digits[1:]
            errors.append('Removed leading 0')
        
        # Check length
        if len(digits) != 10:
            errors.append(f'Invalid length: {len(digits)} (expected 10)')
            confidence *= 0.3
        
        # Check first digit (must be 6-9)
        if digits and digits[0] not in '6789':
            errors.append(f'Invalid first digit: {digits[0]} (must be 6-9)')
            confidence *= 0.5
        
        is_valid = len(digits) == 10 and digits[0] in '6789'
        
        return {
            'is_valid': is_valid,
            'corrected_value': digits if len(digits) == 10 else None,
            'confidence': confidence,
            'errors': errors,
            'original': text,
            'needs_llm_correction': not is_valid and len(digits) not in [10, 0]
        }


class DateValidator:
    DATE_FORMATS = [
        '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',  # DD/MM/YYYY variants
        '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',  # YYYY-MM-DD variants
        '%d %b %Y', '%d %B %Y',               # DD Mon YYYY
        '%b %d %Y', '%B %d %Y',               # Mon DD YYYY
        '%d-%m-%y', '%d/%m/%y',               # DD/MM/YY
    ]
    
    @staticmethod
    def validate(text: Optional[str]) -> Dict[str, Any]:
        """Validate and standardize date"""
        if not text:
            return {
                'is_valid': False,
                'corrected_value': None,
                'confidence': 0.0,
                'errors': ['No date provided']
            }
        
        errors = []
        confidence = 1.0
        cleaned = text.strip()
        
        # Try to parse with different formats
        parsed_date = None
        matched_format = None
        
        for fmt in DateValidator.DATE_FORMATS:
            try:
                parsed_date = datetime.strptime(cleaned, fmt)
                matched_format = fmt
                break
            except ValueError:
                continue
        
        if not parsed_date:
            errors.append(f'Could not parse date format: {cleaned}')
            confidence = 0.0
        else:
            # Validate date is reasonable (not in future, not too old)
            now = datetime.now()
            if parsed_date > now:
                errors.append('Date is in the future')
                confidence *= 0.3
            elif (now - parsed_date).days > 3650:  # More than 10 years old
                errors.append('Date is more than 10 years old')
                confidence *= 0.7
        
        is_valid = parsed_date is not None and len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'corrected_value': parsed_date.strftime('%Y-%m-%d') if parsed_date else None,
            'date_object': parsed_date,
            'confidence': confidence,
            'errors': errors,
            'original': text,
            'matched_format': matched_format,
            'needs_llm_correction': not is_valid
        }


class InvoiceNumberValidator:    
    @staticmethod
    def validate(text: Optional[str]) -> Dict[str, Any]:
        if not text:
            return {
                'is_valid': False,
                'corrected_value': None,
                'confidence': 0.0,
                'errors': ['No invoice number provided']
            }
        
        cleaned = text.strip()
        errors = []
        confidence = 1.0
        
        # Check minimum length
        if len(cleaned) < 3:
            errors.append(f'Invoice number too short: {len(cleaned)} characters')
            confidence *= 0.5
        
        # Check for valid characters (alphanumeric + common separators)
        if not re.match(r'^[A-Za-z0-9\-/._]+$', cleaned):
            errors.append('Invoice number contains invalid characters')
            confidence *= 0.7
        
        # Should have at least some digits
        if not any(c.isdigit() for c in cleaned):
            errors.append('Invoice number should contain at least one digit')
            confidence *= 0.6
        
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'corrected_value': cleaned.upper() if is_valid else None,
            'confidence': confidence,
            'errors': errors,
            'original': text,
            'needs_llm_correction': False  # Usually not needed
        }


class AmountValidator:
    @staticmethod
    def validate(text: Optional[str]) -> Dict[str, Any]:
        """Validate and extract amount"""
        if not text:
            return {
                'is_valid': False,
                'corrected_value': None,
                'confidence': 0.0,
                'errors': ['No amount provided']
            }
        
        errors = []
        confidence = 1.0
        
        # Rs. needs special handling - remove "Rs" but keep the dot if it's a decimal
        cleaned = text.strip()
        # Remove currency symbols
        cleaned = re.sub(r'[₹$]|Rs\.?|INR', '', cleaned)
        # Remove extra spaces
        cleaned = re.sub(r'\s+', '', cleaned)
        
        # Replace comma with dot if it's used as decimal separator
        # Heuristic: if only one comma and it's followed by 2 digits, it's decimal
        if cleaned.count(',') == 1 and re.search(r',\d{2}$', cleaned):
            cleaned = cleaned.replace(',', '.')
        else:
            # Remove thousands separators
            cleaned = cleaned.replace(',', '')
        
        # Try to extract numeric value
        try:
            amount = float(cleaned)
            
            # Validate reasonable range
            if amount < 0:
                errors.append('Amount is negative')
                confidence *= 0.3
            elif amount == 0:
                errors.append('Amount is zero')
                confidence *= 0.5
            elif amount > 1000000:  # > 10 lakhs
                errors.append('Amount seems unusually high')
                confidence *= 0.7
            
            # Round to 2 decimal places
            amount = round(amount, 2)
            
        except ValueError:
            errors.append(f'Could not parse amount: {cleaned}')
            amount = None
            confidence = 0.0
        
        is_valid = amount is not None and len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'corrected_value': amount,
            'confidence': confidence,
            'errors': errors,
            'original': text,
            'needs_llm_correction': False
        }