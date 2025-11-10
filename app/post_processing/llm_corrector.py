import os
from typing import Optional, Dict, Any
import torch

class MedGemmaCorrector:
    def __init__(self, model_path: Optional[str] = None, enable_llm: bool = True):
        self.model_path = model_path or "google/medgemma-4b-it"
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False
        self.load_attempted = False  # Track if we've already tried loading
        self.max_new_tokens = 100
        self.enable_llm = enable_llm  # Allow disabling LLM
        
        # Auto-load if enabled
        if self.enable_llm:
            self.load_model()
    
    def load_model(self):
        if self.loaded or self.load_attempted:
            return
        self.load_attempted = True  # Mark that we've tried
        
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
            print(f"Loading MedGemma from {self.model_path}...")
            
            # Configure 4-bit quantization to save GPU memory (exactly like your working code)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load processor (allow download if needed, will cache for future use)
            print("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                use_fast=True
            )
            
            # Load model with quantization (exactly like your working code)
            print("Loading model with 4-bit quantization...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            
            if self.device == "cpu" and self.model is not None:
                self.model = self.model.to(self.device)
            
            if self.model is not None:
                self.model.eval()
            self.loaded = True
            print(f"✓ MedGemma loaded on {self.device}")
            
        except Exception as e:
            print(f"✗ Failed to load MedGemma: {e}")
            print("  LLM corrections will not be available")
            self.loaded = False
    
    def is_available(self) -> bool:
        if not self.loaded:
            self.load_model()
        return self.loaded
    
    def _generate_response(self, prompt: str, max_tokens: int = 50) -> Optional[str]:
        if not self.loaded or self.processor is None or self.model is None:
            return None
        
        try:
            messages = [
                {
                    "role": "system", 
                    "content": [{"type": "text", "text": "You are a helpful AI assistant. Provide concise, accurate responses."}]
                },
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            # Apply chat template
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate with inference mode
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                )
                generation = generation[0][input_len:]
            
            # Decode output
            response = self.processor.decode(generation, skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            print(f"  ✗ LLM generation failed: {e}")
            return None
    
    def correct_gstin(self, raw_text: str, bbox_info: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.is_available():
            return {
                'success': False,
                'corrected_value': None,
                'confidence': 0.0,
                'error': 'MedGemma model not available'
            }
        
        prompt = f"""Correct this OCR text to a valid GSTIN (15 characters):
Input: "{raw_text}"
Rules: 2-digit state code + 10-char PAN + 1 char + Z + checksum
Output only the corrected GSTIN:"""

        try:
            corrected = self._generate_response(prompt, max_tokens=20)
            
            if not corrected:
                return {
                    'success': False,
                    'corrected_value': None,
                    'confidence': 0.0,
                    'error': 'LLM generation failed'
                }
            
            # Extract 15-character GSTIN from response
            corrected = corrected.strip().upper()
            corrected = ''.join(c for c in corrected if c.isalnum())
            
            if len(corrected) == 15:
                return {
                    'success': True,
                    'corrected_value': corrected,
                    'confidence': 0.8,
                    'method': 'medgemma_llm'
                }
            else:
                return {
                    'success': False,
                    'corrected_value': None,
                    'confidence': 0.0,
                    'error': f'Invalid GSTIN length: {len(corrected)}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'corrected_value': None,
                'confidence': 0.0,
                'error': f'LLM correction failed: {str(e)}'
            }
    
    def correct_mobile_number(self, raw_text: str) -> Dict[str, Any]:
        if not self.is_available():
            return {
                'success': False,
                'corrected_value': None,
                'confidence': 0.0,
                'error': 'MedGemma model not available'
            }
        
        prompt = f"""Correct this to a valid 10-digit Indian mobile number:
Input: "{raw_text}"
Rules: Starts with 6/7/8/9, exactly 10 digits
Output only 10 digits:"""

        try:
            corrected = self._generate_response(prompt, max_tokens=15)
            
            if not corrected:
                return {
                    'success': False,
                    'corrected_value': None,
                    'confidence': 0.0,
                    'error': 'LLM generation failed'
                }
            
            # Extract only digits
            import re
            corrected = re.sub(r'\D', '', corrected)
            
            if len(corrected) == 10 and corrected[0] in '6789':
                return {
                    'success': True,
                    'corrected_value': corrected,
                    'confidence': 0.8,
                    'method': 'medgemma_llm'
                }
            else:
                return {
                    'success': False,
                    'corrected_value': None,
                    'confidence': 0.0,
                    'error': 'Invalid mobile number format'
                }
                
        except Exception as e:
            return {
                'success': False,
                'corrected_value': None,
                'confidence': 0.0,
                'error': f'LLM correction failed: {str(e)}'
            }
    
    def correct_store_name(self, raw_text: str) -> Dict[str, Any]:
        if not self.is_available():
            return {
                'success': False,
                'corrected_value': None,
                'confidence': 0.0,
                'error': 'MedGemma model not available'
            }
        
        prompt = f"""Fix OCR errors, correct spelling, add proper spacing, and complete abbreviations in this medical store/pharmacy name.

Rules:
- Fix spelling mistakes
- Add proper spacing between words
- Complete common abbreviations: PVT → Pvt. Ltd., INTL → International, PHARMA → Pharma
- Use Title Case (capitalize first letter of each word)
- Keep it concise and professional
- Output ONLY the corrected name, nothing else

Input: "{raw_text}"
Corrected name:"""

        try:
            corrected = self._generate_response(prompt, max_tokens=50)
            
            if corrected and len(corrected) >= 3:
                # Take first line only and clean up
                corrected = corrected.split('\n')[0].strip()
                # Remove any quotes or extra punctuation
                corrected = corrected.strip('"\'')
                
                return {
                    'success': True,
                    'corrected_value': corrected[:100],  # Max 100 chars
                    'confidence': 0.85,
                    'method': 'medgemma_llm'
                }
            else:
                return {
                    'success': False,
                    'corrected_value': None,
                    'confidence': 0.0,
                    'error': 'LLM output too short'
                }
                
        except Exception as e:
            return {
                'success': False,
                'corrected_value': None,
                'confidence': 0.0,
                'error': f'LLM correction failed: {str(e)}'
            }
    
    def correct_store_address(self, raw_text: str) -> Dict[str, Any]:
        if not self.is_available():
            return {
                'success': False,
                'corrected_value': None,
                'confidence': 0.0,
                'error': 'MedGemma model not available'
            }
        
        prompt = f"""Fix OCR errors and format this address properly with correct spacing, punctuation, and commas.

Rules:
- Fix spelling mistakes
- Add commas after street address, area names, and before city/pincode
- Add proper spacing between words
- Correct abbreviations: FLCOR → Floor, OPP → Opposite, NR → Near, RD → Road
- Keep numbers and pincodes intact
- Format as a single line with proper commas
- Output ONLY the corrected address, nothing else

Input: "{raw_text}"
Corrected address:"""

        try:
            corrected = self._generate_response(prompt, max_tokens=100)
            
            if corrected and len(corrected) >= 10:
                # Take first line only and clean up
                corrected = corrected.split('\n')[0].strip()
                # Remove any quotes
                corrected = corrected.strip('"\'')
                
                return {
                    'success': True,
                    'corrected_value': corrected[:200],  # Max 200 chars
                    'confidence': 0.85,
                    'method': 'medgemma_llm'
                }
            else:
                return {
                    'success': False,
                    'corrected_value': None,
                    'confidence': 0.0,
                    'error': 'LLM output too short'
                }
                
        except Exception as e:
            return {
                'success': False,
                'corrected_value': None,
                'confidence': 0.0,
                'error': f'LLM correction failed: {str(e)}'
            }

# Singleton instance
_medgemma_corrector = None

def get_medgemma_corrector() -> MedGemmaCorrector:
    """Get or create MedGemma corrector instance"""
    global _medgemma_corrector
    if _medgemma_corrector is None:
        # Check if LLM should be enabled via environment variable
        enable_llm = os.getenv("ENABLE_LLM_CORRECTIONS", "true").lower() == "true"
        _medgemma_corrector = MedGemmaCorrector(enable_llm=enable_llm)
    return _medgemma_corrector