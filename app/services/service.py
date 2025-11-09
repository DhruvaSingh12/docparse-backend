from typing import Dict, Any, List, Optional, Union
import numpy as np
import cv2
import tempfile
import os
from enum import Enum
from .engines import TesseractEngine, PaddleOCREngine

_ocr_service = None


class OCREngine(str, Enum):
    """OCR Engine options"""
    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"


class OCRService:
    def __init__(self):
        self.engines = []
        self.default_engine = None

        # Initialize available engines
        try:
            tesseract = TesseractEngine()
            self.engines.append(tesseract)
            if not self.default_engine:
                self.default_engine = tesseract
        except Exception:
            pass

        try:
            paddle = PaddleOCREngine()
            self.engines.append(paddle)
            self.default_engine = paddle
        except Exception:
            pass

        if not self.engines:
            raise RuntimeError("No OCR engines available. Please install tesseract or paddleocr.")

    def get_available_engines(self) -> List[str]:
        return [engine.name for engine in self.engines]
    
    async def process_image(
        self, 
        image: Union[str, np.ndarray], 
        engine: Optional[OCREngine] = None
    ) -> Dict[str, Any]:
        """
        Process image with OCR (supports both file paths and numpy arrays)
        
        Args:
            image: Image file path or numpy array
            engine: Specific engine to use (optional)
        
        Returns:
            OCR results dictionary
        """
        # If numpy array, save to temp file
        temp_path = None
        image_path: str = ""
        
        if isinstance(image, np.ndarray):
            # Create temporary file with delete=False to avoid file locking issues
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_path = temp_file.name
            
            # Write image to the temp file (now it's closed, so no lock)
            cv2.imwrite(temp_path, image)
            image_path = temp_path
        else:
            image_path = image
        
        try:
            # Process with specified or best engine
            if engine:
                result = self.process_with_engine(image_path, engine.value)
            else:
                result = self.process_with_best_engine(image_path)
            
            return result
        
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass  # Ignore cleanup errors

    def process_with_engine(self, image_path: str, engine_name: Optional[str] = None) -> Dict[str, Any]:
        if engine_name:
            engine = next((e for e in self.engines if e.name == engine_name), None)
            if not engine:
                raise ValueError(f"Engine '{engine_name}' not available")
        else:
            engine = self.default_engine
        
        if not engine:
            raise RuntimeError("No OCR engine available")
        
        return engine.process(image_path)

    def process_with_best_engine(self, image_path: str) -> Dict[str, Any]:
        if len(self.engines) == 1:
            return self.engines[0].process(image_path)
        results = []
        for engine in self.engines:
            try:
                results.append(engine.process(image_path))
            except Exception as e:
                print(f"Engine {engine.name} failed: {e}")
        if not results:
            return {"model": "none", "text": "", "confidence": 0.0, "blocks": [], "error": "All OCR engines failed"}
        best = max(results, key=lambda x: x.get("confidence", 0))
        return best


def get_ocr_service() -> OCRService:
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service


async def ocr_process(image_path: str, engine_name: Optional[str] = None) -> Dict[str, Any]:
    service = get_ocr_service()
    if engine_name:
        return service.process_with_engine(image_path, engine_name)
    return service.process_with_best_engine(image_path)
