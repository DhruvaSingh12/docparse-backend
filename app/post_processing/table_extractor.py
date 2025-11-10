import numpy as np
import cv2
from typing import List, Dict, Any, Optional
import torch
from PIL import Image
import json
import re

class TableTransformerExtractor:
    def __init__(self):
        self.detection_model: Optional[Any] = None
        self.structure_model: Optional[Any] = None
        self.detection_processor: Optional[Any] = None
        self.structure_processor: Optional[Any] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False
        self.llm_corrector = None  # Will be injected from pipeline
        
    def load_models(self):
        if self.loaded:
            return
        try:
            from transformers import AutoModelForObjectDetection, AutoImageProcessor
            print("Loading Table Transformer models...")
            
            # Table detection model
            self.detection_processor = AutoImageProcessor.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            self.detection_model = AutoModelForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            ).to(self.device)
            
            # Table structure recognition model
            self.structure_processor = AutoImageProcessor.from_pretrained(
                "microsoft/table-transformer-structure-recognition"
            )
            self.structure_model = AutoModelForObjectDetection.from_pretrained(
                "microsoft/table-transformer-structure-recognition"
            ).to(self.device)
            
            self.loaded = True
            print(f"✓ Table Transformer models loaded on {self.device}")
            
        except Exception as e:
            print(f"✗ Failed to load Table Transformer models: {e}")
            self.loaded = False
    
    def detect_tables(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if not self.loaded:
            self.load_models()
        if not self.loaded or self.detection_model is None or self.detection_processor is None:
            return []
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Prepare inputs
            inputs = self.detection_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run detection
            with torch.no_grad():
                outputs = self.detection_model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image_rgb.shape[:2]]).to(self.device)
            results = self.detection_processor.post_process_object_detection(
                outputs, 
                threshold=0.6,
                target_sizes=target_sizes
            )[0]
            
            tables = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if label.item() == 0:  # Table class
                    x1, y1, x2, y2 = box.tolist()
                    tables.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(score),
                        'type': 'table'
                    })
            
            return tables
            
        except Exception as e:
            print(f"✗ Table detection failed: {e}")
            return []
    
    def extract_table_structure(self, image: np.ndarray, table_bbox: List[int]) -> Dict[str, Any]:
        if not self.loaded:
            self.load_models()
        
        if not self.loaded or self.structure_model is None or self.structure_processor is None:
            return {'rows': [], 'columns': [], 'cells': []}
        
        try:
            # Crop table region
            x1, y1, x2, y2 = table_bbox
            table_crop = image[y1:y2, x1:x2]
            
            # Convert BGR to RGB
            table_rgb = cv2.cvtColor(table_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(table_rgb)
            
            # Prepare inputs
            inputs = self.structure_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run structure recognition
            with torch.no_grad():
                outputs = self.structure_model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([table_rgb.shape[:2]]).to(self.device)
            results = self.structure_processor.post_process_object_detection(
                outputs,
                threshold=0.6,
                target_sizes=target_sizes
            )[0]
            
            # Organize by type
            rows = []
            columns = []
            cells = []
            
            # Label mapping for table structure
            id2label = self.structure_model.config.id2label
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = id2label[label.item()]
                x1_rel, y1_rel, x2_rel, y2_rel = box.tolist()
                
                # Convert to absolute coordinates
                abs_box = [
                    int(x1_rel + table_bbox[0]),
                    int(y1_rel + table_bbox[1]),
                    int(x2_rel + table_bbox[0]),
                    int(y2_rel + table_bbox[1])
                ]
                
                element = {
                    'bbox': abs_box,
                    'confidence': float(score),
                    'type': label_name
                }
                
                if 'row' in label_name.lower():
                    rows.append(element)
                elif 'column' in label_name.lower():
                    columns.append(element)
                elif 'cell' in label_name.lower():
                    cells.append(element)
            
            # Sort rows by y-coordinate
            rows.sort(key=lambda r: r['bbox'][1])
            
            # Sort columns by x-coordinate
            columns.sort(key=lambda c: c['bbox'][0])
            
            return {
                'rows': rows,
                'columns': columns,
                'cells': cells,
                'num_rows': len(rows),
                'num_columns': len(columns),
                'num_cells': len(cells)
            }
            
        except Exception as e:
            print(f"✗ Table structure extraction failed: {e}")
            return {'rows': [], 'columns': [], 'cells': []}
    
    def map_cells_to_grid(self, structure: Dict[str, Any]) -> List[List[Optional[Dict]]]:
        rows = structure['rows']
        columns = structure['columns']
        cells = structure['cells']
        
        if not rows or not columns or not cells:
            return []
        
        # Create empty grid
        grid: List[List[Optional[Dict]]] = [[None for _ in range(len(columns))] for _ in range(len(rows))]
        
        # Assign each cell to grid position
        for cell in cells:
            cell_bbox = cell['bbox']
            cell_center_x = (cell_bbox[0] + cell_bbox[2]) / 2
            cell_center_y = (cell_bbox[1] + cell_bbox[3]) / 2
            
            # Find row
            row_idx = None
            for i, row in enumerate(rows):
                if row['bbox'][1] <= cell_center_y <= row['bbox'][3]:
                    row_idx = i
                    break
            
            # Find column
            col_idx = None
            for j, col in enumerate(columns):
                if col['bbox'][0] <= cell_center_x <= col['bbox'][2]:
                    col_idx = j
                    break
            
            if row_idx is not None and col_idx is not None:
                grid[row_idx][col_idx] = cell
        
        return grid

    async def extract_table_with_llm(
        self, 
        image: np.ndarray, 
        table_bbox: List[int],
        ocr_service: Any,
        llm_corrector: Any = None
    ) -> List[Dict[str, Any]]:
        """
        Extract table data using OCR + MedGemma LLM for intelligent parsing.
        This method handles tables without clear borders and varying column names.
        """
        try:
            # Crop table region
            x1, y1, x2, y2 = table_bbox
            table_crop = image[y1:y2, x1:x2]
            
            print(f"  Table region: {x2-x1}x{y2-y1} pixels")
            
            # Save temp crop for OCR
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, table_crop)
                temp_path = tmp.name
            
            try:
                # Run PaddleOCR to get all text with coordinates
                from app.services.ocr_service import OCREngine
                
                print("  Running OCR on table region...")
                ocr_result = await ocr_service.process_image(
                    table_crop,
                    engine=OCREngine.PADDLEOCR
                )
                
                # Parse OCR blocks to get text with coordinates
                text_boxes = []
                blocks = ocr_result.get('blocks', [])
                
                for block in blocks:
                    bbox = block.get('bbox', [])
                    if len(bbox) >= 4:
                        text_boxes.append({
                            'text': block.get('text', ''),
                            'x1': int(bbox[0]),
                            'y1': int(bbox[1]),
                            'x2': int(bbox[2]),
                            'y2': int(bbox[3]),
                            'confidence': float(block.get('confidence', 0))
                        })
                
                if not text_boxes:
                    print("  ⚠ No text detected in table region")
                    return []
                
                print(f"  Detected {len(text_boxes)} text elements")
                
                # Group text boxes into rows based on y-coordinates
                rows = self._group_into_rows(text_boxes)
                print(f"  Grouped into {len(rows)} rows")
                
                if len(rows) < 2:  # Need at least header + 1 data row
                    print("  ⚠ Not enough rows detected")
                    return []
                
                # Use LLM to understand the table structure and extract data
                if llm_corrector and llm_corrector.is_available():
                    print("  Using MedGemma to parse table structure...")
                    products = self._parse_table_with_llm(table_crop, rows, llm_corrector)
                    return products
                else:
                    # Fallback: Simple heuristic-based parsing
                    print("  Using heuristic parsing (LLM not available)...")
                    products = self._parse_table_heuristic(rows)
                    return products
                    
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
        except Exception as e:
            print(f"✗ LLM-based table extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _group_into_rows(self, text_boxes: List[Dict], y_threshold: int = 15) -> List[List[Dict]]:
        """
        Group text boxes into rows based on y-coordinate proximity.
        """
        if not text_boxes:
            return []
        
        # Sort by y-coordinate
        sorted_boxes = sorted(text_boxes, key=lambda b: b['y1'])
        
        rows = []
        current_row = [sorted_boxes[0]]
        current_y = sorted_boxes[0]['y1']
        
        for box in sorted_boxes[1:]:
            # If this box is close to current row's y-coordinate, add to same row
            if abs(box['y1'] - current_y) <= y_threshold:
                current_row.append(box)
            else:
                # Start new row
                # Sort current row by x-coordinate
                current_row.sort(key=lambda b: b['x1'])
                rows.append(current_row)
                current_row = [box]
                current_y = box['y1']
        
        # Don't forget the last row
        if current_row:
            current_row.sort(key=lambda b: b['x1'])
            rows.append(current_row)
        
        return rows
    
    def _parse_table_with_llm(self, table_image: np.ndarray, rows: List[List[Dict]], llm_corrector: Any) -> List[Dict[str, Any]]:
        """
        Use MedGemma to intelligently parse the table structure.
        """
        try:
            # Prepare the table data as text
            table_text = self._format_table_as_text(rows)
            
            # Create prompt for LLM
            prompt = f"""You are analyzing a medical bill product table. Extract each product as a structured JSON object.

Table text:
{table_text}

Your task:
1. Identify columns: Comp/Company, Product Description/Name, Qty/Quantity, Pack, Batch No, Exp/Expiry, MRP, and Amount (RIGHTMOST/LAST numeric column)
2. Extract each data row (skip the header row)
3. Return ONLY a valid JSON array of product objects

Required output format (return ONLY this JSON, no other text):
[
  {{
    "product": "full product name from Product Description column",
    "quantity": "quantity value",
    "pack": "pack size",
    "mrp": "MRP per unit price",
    "expiry": "expiry date if available, else null",
    "total_amount": "LAST/RIGHTMOST amount in the row (final payable amount)"
  }}
]

CRITICAL Rules:
- Return ONLY valid JSON array, no explanations or additional text
- Skip header rows completely
- Product name: Use the FULL text from "Product Description" column, NOT the company code
- total_amount: MUST be the LAST/RIGHTMOST numeric value in each row (usually the largest amount)
- total_amount is NOT "Net Rate" or "MRP" - it's the final "Amount" column
- MRP is the per-unit price, total_amount is the final line total
- If a field is not found, use null
- Quantity is usually a small number (1-100)
- Expiry date format: MM-YY or DD-MM-YYYY
- Pack format: examples "1X1TAB", "1X100GM", "1X30CAP"
- Expiry date might be in format MM/YY or DD/MM/YYYY
- If expiry is not found, use null"""

            # Use LLM to parse (using text-only since we have OCR'd the table)
            response = llm_corrector._generate_response(prompt, max_tokens=500)
            
            if not response:
                print("  ⚠ LLM returned empty response")
                return []
            
            # Extract JSON from response
            products = self._extract_json_from_response(response)
            
            if products:
                print(f"  ✓ LLM extracted {len(products)} products")
                return products
            else:
                print("  ⚠ Could not parse LLM response as JSON")
                return []
                
        except Exception as e:
            print(f"  ✗ LLM parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _format_table_as_text(self, rows: List[List[Dict]]) -> str:
        """
        Format the grouped rows as readable text for the LLM.
        """
        lines = []
        for i, row in enumerate(rows):
            row_text = " | ".join([box['text'] for box in row])
            lines.append(f"Row {i}: {row_text}")
        return "\n".join(lines)
    
    def _extract_json_from_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract JSON array from LLM response, handling various formats.
        """
        try:
            # Try direct JSON parse
            products = json.loads(response)
            if isinstance(products, list):
                return products
        except:
            pass
        
        # Try to find JSON array in the response
        json_pattern = r'\[\s*\{.*?\}\s*\]'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                products = json.loads(match)
                if isinstance(products, list):
                    return products
            except:
                continue
        
        # Try to find individual JSON objects
        json_obj_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_obj_pattern, response, re.DOTALL)
        
        products = []
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict):
                    products.append(obj)
            except:
                continue
        
        return products if products else []
    
    def _parse_table_heuristic(self, rows: List[List[Dict]]) -> List[Dict[str, Any]]:
        """
        Fallback heuristic-based parsing when LLM is not available.
        """
        products = []
        
        # Skip first row (header)
        for row in rows[1:]:
            if len(row) < 3:  # Need at least 3 columns
                continue
            
            # Simple heuristic mapping
            product_data = {
                'product': row[0]['text'] if len(row) > 0 else None,
                'quantity': row[1]['text'] if len(row) > 1 else None,
                'pack': row[2]['text'] if len(row) > 2 else None,
                'mrp': row[3]['text'] if len(row) > 3 else None,
                'expiry': row[4]['text'] if len(row) > 4 else None,
                'total_amount': row[-1]['text'] if len(row) > 0 else None  # Last column often total
            }
            
            products.append(product_data)
        
        return products

# Singleton instance
_table_transformer = None

def get_table_transformer() -> TableTransformerExtractor:
    global _table_transformer
    if _table_transformer is None:
        _table_transformer = TableTransformerExtractor()
    return _table_transformer