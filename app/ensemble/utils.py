import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from . import config

def bbox_to_json(bbox: np.ndarray) -> Optional[str]:
    if bbox is None or len(bbox) == 0:
        return None
    
    return json.dumps({
        "x1": float(bbox[0]),
        "y1": float(bbox[1]),
        "x2": float(bbox[2]),
        "y2": float(bbox[3])
    })

def json_to_bbox(bbox_json: str) -> Optional[np.ndarray]:
    if not bbox_json:
        return None
    
    bbox_dict = json.loads(bbox_json)
    return np.array([
        bbox_dict["x1"],
        bbox_dict["y1"],
        bbox_dict["x2"],
        bbox_dict["y2"]
    ])

def get_db_field_name(class_name: str) -> str:
    return config.CLASS_TO_DB_FIELD.get(class_name, class_name)

def organize_predictions_by_class(predictions: Dict) -> Dict[str, Dict]:
    organized = {}
    
    for box, label, score, class_name in zip(
        predictions['boxes'],
        predictions['labels'],
        predictions['scores'],
        predictions['class_names']
    ):
        # Skip if class already detected (take first/highest confidence one)
        if class_name not in organized:
            organized[class_name] = {
                'bbox': box,
                'bbox_json': bbox_to_json(box),
                'confidence': float(score),
                'class_name': class_name,
                'db_field': get_db_field_name(class_name),
                'label': int(label)
            }
    
    return organized

def extract_region_from_image(image: np.ndarray, bbox: np.ndarray, 
                              padding: int = 5) -> np.ndarray:
    h, w = image.shape[:2]
    
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Add padding and clip to image boundaries
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2]

def calculate_average_confidence(predictions: Dict) -> float:
    if len(predictions['scores']) == 0:
        return 0.0
    
    return float(np.mean(predictions['scores']))

def filter_predictions_by_confidence(predictions: Dict, 
                                     min_confidence: float = 0.5) -> Dict:
    mask = predictions['scores'] >= min_confidence
    
    return {
        'boxes': predictions['boxes'][mask],
        'labels': predictions['labels'][mask],
        'scores': predictions['scores'][mask],
        'class_names': [name for i, name in enumerate(predictions['class_names']) if mask[i]]
    }

def get_class_specific_iou_threshold(class_name: str) -> float:
    return config.CLASS_IOU_THRESHOLDS.get(class_name, config.IOU_THRESHOLD)

def validate_predictions(predictions: Dict) -> Tuple[bool, List[str]]:
    detected_classes = set(predictions.keys())
    required_classes = set(config.CLASS_NAMES)
    
    missing_classes = required_classes - detected_classes
    
    is_valid = len(missing_classes) == 0
    
    return is_valid, list(missing_classes)

def format_detection_summary(predictions: Dict) -> str:
    summary_lines = [
        "Detection Summary:",
        f"Total detections: {len(predictions.get('boxes', []))}",
        f"Average confidence: {calculate_average_confidence(predictions):.2%}",
        "\nDetected classes:"
    ]
    
    for class_name in predictions.get('class_names', []):
        summary_lines.append(f"  - {class_name}")
    
    return "\n".join(summary_lines)

__all__ = [
    'bbox_to_json',
    'json_to_bbox',
    'get_db_field_name',
    'organize_predictions_by_class',
    'extract_region_from_image',
    'calculate_average_confidence',
    'filter_predictions_by_confidence',
    'get_class_specific_iou_threshold',
    'validate_predictions',
    'format_detection_summary'
]