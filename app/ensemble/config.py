import os

FASTER_RCNN_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "weights", "best_model.pth"
)
YOLO_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "weights", "best.pt"
)

CLASS_NAMES = [
    'date_of_reciept',
    'gstin',
    'invoice_no',
    'mobile_no',
    'product_table',
    'store_address',
    'store_name',
    'total_amount'
]
NUM_CLASSES = len(CLASS_NAMES)

# Mapping from ensemble class names to database field names
CLASS_TO_DB_FIELD = {
    'date_of_reciept': 'date_of_receipt',  # Note: fixed spelling for DB
    'gstin': 'gstin',
    'invoice_no': 'invoice_no',
    'mobile_no': 'mobile_no',
    'product_table': 'product_table',  # Special: goes to separate ProductItem table
    'store_address': 'store_address',
    'store_name': 'store_name',
    'total_amount': 'total_amount'
}

# Mapping from class index to class name (0-indexed)
CLASS_INDEX_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}

# Mapping from class name to index
CLASS_NAME_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

WBF_WEIGHTS = {
    'date_of_reciept': {'faster_rcnn': 0.48, 'yolo': 0.52},    # YOLO slightly better (99.24% vs 97.43%)
    'gstin': {'faster_rcnn': 0.55, 'yolo': 0.45},              # Faster R-CNN better (88.45% vs 83.60%)
    'invoice_no': {'faster_rcnn': 0.45, 'yolo': 0.55},         # YOLO better (97.32% vs 89.92%)
    'mobile_no': {'faster_rcnn': 0.52, 'yolo': 0.48},          # Both weak, slight edge to Faster R-CNN
    'product_table': {'faster_rcnn': 0.42, 'yolo': 0.58},      # YOLO much better (99.5% vs 90.91%)
    'store_address': {'faster_rcnn': 0.47, 'yolo': 0.53},      # YOLO better (95.55% vs 90.42%)
    'store_name': {'faster_rcnn': 0.48, 'yolo': 0.52},         # YOLO better (98.41% vs 90.91%)
    'total_amount': {'faster_rcnn': 0.43, 'yolo': 0.57}        # YOLO much better (99.48% vs 90.74%)
}

IOU_THRESHOLD = 0.5 

CLASS_IOU_THRESHOLDS = {
    'product_table': 0.5,      # Larger region, higher threshold
    'date_of_reciept': 0.3,    # Small field, lower threshold
    'gstin': 0.3,              # Small field
    'invoice_no': 0.3,         # Small field
    'mobile_no': 0.3,          # Small field
    'store_address': 0.4,      # Medium size
    'store_name': 0.4,         # Medium size
    'total_amount': 0.3        # Small field
}

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.5      # Minimum confidence to keep a detection
SKIP_BOX_THRESHOLD = 0.0001     # WBF parameter: skip boxes below this threshold

# Device configuration
DEVICE = 'cuda'  # 'cuda' or 'cpu'

# Image preprocessing
IMAGE_SIZE = 640 

# Ensemble mode
ENSEMBLE_MODE = 'wbf'  # Options: 'wbf', 'nms', 'class_specific'

DEBUG = False
SAVE_VISUALIZATIONS = True
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)