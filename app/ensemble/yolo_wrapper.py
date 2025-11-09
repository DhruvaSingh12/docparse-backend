import numpy as np
from ultralytics.models import YOLO
import cv2

class YOLOWrapper:   
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        self.model = YOLO(weights_path)
        if device == 'cuda':
            self.model.to('cuda')
        
        print(f"YOLO v11n loaded on {self.device}")
    
    def predict(self, image, confidence_threshold=0.5, imgsz=640):
        # Run inference
        results = self.model.predict(
            source=image,
            conf=confidence_threshold,
            imgsz=imgsz,
            verbose=False,
            device=self.device
        )[0]
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # type: ignore
            labels = results.boxes.cls.cpu().numpy().astype(int)  # type: ignore
            scores = results.boxes.conf.cpu().numpy()  # type: ignore
        else:
            boxes = np.array([])
            labels = np.array([])
            scores = np.array([])
        labels = labels + 1
        
        return {
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        }
    
    def predict_batch(self, images, confidence_threshold=0.5, imgsz=640):
        # Run batch inference
        results = self.model.predict(
            source=images,
            conf=confidence_threshold,
            imgsz=imgsz,
            verbose=False,
            device=self.device
        )
        
        # Extract predictions for each image
        batch_predictions = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # type: ignore
                labels = result.boxes.cls.cpu().numpy().astype(int)  # type: ignore
                scores = result.boxes.conf.cpu().numpy()  # type: ignore
            else:
                boxes = np.array([])
                labels = np.array([])
                scores = np.array([])
            labels = labels + 1
            
            batch_predictions.append({
                'boxes': boxes,
                'labels': labels,
                'scores': scores
            })
        
        return batch_predictions
    
    def get_image_size(self, image):
        """Get image dimensions"""
        if isinstance(image, np.ndarray):
            return image.shape[:2]  # height, width
        elif isinstance(image, str):
            # Load image to get size
            img = cv2.imread(image)
            return img.shape[:2] if img is not None else None
        else:
            return image.size[::-1]  # width, height -> height, width


if __name__ == "__main__":
    from . import config
    model = YOLOWrapper(
        weights_path=config.YOLO_WEIGHTS,
        device=config.DEVICE
    )
    print("YOLO wrapper initialized successfully!")