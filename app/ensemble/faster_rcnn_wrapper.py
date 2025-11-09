import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from PIL import Image
import torchvision.transforms as T


class FasterRCNNWrapper:  
    def __init__(self, weights_path, num_classes=9, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load model
        self.model = self._load_model(weights_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Faster R-CNN loaded on {self.device}")
    
    def _load_model(self, weights_path):
        # Create model
        model = fasterrcnn_resnet50_fpn(weights=None)
        
        # Replace the classifier head
        box_predictor = model.roi_heads.box_predictor
        assert isinstance(box_predictor, FastRCNNPredictor)
        in_features = box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Check if it's a training checkpoint with metadata
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
        else:
            # If checkpoint is already the model state dict
            model.load_state_dict(checkpoint)
        
        return model
    
    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            rgb_image = image[:, :, ::-1]
            image = Image.fromarray(rgb_image)
        transform = T.Compose([
            T.ToTensor(),
        ])
        
        img_tensor: torch.Tensor = transform(image)  # type: ignore
        return img_tensor
    
    def predict(self, image, confidence_threshold=0.5):
        # Preprocess
        img_tensor = self.preprocess(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model([img_tensor])[0]
        
        # Filter by confidence
        keep_idx = predictions['scores'] >= confidence_threshold
        
        boxes = predictions['boxes'][keep_idx].cpu().numpy()
        labels = predictions['labels'][keep_idx].cpu().numpy()
        scores = predictions['scores'][keep_idx].cpu().numpy()
        
        return {
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        }
    
    def predict_batch(self, images, confidence_threshold=0.5):
        # Preprocess all images
        img_tensors = [self.preprocess(img).to(self.device) for img in images]
        
        # Inference
        with torch.no_grad():
            predictions = self.model(img_tensors)
        
        # Filter and format results
        results = []
        for pred in predictions:
            keep_idx = pred['scores'] >= confidence_threshold
            
            results.append({
                'boxes': pred['boxes'][keep_idx].cpu().numpy(),
                'labels': pred['labels'][keep_idx].cpu().numpy(),
                'scores': pred['scores'][keep_idx].cpu().numpy()
            })
        
        return results
    
    def get_image_size(self, image):
        """Get image dimensions"""
        if isinstance(image, np.ndarray):
            return image.shape[:2]  # height, width
        else:
            return image.size[::-1]  # width, height -> height, width


if __name__ == "__main__":
    # Test the wrapper
    from . import config
    
    model = FasterRCNNWrapper(
        weights_path=config.FASTER_RCNN_WEIGHTS,
        num_classes=config.NUM_CLASSES + 1,  # +1 for background
        device=config.DEVICE
    )
    
    print("Faster R-CNN wrapper initialized successfully!")