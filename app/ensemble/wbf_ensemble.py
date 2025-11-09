import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import cv2
from .faster_rcnn_wrapper import FasterRCNNWrapper
from .yolo_wrapper import YOLOWrapper
from . import config

class WBFEnsemble:
    def __init__(
        self,
        faster_rcnn_weights=None,
        yolo_weights=None,
        device='cuda',
        confidence_threshold=0.5
    ):

        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Use config paths if not provided
        faster_rcnn_weights = faster_rcnn_weights or config.FASTER_RCNN_WEIGHTS
        yolo_weights = yolo_weights or config.YOLO_WEIGHTS
        
        # Initialize models
        print("Loading Faster R-CNN...")
        self.faster_rcnn = FasterRCNNWrapper(
            weights_path=faster_rcnn_weights,
            num_classes=config.NUM_CLASSES + 1,  # +1 for background
            device=device
        )
        
        print("Loading YOLO v11n...")
        self.yolo = YOLOWrapper(
            weights_path=yolo_weights,
            device=device
        )
        
        print("Ensemble model ready!")
    
    def _normalize_boxes(self, boxes, image_width, image_height):

        if len(boxes) == 0:
            return boxes
        
        normalized = boxes.copy()
        normalized[:, [0, 2]] /= image_width
        normalized[:, [1, 3]] /= image_height
        
        # Clip to [0, 1]
        normalized = np.clip(normalized, 0, 1)
        
        return normalized
    
    def _denormalize_boxes(self, boxes, image_width, image_height):
        if len(boxes) == 0:
            return boxes
        
        denormalized = boxes.copy()
        denormalized[:, [0, 2]] *= image_width
        denormalized[:, [1, 3]] *= image_height
        
        return denormalized
    
    def _get_class_weights(self, labels):
        weights = np.ones(len(labels))
        return weights
    
    def predict(
        self,
        image,
        iou_threshold=None,
        skip_box_threshold=None,
        use_class_weights=True
    ):
        # Set defaults
        iou_threshold = iou_threshold or config.IOU_THRESHOLD
        skip_box_threshold = skip_box_threshold or config.SKIP_BOX_THRESHOLD
        
        # Get image dimensions
        if isinstance(image, np.ndarray):
            image_height, image_width = image.shape[:2]
        else:
            image_width, image_height = image.size
        
        # Get predictions from both models
        print("Running Faster R-CNN inference...")
        frcnn_pred = self.faster_rcnn.predict(
            image,
            confidence_threshold=self.confidence_threshold
        )
        
        print("Running YOLO inference...")
        yolo_pred = self.yolo.predict(
            image,
            confidence_threshold=self.confidence_threshold
        )
        
        print(f"Faster R-CNN detections: {len(frcnn_pred['boxes'])}")
        print(f"YOLO detections: {len(yolo_pred['boxes'])}")
        
        # Normalize boxes for WBF
        frcnn_boxes_norm = self._normalize_boxes(
            frcnn_pred['boxes'], image_width, image_height
        )
        yolo_boxes_norm = self._normalize_boxes(
            yolo_pred['boxes'], image_width, image_height
        )
        
        # Prepare data for WBF
        boxes_list = [frcnn_boxes_norm.tolist(), yolo_boxes_norm.tolist()]
        scores_list = [frcnn_pred['scores'].tolist(), yolo_pred['scores'].tolist()]
        labels_list = [frcnn_pred['labels'].tolist(), yolo_pred['labels'].tolist()]
        
        # Model weights for WBF
        if use_class_weights:
            # Use class-specific weights (averaged for now, can be refined)
            weights = [0.48, 0.52]  # Slight preference to YOLO based on overall performance
        else:
            weights = [0.5, 0.5]
        
        # Apply Weighted Box Fusion
        print("Applying Weighted Box Fusion...")
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_threshold,
            skip_box_thr=skip_box_threshold
        )
        
        # Denormalize boxes
        boxes = self._denormalize_boxes(
            np.array(boxes), image_width, image_height
        )
        
        # Convert labels to integers
        labels = np.array(labels).astype(int)
        scores = np.array(scores)
        
        # Map labels to class names (subtract 1 because background is class 0)
        class_names = [config.CLASS_NAMES[label - 1] if label > 0 else 'background' 
                      for label in labels]
        
        print(f"Ensemble detections: {len(boxes)}")
        
        return {
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'class_names': class_names
        }
    
    def predict_batch(
        self,
        images,
        iou_threshold=None,
        skip_box_threshold=None,
        use_class_weights=True
    ):
        results = []
        for image in images:
            result = self.predict(
                image,
                iou_threshold=iou_threshold,
                skip_box_threshold=skip_box_threshold,
                use_class_weights=use_class_weights
            )
            results.append(result)
        
        return results
    
    def visualize_predictions(
        self,
        image,
        predictions,
        save_path=None,
        show=False
    ):

        # Convert to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Copy image for drawing
        vis_image = image.copy()
        
        # Draw boxes
        for box, label, score, class_name in zip(
            predictions['boxes'],
            predictions['labels'],
            predictions['scores'],
            predictions['class_names']
        ):
            x1, y1, x2, y2 = box.astype(int)
            
            # Color based on class (cycle through colors)
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (128, 0, 128), (255, 165, 0)
            ]
            color = colors[(label - 1) % len(colors)]
            
            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{class_name}: {score:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            cv2.putText(
                vis_image,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"Visualization saved to {save_path}")
        
        # Show if requested
        if show:
            cv2.imshow("Ensemble Predictions", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return vis_image


if __name__ == "__main__":
    print("Initializing ensemble model...")
    ensemble = WBFEnsemble(
        device=config.DEVICE,
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )
    
    print("\nEnsemble model ready for inference!")
    print(f"Models loaded: Faster R-CNN + YOLO v11n")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Classes: {', '.join(config.CLASS_NAMES)}")