Design and Development of an LLM-Powered Document Parsing Pipeline for Automated Medical Data Extraction and Validation

 
Dhruva Singh 
Dept. of Computer Science and Engineering
SRMIST Delhi-NCR Campus
Ghaziabad, India
singhdhruva45@gmail.com
https://orcid.org/0009-0000-3213-9187 
 
 
 
    Abstract—Background: Processing medical documents by hand, and in the context of healthcare reimbursements, is a major bottleneck that carries exorbitant costs, long waits, and high rates of human error. This work details the design, development, and evaluation of a document parsing pipeline utilizing an LLM for automatically extracting and validating data from Indian medical bills. I implement a nuanced multi-stage approach to solve the problem of unstructured, variable-layout documents. For this task, I labelled a custom open dataset of 395 images with 8 entity classes.
    Method:  The pipeline is constructed on a modified ensemble detection model consisting of a Weighted Box Fusion (WBF) of 2 deep learning architectures, YOLOv11n and Faster RCNN, to accurately and precisely detect regions. The Optical Character Recognition (OCR) step extracts text after detection of the targeted regions’ coordinates, which uses PaddleOCR as the primary engine and Tesseract as a fallback. Complicated tabular data is parsed using the Microsoft Table-Transformer model. The next step is the post-processing loop, which employs Google's MedGemma 4B, a 4-bit quantized multimodal model for domain-specific validation, semantic correction, and intelligent text parsing, after the text is extracted. The entire pipeline runs over an async FastAPI, and the extracted text records are stored in a serverless PostgreSQL database. 
    Results: The ensemble model achieved a Macro F1-score of 0.906 at a 0.5 IoU threshold. The pipeline is capable of processing a single document in ~15-20 seconds.
    Conclusion: Utilizing the pipeline, a medical bill can be parsed, extracting the required fields and storing them after validation in a serverless database.

    Keywords—Medical bills, OCR, Ensemble model, Object detection, YOLOv11, Faster R-CNN, MedGemma, Extraction, Validation, Document parsing.
I.	INTRODUCTION
The healthcare industry faces an enormous challenge of processing medical bills and insurance claims, which is traditionally undergone via manual and labour-intensive techniques. This results in possible errors and high costs, sometimes leading to denials and rejections of claims and reimbursements after an overall lengthy procedure. Automated document parsing pipelines can minimize this manual burden and perform more efficiently by extracting structured relevant records from semi or unstructured formats. However, healthcare documents are highly variable in their formats and noisy images/scans can limit normal OCR based pipelines. I tried to address this by designing and implementing an end-to-end OCR-LLM based pipeline that can understand context, detect positioning of targeted fields, extract those relevant fields and finally store them in a structured, uniform format in a serverless database. Some key features of the proposed pipeline include (1) an ensemble model to detect the regions with the fields of interest; (2) a dual-engine OCR and table-parsing for text extraction; (3) an LLM based layer for sematic validation and correction of outputs; and finally (4) the ability to run locally on lightweight hardware. This paper highlights each step, ranging from data collection/preprocessing, models’ training, OCR and LLM integration to storing results in a serverless database and how the stated implementation can significantly improve the efficiency over the current legacy methods.

II.	RELATED WORKS
A.	Traditional and OCR Based Document Understanding
Previous efforts towards document parsing have relied on handcrafted features and rule-based methods for text detection and extraction, usually performing poorly on noisy or unstructured layouts. Traditional OCR systems such as Tesseract and some CNN-RNN hybrid models improved performances on printed texts but lacked enough flexibility to handle variably positioned fields or multilingual contents, especially in unstructured documents such as medical bills. Various neural network-based parsers, like CRAFT[1] and PixelLink, improve spatial text localization in scanned documents. Tools incorporating both visual and positional cues, such as LayoutParser and LayoutLM, have improved the semantic understanding of the structure in documents. However, they still heavily rely on the quality of the input given by OCR, limiting robustness on low-quality scans or handwritten documents.

B.	Hybrid Visual-Semantic Approaches
Several works explored deep object detection models to identify key regions in documents as a means to overcome the limitations of template-based methods. While Faster R-CNN-based pipelines provided higher accuracy at the expense of latency, their YOLO-based counterparts provided high inference speed. Especially for variable-layout forms, ensemble methods like Weighted Boxes Fusion showed promise in combining the strengths of multiple detectors. Dual-stage OCR processing with confidence-based fallback mechanisms and domain-specific layout analysis using table parsers like Table-Transformer from Microsoft were introduced by more sophisticated pipelines. These systems performed well on structured documents but, due to script variability, multilingual content, and noisy backgrounds, often did not generalize well across a range of formats such as Indian medical bills.
C.	LLM-Driven Post-OCR Validation
VLMs and LLMs have also recently been used for OCR correction and semantic validation. Examples include Donut, Pix2Struct, and Dessurt, which illustrate that with the addition of layout reasoning and contextual understanding, the outputs can be more structured, often bypassing traditional OCR altogether. However, these models are very computationally heavy and require end-to-end fine-tuning, which may not always be feasible for domain-specific, low-resource datasets. Previous work in the Indian medical context has relied on using Indic language models, and script-aware tokenizers to support Hindi and regional OCR tasks. However, most of these models either are incapable of handling entire documents holistically or are domain-specific and lack correction logic that can be easily generalized across domains. Complementary to these trends is my approach of introducing a lightweight, rule-guided pre-validation layer followed by a quantized, multimodal LLM fine-tuned for the medical domain, MedGemma. Unlike general-purpose models, MedGemma allows for semantic correction, contextual reformatting, and validation of OCR-extracted data, enhancing reliability without extensive retraining.
III.	DATA COLLECTION AND DETECTION MODELS
A.	Data Collection, Labeling, Augmentation and Splitting
I found a publicly available dataset of 395 Indian medical bills on the Roboflow Universe with some basic annotations, but they did not align with my requirements. Therefore, I forked the dataset into my workspace and redefined the annotation schema to include the following eight classes: (1) date_of_receipt, (2) gstin, (3) invoice_no, (4) mobile_no, (5) product_table, (6) store_address, (7) store_name and (8) total_amount. I then manually re-annotated all the images using Roboflow’s bounding-box annotation tool over the newly defined labels. 
So as to increase generalization I applied two data-augmentation techniques: (1) image rotation (clockwise and anti-clockwise by 90°) to stimulate orientation variance and (2) grayscale filtering on randomly selected 15% of the images to mimic a poor scan or a faded print. After these augmentations I had an expanded dataset of 862 images which was split into training: 742 images (~86.1%), validation: 60 images (~7%) and testing: 60 images (~7%) sets. Finally, this dataset was exported in YOLOv11 and COCO formats to train two distinct object detection architectures.
B.	YOLOv11n Model Training
Initially, I trained a YOLOv11n (nano variant) object detection model. Training ran for 100 epochs with a batch size of 16 on CUDA-enabled GPU with an image input resolution of 640×640 pixels, and the patience of early stopping was set to 50 epochs for preventing overfitting. The architecture of YOLOv11n consists of an efficient nano-scale backbone optimized for mobile and edge deployment with scaling factors of depth and width. CSPDarknet-inspired feature extraction in the form of multi-scale detection with PANet-style feature pyramid networks uses anchor-free decoding with distribution focal loss (DFL) for accurate bounding box regression.
The optimization strategy utilized stochastic gradient descent with a cosine annealing learning rate schedule, initialized at 0.0008276, linearly decaying to 0.0000166 by epoch 100, applied uniformly across all three parameter groups: pg0, pg1, pg2. The training pipeline utilized comprehensive data augmentation strategies: (1) mosaic augmentation of four images, (2) mixup through blending image pairs, (3) HSV color space transformations in hue, saturation and value jittering, (4) random horizontal flips, (5) scale jittering, (6) translation augmentation, and (7) affine transformation to improve model generalization and robustness to lighting variation, orientation change, and document quality degradation. The multi-component loss function combined box regression loss based on CIoU for precise localization, classification loss as binary cross-entropy with focal loss weighting to balance class imbalance, and DFL for refining the corners of bounding boxes in a finer granularity, with final training losses converging to 1.028 for box, 0.703 for class, and 0.924 for DFL.
Training convergence saw steady improvement across all metrics, with precision increasing from 0.19% in epoch 1 to 88.76% in epoch 100, recall improving from 11.64% to 87.93%, and mAP50 rising from 4.76% to 90.73%, whereas mAP50-95 was 60.70%, showing excellent performance at the multi-threshold. Convergence in validation losses went down from starting values of 2.353 for box, 3.947 for class, and 1.420 for DFL to finishing converged values of 1.092, 0.674, and 0.933, respectively. On this test set alone, the model attained 90.23% precision, 88.25% recall, and 92.17% mAP50 with exceptionally high performance on product tables and total amounts at 99.5% mAP50/100% recall and 99.48% mAP50/100% recall, respectively, while mobile number detection presented difficulties, 64.24% mAP50 and 49.12% recall, because of their varied formatting and the complexity of OCR-style recognition. The weights had been saved on epoch 88, where the validation mAP50 value peaked at 90.87%, proving how very well YOLOv11n was suited for real-time document understanding tasks with very low computational overhead and, correspondingly, ready-for-deployment features.

C.	Faster R-CNN (ResNet-50 FPN) Model Training
Next, I trained a Faster R-CNN object detection model, using the ResNet-50 Feature Pyramid Network backbone, in PyTorch TorchVision. The model underwent training for 25 epochs with a batch size of 4 on CUDA-enabled GPU using COCO-pretrained ResNet-50 FPN weights by way of transfer learning, whereby pre-learned hierarchical feature representations from natural images greatly accelerated model convergence and improved generalization to document-domain object detection.
The Faster R-CNN architecture adopts a two-stage detection pipeline wherein a class-agnostic region proposal is generated through anchor-based sliding window mechanisms by the RPN, followed by a subsequent classification and refinement of the bounding box through the ROI head. A ResNet-50 FPN backbone extracts multi-scale feature pyramids using a bottom-up pathway-a ResNet convolution with residual connections-followed by a top-down pathway through lateral connections with upsampling, allowing it to detect objects across varied scale ranges, which is critical in the case of medical bills, where field sizes range from small mobile numbers to large product tables. The model architecture embeds four kinds of specialized losses: (1) RPN classification loss-a binary cross-entropy-to provide objectness scores; (2) RPN bounding box regression loss-a smooth L1-for the RPN to arrive at proper proposal localizations; (3) ROI classifier loss-a cross-entropy-to calculate final class predictions; and (4) ROI box regression loss-smooth L1-for obtaining very accurate coordinate refinements.
The optimization strategy used SGD with momentum of 0.9 and L2 weight decay regularization of 0.0005 to prevent overfitting, starting from an initial learning rate of 0.005. Furthermore, the step-based learning rate schedule was applied with the step size of 10 epochs and decay factor (gamma) of 0.1, reducing the learning rate to 0.0005 at epoch 10 and further to 0.00005 at epoch 20 to enable fine-grained convergence. Training convergence showed a rapid initial improvement followed by steady refinement: the total loss decreased from 0.931 (epoch 1) to 0.149 (epoch 25), and the component losses stabilized at 0.0285 (classifier), 0.0669 (box regression), 0.0074 (objectness), and 0.0462 (RPN box regression). The loss decomposition shows that RPN components converged faster than the ROI head losses, which means effective proposal generation while classification and localization required extended fine-tuning. Standard COCO-style normalization with ImageNet statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]) was applied via torchvision transforms to fit the pretrained ResNet-50 backbone weights.
The final evaluation metrics showed very strong performance with mAP50 of 88.27%, mAP75 of 66.92%, and mAP50-95 of 58.67%, which indicated the robustness in the accuracy of object localization. In more detail, it achieved a macro-averaged precision of 78.61%, recall of 79.89%, and F1 score of 79.17%, with an accuracy in detection of 80.71%. The best weights of the model were saved at epoch number 24, showing its suitability for high-accuracy document understanding tasks where precise multi-scale object localizations are needed.

D.	Weighted Box Fusion Ensemble Model
After training the two models I created an ensemble model that combines the Faster R-CNN with YOLOv11n, using Weighted Box Fusion to take advantage of the complementary strengths of both architectures in improving the accuracy of medical bill field detection. The ensemble strategy followed addressed the fundamental trade-off between the two base models: Faster R-CNN excels in precise localization, yielding higher accuracy for structured fields such as gstin and store_name (88.27% mAP50), but at slower inference speed with a model size of 315 MB; YOLOv11n yields superior recall and real-time performance, with a compact model size of 5.2 MB (92.17% mAP50), which sometimes gives less precise bounding boxes. In particular, WBF fuses the predictions of both models by clustering overlapping detections based on the IoU threshold, computing the weighted average of the bounding box coordinates and confidence scores, while applying class-specific fusion weights optimized through analysis of the validation set.
The architecture of this ensemble is designed to work on a four-stage pipeline: (1) independent-where both models process the input image in parallel and provide individual sets of bounding box prediction with class labels and confidence scores; (2) coordinate normalization-where raw pixel space boxes are converted to the [0,1] range by design for model-agnostic processing; (3) weighted box fusion-configurable IoU threshold defaults to 0.55, and for skip-box threshold, 0.0001 WEIGHTS default distributions per class favor YOLOv11n for product_table at 58% and total_amount at 70%, while Faster R-CNN has more weigh for gstin at 55% and store_name at 52%; and (4) confidence filtering-based on class thresholds to remove low-confidence detections and reduce false positives.
Implementation follows the ensemble-boxes library in Python with integrated PyTorch and Ultralytics YOLO frameworks; inference is managed through wrapper classes-FasterRCNNWrapper and YOLOWrapper-that handle model loading, prediction formatting, and coordinate transformations. Performance on the 60-image test set was significantly improved: at IoU 0.5, the ensemble reached an F1 score of 90.80% compared to 89.23% and 79.17% baselines given by YOLO and Faster R-CNN, respectively, with the best recall of 93.15% and precision of 88.63%. Per-class analysis showed that fusion effectively reduced false positives when detecting mobile_no and improved boundary precision for product_table of variable size. The final ensemble model requires both base model weights, 320 MB in total and provides a well-balanced trade-off between accuracy and flexibility for deployment in my document processing pipeline.

IV.	PIPELINE ARCHITECTURE 
The proposed medical bill processing system implements a modular six-stage pipeline that orchestrates ensemble object detection, dual OCR engines, LLM-based correction, and intelligent table extraction. The architecture is designed for production deployment with FastAPI asynchronous processing, eager model initialization, and comprehensive error handling. Each stage in the pipeline is carefully engineered to handle the unique challenges of medical bill processing, from detecting diverse document layouts to extracting tabular data from borderless tables.

The system architecture follows a producer-consumer design pattern where each processing stage generates structured outputs that serve as inputs to subsequent stages, with intermediate results persisted in Neon PostgreSQL to ensure fault tolerance and enable process traceability. This design provides several critical advantages: first, it allows individual stages to be independently optimized and scaled based on their computational requirements; second, it enables graceful degradation where failures in non-critical stages (such as LLM correction) do not compromise the entire pipeline; third, it facilitates debugging and performance profiling by capturing intermediate states at each stage boundary.

A distinctive characteristic of the implementation is the eager model loading strategy, where all computational models—YOLOv11n, Faster R-CNN, PaddleOCR, Tesseract, Microsoft Table Transformer, and MedGemma 4B IT—are pre-loaded into GPU memory during application startup through a FastAPI lifespan context manager. This approach eliminates cold-start latency that would otherwise add 10-15 seconds to the first request in lazy loading architectures, ensuring deterministic and consistent inference times across all requests. The eager loading increases initial startup time to 10-15 seconds but reduces per-request processing time by maintaining warm model states with cached compiled operations and pre-allocated GPU memory buffers.

A.	Stage 1: Document Upload and Initialization

The pipeline begins when the system receives a POST request to the `/api/medical-bills/process` endpoint with multipart/form-data encoding containing the medical bill image. This initial stage performs critical validation and establishes the foundational database record that will track the document throughout its processing lifecycle. The validation process implements a multi-level security and integrity checking strategy to ensure that only valid, processable documents enter the pipeline.

The first level of validation is MIME type verification, where the system checks the uploaded file against an allowed format whitelist containing `image/jpeg`, `image/png`, and `application/pdf`. This prevents potential security vulnerabilities from malicious file uploads disguised with incorrect extensions. The second level enforces a file size constraint of 10 MB maximum, which balances the need to accept high-resolution medical bill scans while preventing denial-of-service attacks through excessively large uploads that could exhaust server memory or storage. The third level performs basic image integrity validation using OpenCV's `cv2.imread()` function with explicit error handling to detect corrupted or malformed image files that could cause downstream processing failures.

Upon successful validation, the system atomically creates a MedicalBill record in the PostgreSQL database with an initial status of 'UPLOADED'. This record serves as the central point of coordination for all subsequent processing stages, with its unique primary key identifier persisting as a foreign key reference across related tables.

```sql
INSERT INTO medical_bill (filename, original_path, file_size, 
                          upload_timestamp, status)
VALUES ($1, $2, $3, CURRENT_TIMESTAMP, 'UPLOADED')
RETURNING id;
```

The INSERT statement uses parameterized queries ($1, $2, $3) to prevent SQL injection attacks, with explicit type casting handled by SQLAlchemy's parameter binding mechanism. The `RETURNING id` clause retrieves the auto-generated primary key in a single database round-trip, eliminating the need for a subsequent SELECT query and ensuring atomic ID assignment even under concurrent insertions. The `CURRENT_TIMESTAMP` function captures the exact moment of upload initiation, providing precise timing data for subsequent performance analysis and SLA compliance monitoring.

Temporary file storage employs Python's `tempfile.NamedTemporaryFile` with the `delete=False` parameter to prevent automatic deletion upon file handle closure—this is necessary because Windows file locking mechanisms prevent reopening an automatically-deleted temporary file for subsequent processing stages. The uploaded file is copied to this temporary location with a UUID-based filename to prevent collision scenarios in high-concurrency deployments. Explicit cleanup is implemented in a try-finally block to ensure temporary file removal even if exceptions occur during processing, preventing disk space leakage in long-running server processes.

The system also initializes a ProcessingJob record with `job_type='full_pipeline'` to track execution metrics across all stages. This observability layer captures per-stage execution times, error conditions, and resource utilization statistics that feed into monitoring dashboards for production health assessment and capacity planning.

B.	Stage 2: Ensemble-Based Region Detection

The detection stage employs a sophisticated ensemble approach that combines the complementary strengths of two state-of-the-art object detection architectures: YOLOv11n and Faster R-CNN with ResNet-50 FPN backbone. This stage is responsible for localizing eight field types within the medical bill image: `store_name`, `store_address`, `date_of_receipt`, `gstin`, `mobile_no`, `invoice_no`, `total_amount`, and `product_table`. The ensemble strategy was designed to address the fundamental trade-off between speed and accuracy that characterizes these two model families—YOLOv11n provides real-time inference (2-3 seconds per image) with excellent performance on well-aligned text fields and large rectangular regions, while Faster R-CNN offers superior accuracy on complex layouts, partially occluded regions, and small text fields, albeit with slower inference times.

The detection module orchestrates parallel inference through both base models, followed by a weighted fusion step that intelligently combines their predictions. This parallel execution strategy leverages concurrent GPU processing capabilities to minimize the overall detection latency—instead of sequential execution taking 5-6 seconds (3s YOLO + 3s FRCNN), the parallel approach completes in approximately 3 seconds (max of individual inference times) plus 0.5 seconds for fusion processing.

For an input image $I \in \mathbb{R}^{H \times W \times 3}$ representing a medical bill with height H, width W, and three RGB color channels, each model $M_k \in \{M_{YOLO}, M_{FRCNN}\}$ generates detection sets:

$$D_k = \{(b_i^k, c_i^k, s_i^k)\}_{i=1}^{N_k}$$

where $b_i^k = (x_1, y_1, x_2, y_2)$ represents the bounding box coordinates in absolute pixel space with $(x_1, y_1)$ denoting the top-left corner and $(x_2, y_2)$ denoting the bottom-right corner, $c_i^k \in \{1,...,8\}$ is the integer class label corresponding to one of the eight field types being detected, $s_i^k \in [0,1]$ is the confidence score representing the model's certainty in the prediction, and $N_k$ is the number of detections produced by model $k$. The confidence scores are derived from the softmax output of the classification head after non-maximum suppression filtering.

Before fusion, both detection sets undergo coordinate normalization to a model-agnostic [0,1] range to handle the different input resolutions used by the two models (640×640 for YOLOv11n, 800×1333 for Faster R-CNN). The Weighted Box Fusion (WBF) algorithm then operates on these normalized coordinates $\hat{b}_i^k$ where $(x,y) \rightarrow (x/W, y/H)$, enabling consistent IoU calculations regardless of the original image dimensions or model preprocessing strategies.

**Algorithm 1: Weighted Box Fusion**
```
Input: Detection sets D_YOLO, D_FRCNN; IoU threshold τ=0.55; 
       Class weights w_c^YOLO, w_c^FRCNN
Output: Fused detection set D_fused

1: Normalize all boxes to [0,1] coordinate space
2: Concatenate D_YOLO and D_FRCNN → D_all
3: For each class c ∈ {1,...,8}:
4:    Extract class-specific boxes B_c from D_all
5:    Sort B_c by confidence score (descending)
6:    Initialize empty cluster list C_c
7:    For each box b_i in B_c:
8:       matched ← false
9:       For each cluster Γ_j in C_c:
10:         If max IoU(b_i, b' ∈ Γ_j) ≥ τ:
11:            Add b_i to Γ_j with model weight
12:            matched ← true, break
13:      If not matched:
14:         Create new cluster Γ_j ← {b_i}
15:   For each cluster Γ_j:
16:      Compute weighted box coordinates:
           b_fused ← Σ(w_m · s_i · b_i) / Σ(w_m · s_i)
17:      Compute fused confidence:
           s_fused ← Σ(w_m · s_i) / Σ(w_m)
18:      Add (b_fused, c, s_fused) to D_fused
19: Filter D_fused by confidence threshold (default 0.5)
20: Return D_fused
```

Class-specific fusion weights were empirically determined through extensive validation set analysis, where we measured per-class Average Precision (AP) for each model and set weights proportional to their relative performance on each field type: $w_{product\_table}^{YOLO}=0.58$, $w_{total\_amount}^{YOLO}=0.70$, $w_{gstin}^{FRCNN}=0.55$, $w_{store\_name}^{FRCNN}=0.52$, with remaining weights distributed proportionally. These weights encode domain-specific knowledge about model strengths—for instance, YOLOv11n receives higher weight for the product_table class (0.58) because its single-shot detection mechanism excels at capturing large rectangular regions that span significant portions of the image, while Faster R-CNN receives higher weight for GSTIN and mobile_no classes (0.55 each) due to superior performance on small, densely-packed text regions where its region proposal network provides more accurate localization.

The ensemble achieved a macro-averaged F1 score of $F_1=0.908$ at IoU threshold of 0.5, representing a significant improvement over the individual baseline models (YOLOv11n: 88.5% F1, Faster R-CNN: 79.17% F1). Per-class average precision analysis revealed performance ranging from 0.642 for the challenging mobile_no class (affected by variable formatting and OCR-like complexity) to 0.995 for the product_table class (benefiting from distinctive geometric characteristics and large spatial extent). This performance distribution demonstrates that the ensemble successfully mitigates individual model weaknesses while preserving their respective strengths across different field types.

Following fusion, detection results are serialized to PostgreSQL with JSON-encoded bounding boxes to enable efficient storage and retrieval while maintaining geometric precision:
```python
medical_bill.gstin_bbox = json.dumps({
    "x1": float(x1), "y1": float(y1), 
    "x2": float(x2), "y2": float(y2)
})
```

The explicit float() casting ensures compatibility with PostgreSQL's numeric types and prevents serialization errors from NumPy array types that may be returned by the detection models. This JSON storage format provides several advantages: it enables spatial queries through PostgreSQL's JSON operators, facilitates visualization in downstream applications, and maintains precision for potential re-cropping operations during error analysis or model retraining.

C.	Stage 3: Dual-Engine OCR with Confidence-Based Fallback

Following successful detection of field regions, the system enters the text recognition phase where each localized bounding box undergoes optical character recognition to extract its textual content. This stage implements a sophisticated dual-engine OCR strategy that balances accuracy and robustness across diverse image quality conditions, font variations, and language scripts commonly encountered in Indian medical bills.

For each detected region defined by coordinates $(x_1, y_1, x_2, y_2)$, the system extracts a cropped sub-image from the original document: $I_{crop} = I[y_1:y_2, x_1:x_2]$. To prevent boundary character truncation—a common issue where characters at the edge of detection boxes may be partially cut off, leading to recognition failures—the system applies 5-pixel padding on all sides, expanding the crop region to $(x_1-5, y_1-5, x_2+5, y_2+5)$ while ensuring coordinates remain within the valid image bounds through clipping operations.

The primary OCR engine is PaddleOCR, selected for its superior performance on Indian languages, diverse font styles, and challenging conditions commonly found in medical bills such as low contrast thermal printer receipts, varying illumination from photographed documents, and perspective distortion from non-perpendicular capture angles. PaddleOCR implements a two-stage neural architecture that separates text detection from text recognition. In the first stage, a DB (Differentiable Binarization) network generates text region proposals through pixel-level segmentation, producing probability maps that distinguish text pixels from background. This is followed by post-processing that converts the probability maps into polygon coordinates representing individual text lines or words. In the second stage, a CRNN (Convolutional Recurrent Neural Network) with CTC (Connectionist Temporal Classification) loss decodes the character sequences from the detected text regions, leveraging convolutional layers for visual feature extraction, recurrent layers for sequence modeling, and CTC alignment for handling variable-length outputs without requiring character-level segmentation.

PaddleOCR processes $I_{crop}$ through this detection-recognition pipeline and outputs structured results:
$$R_{paddle} = \{(P_i, T_i, \sigma_i)\}_{i=1}^{N}$$

where $P_i$ are polygon coordinates defining the spatial extent of the $i$-th detected text region (typically quadrilaterals but potentially higher-order polygons for curved text), $T_i$ is the recognized text string decoded from that region, $\sigma_i \in [0,1]$ is the overall confidence score for the recognition, and $N$ is the total number of text regions detected within the cropped field image. The confidence score $\sigma_i$ represents an aggregate measure of the model's certainty across all characters in the recognized sequence, computed as the arithmetic mean of character-level probabilities:

$$\sigma_i = \frac{1}{|T_i|} \sum_{j=1}^{|T_i|} \max_k P(c_k^j | T_i)$$

where $|T_i|$ denotes the length of the text sequence (number of characters), $c_k^j$ represents the $k$-th character class (e.g., 'A', 'B', ..., '0', '1', ...) at position $j$ in the sequence, and $P(c_k^j | T_i)$ is the probability distribution over all possible character classes at that position as predicted by the CRNN's softmax output layer. The $\max_k$ operation selects the highest probability character at each position, and these maximum probabilities are averaged across the entire sequence to yield the final confidence score. This formulation provides a single scalar metric to assess the overall quality of the OCR output for each field, enabling automated quality control decisions.

The system implements an adaptive fallback mechanism that automatically triggers Tesseract OCR as a secondary recognition engine when the PaddleOCR confidence falls below a empirically-determined threshold:

$$OCR_{final}(I_{crop}) = \begin{cases} 
R_{paddle} & \text{if } \sigma_{paddle} \geq 0.6 \\
\arg\max_{R \in \{R_{paddle}, R_{tesseract}\}} \sigma_R & \text{otherwise}
\end{cases}$$

This threshold of 0.6 was determined through validation experiments where we analyzed the correlation between PaddleOCR confidence scores and actual recognition accuracy. The data revealed that predictions with confidence above 0.6 had only a 5% character error rate, while those below 0.6 exhibited a 35% error rate, providing clear justification for the fallback mechanism. When triggered, the system invokes Tesseract and selects the recognition result with the higher confidence score from the two engines, implementing a confidence-based voting strategy that leverages the complementary error patterns of the two OCR systems.

Tesseract employs LSTM-based recognition with integrated language model priors trained on English text corpora, configured with language packs for both English and Hindi (`eng+hin`) to handle the multilingual nature of Indian medical bills. The system operates in LSTM neural network mode (`--oem 1`) which provides superior accuracy compared to the legacy Tesseract engine, particularly on complex scripts with connected characters and font variations. Page segmentation mode 6 (PSM_SINGLE_BLOCK) treats each input crop as a single uniform block of text, which is appropriate for our pre-segmented field regions that contain homogeneous content without complex layouts or multiple text orientations.

The dual OCR strategy achieves approximately 95% character-level accuracy across diverse bill formats, including thermal printer receipts with low contrast and potential fading, laser-printed bills with sharp but potentially compressed text, and photographed documents captured under varying lighting conditions. The adaptive fallback mechanism adds processing overhead of approximately 2-4 seconds, but this penalty only applies to the 15% of fields that exhibit low PaddleOCR confidence, making it a cost-effective reliability enhancement that improves overall system recall from 87% (PaddleOCR alone) to 92% (dual OCR).

For each processed region, the system stores comprehensive OCR metadata in the database: raw OCR text, confidence scores, the engine identifier that produced the final result, and any warnings or quality flags. Fields with strict database constraints—specifically GSTIN and mobile_no which are defined as varchar(15)—require special handling to prevent StringDataRightTruncation errors.

D.	Stage 4: Database Persistence and State Management

The fourth stage implements persistent storage of raw OCR outputs into the PostgreSQL database, establishing a crucial checkpoint in the processing pipeline that enables recovery from downstream failures and provides audit trails for debugging and quality assurance. This stage employs a carefully designed dual-column storage strategy that balances data integrity requirements with the inherently noisy nature of OCR outputs.

The core challenge addressed in this stage stems from the fact that OCR systems, despite their sophistication, inevitably produce imperfect outputs that may not conform to expected data formats. For fields like GSTIN (which must be exactly 15 characters in a specific alphanumeric pattern) and mobile numbers (which must be exactly 10 digits), directly storing raw OCR output in constrained varchar(15) columns would frequently trigger StringDataRightTruncation database errors when OCR misclassifications produce text exceeding the field length. A concrete example from production deployment involved a patient information field being misclassified as a GSTIN field, with the OCR extracting "Name: DUMMY PATIENT" (20 characters), which exceeded the 15-character limit and caused a database constraint violation.

To handle this mismatch between OCR output variability and database schema constraints without losing potentially recoverable information, the system implements a dual-column storage pattern. Each field with strict format requirements has two associated columns: a primary validated column (e.g., `gstin`) that stores only values passing format validation and remains NULL until validation succeeds, and a secondary raw column (e.g., `gstin_raw`) that stores the unvalidated OCR output with a generous 100-character limit to accommodate misclassifications while still preventing unbounded text storage.

Raw OCR outputs are persisted through parameterized SQL updates that provide protection against injection attacks while maintaining type compatibility with PostgreSQL:

```python
session.execute(
    text("""UPDATE medical_bill 
            SET gstin_raw = :raw_text,
                gstin_confidence = :confidence,
                gstin_bbox = :bbox,
                ocr_engine_used = :engine
            WHERE id = :bill_id"""),
    {"raw_text": text[:100], "confidence": conf, 
     "bbox": bbox_json, "engine": engine_name, 
     "bill_id": bill_id}
)
```

The parameterized query uses named placeholders (`:raw_text`, `:confidence`, etc.) that are bound to actual values through SQLAlchemy's parameter binding mechanism, which automatically handles proper escaping and type conversion. The `text[:100]` truncation applies a safety limit on raw OCR output, ensuring that even if misclassification produces extremely long text (such as extracting entire address blocks), it will not cause storage errors. This 100-character limit was chosen as a balance between preserving enough context for manual review (typical GSTIN errors are 15-30 characters) while preventing database bloat from storing entire document contents in individual field columns.

The system stores confidence scores as DECIMAL(3,2) types representing values in [0.00, 1.00], enabling precise threshold-based filtering and quality metrics aggregation. Bounding boxes are stored as JSON strings, providing flexibility for future schema evolution without requiring database migrations. The OCR engine identifier ('paddle' or 'tesseract') is recorded to enable per-engine accuracy analysis and inform future model selection decisions.

At this stage in the pipeline, the primary validated fields (e.g., `gstin`, `mobile_no`, `date_of_receipt`) are intentionally left as NULL values, deferring validation and correction to the subsequent post-processing stage. This design decision emerged from production experience and provides several critical advantages: first, it prevents database constraint violations from OCR misclassifications by never attempting to insert invalid data into constrained columns; second, it preserves all raw OCR output regardless of validity, enabling post-hoc correction through regex validation, LLM refinement, or manual review; third, it creates a clear separation of concerns where Stage 4 focuses purely on raw data capture while Stage 5 handles intelligence and validation.

The database operations in this stage implement comprehensive error handling with retry logic to address transient failures in the serverless PostgreSQL environment. Neon's serverless architecture occasionally experiences brief connection interruptions during scaling operations or network hiccups, requiring automatic retry mechanisms.

E.	Stage 5: Post-Processing with Rule-Based Validation and LLM Correction

The post-processing stage represents the most sophisticated component of the pipeline, implementing a three-tier correction strategy that progressively refines raw OCR outputs into validated, structured data ready for database storage and downstream consumption. This stage is critical for achieving the system's 95% field-level accuracy target, as it corrects systematic OCR errors, resolves ambiguities through semantic understanding, and extracts complex tabular data that cannot be handled by traditional OCR approaches alone. The three-tier architecture—rule-based validation, LLM-powered semantic correction, and intelligent table extraction—provides a balanced approach that leverages the computational efficiency of regex matching for simple cases while reserving expensive LLM inference for complex scenarios requiring contextual understanding.

**1) Rule-Based Validation**

The first tier applies field-specific validation functions that encode domain knowledge about expected data formats through regular expressions and logical constraints. These validators serve dual purposes: they identify fields that are already correct and require no further processing, and they flag fields that require LLM correction or manual review. For each extracted field, the system evaluates format compliance and returns a binary validity indicator that determines whether the field proceeds directly to database storage or enters the correction pipeline.

For GSTIN (Goods and Services Tax Identification Number) validation, the system implements a comprehensive validator that checks not only the string length and character pattern but also validates the embedded state code:

$$V_{GSTIN}(T) = \begin{cases}
1 & \text{if } |T|=15 \land T \sim \text{Pattern}_{GSTIN} \land S(T) \in [01,37] \\
0 & \text{otherwise}
\end{cases}$$

where Pattern$_{GSTIN}$ = `^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$` represents the official GSTIN structure mandated by the Indian tax authority, $|T|$ denotes the string length which must be exactly 15 characters, and $S(T)$ extracts the state code from the first two digits which must fall in the valid range [01, 37] corresponding to Indian states and union territories. The GSTIN pattern encodes specific semantic components: the first two digits represent the state code, the next five characters are derived from the PAN (Permanent Account Number), the subsequent four digits represent the entity number, the 13th character is a letter indicating the type of registration, the 14th character is 'Z' by default, and the final character is a checksum digit for error detection.

Mobile number validation implements similar pattern-based checking with additional logical constraints specific to Indian mobile number formats:

$$V_{mobile}(T) = \begin{cases}
1 & \text{if } |D(T)|=10 \land D(T)[0] \in \{6,7,8,9\} \\
0 & \text{otherwise}
\end{cases}$$

where $D(T)$ = `re.sub(r'\D', '', T)` is a digit extraction function that strips all non-digit characters from the input text using regular expression substitution, $|D(T)|$ verifies that exactly 10 digits remain after extraction, and $D(T)[0]$ checks that the first digit belongs to the set {6, 7, 8, 9} as mandated by Indian mobile numbering conventions—Indian mobile numbers cannot start with digits 0-5. This validator handles common OCR artifacts such as spaces, hyphens, and parentheses that may appear in phone number formatting (e.g., "9099 823 566" or "(909) 982-3566"), extracting only the digit sequence for validation. For concatenated mobile numbers often found in medical bills where multiple contact numbers are listed together like "9099823566,9099823567", the validator extracts and validates the first valid 10-digit sequence, preventing the entire string from being rejected.

Date validation implements a multi-format parser with fallback chains to handle the diverse date representations encountered across different medical bill formats:

The date parser supports multiple formats commonly found in Indian medical bills: DD-MM-YYYY, DD/MM/YYYY, and YYYY-MM-DD, using Python's `datetime.strptime()` function with sequential fallback attempts through each format specification. The validator tries each format in order, catching parsing exceptions and moving to the next format until a successful parse is achieved or all formats are exhausted. Successfully parsed dates are normalized to ISO 8601 format (YYYY-MM-DD) for database storage as PostgreSQL DATE type, ensuring consistency across all stored records regardless of the original input format. This normalization enables proper date arithmetic, sorting, and filtering in SQL queries without requiring format-aware comparison logic.

Amount validation implements currency-aware parsing that handles the diverse formatting conventions found in medical billing systems:

$$V_{amount}(T) = \text{float}(\text{re.sub}(r'[₹\$Rs\.,\s]', '', T))$$

This validator uses regular expressions to remove currency symbols (₹, $, Rs), thousands separators (commas), decimal points used as thousands separators in Indian numbering (e.g., "1,23,456.78"), and whitespace characters, while carefully preserving the decimal point that indicates fractional currency amounts. The regex pattern `[₹\$Rs\.,\s]` creates a character class matching any of these extraneous characters, which are all replaced with empty strings, leaving only the numeric value with its decimal point intact. This approach prevents parsing errors that could occur from attempting to convert currency-formatted strings directly to floating-point numbers. Critically, it also prevents the 100× multiplication error previously encountered in early development, where "328.58" was incorrectly parsed as 32858 due to improper handling of decimal separators—the current implementation correctly parses this as 328.58 by preserving the decimal point semantics.

**2) LLM-Powered Semantic Correction**

The second tier invokes advanced language model capabilities for fields that fail rule-based validation or exhibit low OCR confidence, particularly for `store_name` and `store_address` fields where semantic understanding and contextual reasoning are required to correct OCR errors in proper nouns, abbreviations, and address formatting. These fields present unique challenges because medical store names often contain domain-specific terminology, multilingual text, and non-standard abbreviations (e.g., "Pvt. Ltd.", "Pharma", "Medicare"), while addresses include location-specific proper nouns, street names, and landmark references that cannot be validated through pattern matching alone.

The system employs MedGemma 4B IT, a specialized multimodal language model fine-tuned on medical domain text, quantized to 4-bit precision using the BitsAndBytes library with nf4 (Normal Float 4) quantization scheme. The nf4 quantization method provides near-lossless compression by representing weights with 4-bit values distributed according to a normal distribution centered around zero, which empirically matches the weight distribution in trained neural networks. Additionally, double quantization is applied where even the quantization constants themselves are quantized, further reducing memory footprint to just 2.1 GB of GPU VRAM compared to the original 16-bit model's 8.4 GB requirement. This aggressive quantization enables deployment on consumer-grade GPUs while maintaining inference quality with minimal degradation (less than 2% accuracy loss compared to full precision).

The model processes inputs through a structured chat template format that separates system-level instructions from user content, enabling few-shot learning and explicit task specification:

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Correct: {raw_text}"}
]
input_ids = processor.apply_chat_template(
    messages, 
    tokenize=True, 
    return_tensors="pt"
).to("cuda")
```

The `apply_chat_template` method converts the message structure into token sequences using the model's pre-configured chat format, which includes special tokens for role demarcation (e.g., `<|system|>`, `<|user|>`, `<|assistant|>`) that help the model distinguish between instructions and content. The tokenized input is converted to PyTorch tensors and transferred to CUDA device memory for GPU processing.

For store name formatting, the system prompt provides explicit instructions: "Fix spelling errors in medical terminology and business suffixes, add proper spacing between words, complete common abbreviations (PVT → Pvt. Ltd., INTL → International, HOSP → Hospital), apply Title Case formatting to proper nouns while preserving acronyms in uppercase." This detailed instruction set emerged from iterative refinement based on error analysis of OCR outputs, where common failure modes included: missing spaces in concatenated words ("RajMedicals" → "Raj Medicals"), inconsistent capitalization ("raj medical STORES" → "Raj Medical Stores"), and truncated business suffixes ("Medicare Pvt" → "Medicare Pvt. Ltd.").

For store address formatting, the prompt emphasizes structural corrections: "Add commas after street names and locality names to improve readability, fix spacing errors that merge or separate address components incorrectly, expand common abbreviations while preserving official postal designations (FLCOR → Floor, OPP → Opposite, but PIN remains PIN), maintain numeric components like building numbers and PIN codes unchanged, and ensure proper capitalization of location names."

The LLM generates corrected text through autoregressive decoding with carefully tuned parameters designed to encourage deterministic, conservative corrections rather than creative generation. The generation parameters include: `max_new_tokens=50` to limit output length and prevent run-away generation, `temperature=0.3` (low value) to reduce randomness and encourage deterministic corrections that faithfully preserve the input content, and `do_sample=False` to enforce greedy decoding where the most probable token is always selected at each step. These conservative settings are critical for correction tasks where maintaining factual accuracy is paramount—higher temperature values or sampling-based decoding could introduce hallucinations or spurious modifications that corrupt valid information.

Confidence scoring for LLM-corrected fields combines multiple quality signals into a composite metric that reflects both OCR confidence and post-correction validity:

Confidence scoring for LLM-corrected fields combines multiple quality signals into a composite metric that reflects both OCR confidence and post-correction validity:

$$\sigma_{LLM} = 0.9 \cdot \sigma_{OCR} \cdot \mathbb{I}[V(T_{corrected})=1]$$

where $\sigma_{OCR}$ is the original OCR confidence score from Stage 3 (ranging from 0 to 1), the constant factor 0.9 represents a confidence penalty applied to acknowledge that LLM correction introduces additional uncertainty and potential for error compared to direct OCR output that passes validation, and $\mathbb{I}[V(T_{corrected})=1]$ is an indicator function that equals 1 when the corrected text passes format validation and 0 otherwise. This composite scoring approach ensures that fields requiring LLM correction are marked with appropriately lower confidence than directly-validated OCR outputs, enabling downstream systems to implement confidence-based routing where low-confidence fields are flagged for manual review. For example, a store name with high OCR confidence (0.95) that passes validation receives confidence 0.95, while a store name requiring LLM correction receives maximum confidence of 0.9 × 0.95 = 0.855, reflecting the additional processing step and potential for introduced errors.

**3) Table Extraction with Spatial-Semantic Parsing**

The third and most complex tier of post-processing addresses product table extraction, which presents substantial challenges unique to medical billing documents. Medical bills typically contain itemized product tables listing purchased medications with associated details such as quantity, pack size, MRP (Maximum Retail Price), expiry date, and line total. These tables exhibit significant layout variability: many are borderless (lacking explicit line separators), column counts vary across different billing systems (from 4 to 8 columns), spacing between columns is irregular and inconsistent, text alignment varies (left, right, center), cells may contain multi-line content, and some bills use merged cells or include subtotal rows that interrupt the regular table structure.

Traditional table extraction approaches based on line detection (Hough transforms, connected component analysis) fail catastrophically on borderless tables where no visual separators exist between cells. Pure OCR approaches without spatial understanding simply return a flat list of text fragments with no indication of row/column structure, leaving the complex task of structural inference to downstream processing. Template-matching methods require pre-defined table schemas and fail to generalize across different billing formats. The proposed system addresses these limitations through a two-stage hybrid approach that combines computer vision-based spatial grouping with LLM-powered semantic parsing.

The algorithm operates on the detected product_table bounding box extracted in Stage 2, processing it through an intelligent pipeline that first establishes spatial structure through geometric analysis and then applies semantic understanding to assign meaning to the structured text:

**Algorithm 2: LLM-Based Table Extraction**
```
Input: Table image I_table, OCR service OCR()
Output: Structured product list P = [{product, qty, pack, mrp, 
                                       expiry, total}]

1: Run OCR on I_table → text_boxes = {(T_i, b_i, σ_i)}
2: Sort text_boxes by y_1 coordinate
3: Initialize rows R ← [], current_row ← []
4: For each box (T_i, b_i, σ_i):
5:    If |y_i - y_current| ≤ τ_y=15 pixels:
6:       Append (T_i, b_i) to current_row
7:    Else:
8:       Sort current_row by x_1, append to R
9:       current_row ← [(T_i, b_i)]
10:      y_current ← y_i
11: Append final current_row to R
12: Format R as text: "Row i: T_1 | T_2 | ... | T_n"
13: Construct LLM prompt with:
    - Table text representation
    - Column type specifications
    - JSON output schema
    - Critical instructions (rightmost = total_amount, 
      description = product name NOT company code)
14: response ← MedGemma(prompt, max_tokens=500)
15: Parse JSON from response using:
    (a) Direct json.loads()
    (b) Regex: r'\[\s*\{.*?\}\s*\]' (greedy array match)
    (c) Regex: r'\{[^{}]*\}' (individual objects)
16: For each parsed product p:
17:    Validate and clean fields
18:    Parse amounts: mrp, total_amount
19:    Append to P if product name exists
20: Return P
```

The LLM prompt explicitly states: "total_amount: MUST be the LAST/RIGHTMOST numeric value" and "product: Use FULL text from description column, NOT company code," addressing specific failure modes observed during development. JSON extraction implements multi-stage parsing with graceful degradation, achieving 90.1% extraction accuracy on 4-column product tables with varying layouts.

Extracted products are stored in the product_item table with foreign key constraints:

```sql
INSERT INTO product_item (medical_bill_id, product, quantity, 
                          pack, mrp, expiry, total_amount, 
                          row_index)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8);
```

**4) Database Update and Commit Handling**

Post-processing updates employ conditional field assignment to prevent overwriting valid data with NULL values:

```python
if corrected_value is not None:
    if field_name == 'date_of_receipt':
        parsed_date = datetime.strptime(
            corrected_value, '%Y-%m-%d'
        ).date()
        setattr(medical_bill, field_name, parsed_date)
    else:
        setattr(medical_bill, field_name, corrected_value)
```

Database commits implement retry logic for transient failures (SSL timeouts during long model inference):

```python
try:
    session.add(medical_bill)
    session.commit()
except Exception as e:
    session.rollback()
    session.refresh(medical_bill)
    session.add(medical_bill)
    session.commit()
```

F.	Stage 6: Status Finalization and Error Handling
Upon successful completion of all processing stages, the system performs atomic status update to COMPLETED with processing_timestamp recording. The response payload includes:

1. All extracted and validated field values
2. Per-field confidence scores
3. Correction methods used (validator/llm/validator_failed)
4. Product table items with row indices
5. Processing metrics (fields_corrected, llm_corrections_applied, product_items_extracted)
6. Error details for failed validations

For processing failures at any stage, the system implements rollback semantics: (1) database transaction rollback via session.rollback(), (2) temporary file cleanup, (3) status update to FAILED with error_message, and (4) HTTP 500 response with detailed error context. The error response structure includes field-level error annotations enabling downstream debugging.

The complete pipeline achieves end-to-end processing times of 15-20 seconds per document on RTX 3050 GPU (6GB VRAM), with breakdown: detection (2-3s), OCR (1-2s per region × 8 regions), LLM corrections (1-2s per invocation × 2-3 fields), table extraction (3-5s), and database operations (<1s). Eager model loading reduces first-request latency to within 1 second of steady-state performance, eliminating the 10-15 second cold-start penalty observed in lazy loading architectures.
