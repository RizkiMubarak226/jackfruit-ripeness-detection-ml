# Jackfruit Ripeness Detection (MobileNet SSD)

## Overview
This project focuses on detecting jackfruit ripeness using object detection with MobileNet SSD.

## Features
- Data preprocessing & TFRecord pipeline
- Multi-model training (8 variants)
- Model evaluation (accuracy, confusion matrix, metrics)
- Real-time detection using webcam

## Tech Stack
- Python
- TensorFlow
- OpenCV

## Results & insights 
- Best model: SSD MobileNet V2 FPN Lite 320x320 (Optimized)
- mAP@0.5: ~77%
- Accuracy: up to 88%

### Key Findings
- Higher resolution (640x640) improves accuracy and F1-score, but increases latency.
- Lower resolution (320x320) provides faster inference (~28–31 FPS), suitable for real-time applications.
- Optimized models show better balance between precision and recall.
- MobileNet V2 FPN Lite is more efficient (~26 MB) compared to V1 (~88 MB), making it suitable for lightweight deployment.
- Trade-off exists between accuracy, speed, and model size.

### Performance
- Latency: ~0.03–0.06 seconds
- FPS: ~16–31
- RAM usage: ~2.6–2.8 GB

## Notes
This project was developed as my final thesis.
