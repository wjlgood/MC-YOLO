# MC-YOLO: A Lightweight Insulator Defect Detection Model Based on an Improved YOLOv8

**Author:** Wang Junlian (`@wjlgood`)  
**Affiliation:** Kunming University  
**Contact:** https://github.com/wjlgood

## 🚀 Project overview
This repository contains an improved YOLOv8 implementation optimized for UAV-based insulator defect detection.
Key improvements:
- MobileNetV3 as lightweight backbone for lower FLOPs and faster inference.
- Added attention modules (CBAM) to enhance feature representation.
- Replaced the original loss function with **WiouV3** to enhance bounding box regression and overall detection precision.
- Preserved feature fusion capability to maintain detection performance for both large and small targets.
- Dataset: 3000+ UAV-captured insulator images with complex backgrounds.

## 📜 Acknowledgements 
This project is based on the [Ultralytics YOLOv8 official repository](https://github.com/ultralytics/ultralytics).

## 📁 Repository structure

