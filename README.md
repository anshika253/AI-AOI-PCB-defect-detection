# AI-AOI-PCB-defect-detection
PCB Defect Detection using YOLOv8 & Computer Vision

** Project Overview**
An end-to-end AI-powered Automated Optical Inspection system for real-time PCB defect detection. This system replicates industrial AOI machines used in SMT manufacturing lines — built using open-source tools and deep learning.
 Inspired by hands-on experience in PCB Quality Inspection at Bhagwati Products Limited, Bhiwadi (SMT Department)

 Key Features

Real-time PCB defect detection using YOLOv8
Live camera feed with instant PASS/FAIL result
Streamlit web dashboard for image upload and analysis
Detects 7 defect types with bounding boxes
Defect reasoning — explains why each defect is flagged
CSV inspection logging with timestamp and defect details
Trained on 4400 real PCB image


ai-aoi-project/
├── image_basics.py          # Image loading and display
├── image_processing.py      # Preprocessing (grayscale, blur, edges)
├── defect_detection.py      # OpenCV contour-based detection
├── live_camera.py           # Basic live camera feed
├── train.py                 # YOLOv8 model training
├── test_model.py            # Model testing on images
├── live_ai_camera.py        # AI-powered live camera detection
├── dashboard.py             # Streamlit web dashboard
├── inspection_log.py        # CSV inspection logging
├── check_accuracy.py        # Model accuracy evaluation
└── images/                  # Project screenshots and graphs
    ├── dashboard.png
    ├── confusion_matrix.png
    ├── BoxF1_curve.png
    └── results.png
