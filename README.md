# BioMech

A sports biomechanics pipeline built with MediaPipe and OpenCV.

## What it does
- Detects 33 body landmarks on athletes in real time using MediaPipe Pose
- Calculates bilateral knee angles frame by frame using vector mathematics
- Compares left vs right knee symmetry across drills
- Outputs angle data as a time-series graph for performance analysis

## Key findings
From a basketball ladder drill analysis:
- Left knee deepest bend: 52.8° | Right knee: 66.8°
- Average asymmetry of ~6.5° across the full session
- Asymmetry is drill-dependent — more pronounced in lateral movements

### Results
- Left Knee | Avg: 144.06 | Deepest: 52.8 | Straightest: 179.8
- Right Knee | Avg: 137.52 | Deepest: 66.8 | Straightest: 179.6

## Stack
Python, OpenCV, MediaPipe, NumPy, Matplotlib
