# Classroom Analytics with YOLOv8-Pose & BoT-SORT

This repository contains a robust computer vision pipeline for tracking students in a classroom environment and classifying their posture (Sitting vs. Standing).

## 🚀 Key Features

*   **SOTA Detection:** Uses `yolov8x-pose` for high-accuracy keypoint estimation.
*   **Spatio-Temporal Tracking:** Implements a custom **BoT-SORT** configuration with a 150-frame memory to handle long-term occlusion.
*   **Geometric Pose Analysis:** Calculates knee angles (`Hip -> Knee -> Ankle`) to correctly identify sitting students even in side profiles or when partially obscured.
*   **Head-Only Heuristic:** Detects students in the back row by inferring "Sitting" state if only the head/shoulders are visible.
*   **Crowd Handling:** Tuned NMS thresholds (`iou=0.6`) allow for shoulder-to-shoulder seating detection.

## 🛠️ Setup (Strict Environment)

To ensure reproducibility, use the following `conda` environment:

```bash
conda create -n classroom python=3.10 -y
conda activate classroom

# Install Pinned Dependencies (CUDA 12.x)
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 scipy==1.11.4 ultralytics==8.1.0 opencv-python==4.8.1.78 lapx
```

## 🏃‍♂️ Usage

1.  Place your input video in the project directory.
2.  Update `VIDEO_PATH` in `main.py` if necessary.
3.  Run the script:

```bash
python main.py
```

## 📊 Outputs

*   **`Final_FAIR_Classroom.mp4`**: Annotated video with bounding boxes, IDs, and states.
*   **`Analytics_FAIR.csv`**: Frame-by-frame analytics (Frame, ID, State, Confidence).
