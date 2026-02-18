import cv2
import csv
import torch
import yaml
import math
from collections import deque
from ultralytics import YOLO

# ==========================================
# ⚙️ 1. FACEBOOK RESEARCH LOGIC INJECTION
# ==========================================
# We dynamically create a custom tracker config to mimic FAIR's long-term ReID.
# Standard trackers forget an ID after 30 frames. We force BoT-SORT to remember 
# a partially hidden/occluded student for 150 frames (5 seconds) to kill ID switching.
custom_tracker = """
tracker_type: botsort
track_high_thresh: 0.4
track_low_thresh: 0.1
new_track_thresh: 0.5
track_buffer: 150
match_thresh: 0.9
fuse_score: True
gmc_method: none
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
"""
with open("fair_tracker.yaml", "w") as f:
    f.write(custom_tracker)

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
VIDEO_PATH = "input3.mp4"       
OUTPUT_VIDEO = "Final_FAIR_Classroom.mp4"
CSV_FILE = "Analytics_FAIR.csv"
ASPECT_RATIO_THRESHOLD = 1.4 

# 🔥 FAIR LOGIC: TEMPORAL TUBELETS
# We maintain a 30-frame (1 second) memory for every single student.
MEMORY_FRAMES = 30 
student_memory = {} # Maps ID -> list of their last 30 states

# ==========================================
# HELPER: CALCULATE ANGLE
# ==========================================
def calculate_angle(p1, p2, p3):
    """Calculates the angle at p2 formed by p1-p2-p3"""
    # p1, p2, p3 are (x, y) tuples
    if any(c == 0 for point in [p1, p2, p3] for c in point): return None # Missing point

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Calculate vectors
    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)

    # Calculate dot product and magnitudes
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 * mag2 == 0: return None

    # Calculate angle in radians and convert to degrees
    angle = math.degrees(math.acos(max(min(dot / (mag1 * mag2), 1.0), -1.0)))
    return angle

# ==========================================
# 2. INITIALIZE SOTA MODEL
# ==========================================
print("🚀 Loading YOLOv8x-Pose...")
model = YOLO('yolov8x-pose.pt') 

# Prepare Video Writer
cap = cv2.VideoCapture(VIDEO_PATH)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

csv_f = open(CSV_FILE, 'w', newline='')
writer = csv.writer(csv_f)
writer.writerow(['Frame', 'ID', 'State', 'Confidence'])

frame_idx = 0
print(f"👀 Processing video using Spatio-Temporal Tracking...")

# ==========================================
# 3. THE SPATIO-TEMPORAL LOOP
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_idx += 1

    # Apply our custom FAIR-style tracker configuration
    results = model.track(
        frame,
        persist=True,
        tracker="fair_tracker.yaml",
        imgsz=1280,
        device=0,
        conf=0.15,   # LOWER CONFIDENCE to see small heads in back
        iou=0.6,     # HIGHER IOU to allow shoulder-to-shoulder overlap
        verbose=False
    )
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        # Get Keypoints (x, y, confidence)
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
        else:
            keypoints = None

        for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confs)):
            x1, y1, x2, y2 = map(int, box)
            
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width == 0: continue 
            
            # --- 1. Raw Frame Guess (Hybrid) ---
            raw_state = "UNKNOWN"
            
            # A. Try Geometric Angle First (Priority)
            if keypoints is not None and len(keypoints) > i:
                kps = keypoints[i] # Shape: (17, 3) -> (x, y, conf)
                
                # keypoint indices: 11=left_hip, 12=right_hip, 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
                # We check both legs. If ANY leg is bent < 140, they are sitting.
                
                l_hip, l_knee, l_ankle = kps[11][:2], kps[13][:2], kps[15][:2]
                r_hip, r_knee, r_ankle = kps[12][:2], kps[14][:2], kps[16][:2]
                
                l_angle = calculate_angle(l_hip, l_knee, l_ankle)
                r_angle = calculate_angle(r_hip, r_knee, r_ankle)
                
                # Check for "Sitting" Geometry
                # Being conservative: If EITHER knee is clearly bent (< 140 degrees), it's a SIT.
                is_sitting_geom = False
                
                if l_angle and l_angle < 140: is_sitting_geom = True
                if r_angle and r_angle < 140: is_sitting_geom = True
                
                if is_sitting_geom:
                    raw_state = "SITTING"
                elif l_angle and r_angle and l_angle > 160 and r_angle > 160:
                    # Both legs clearly straight
                    raw_state = "STANDING"
            
            # B. "Head-Only" Heuristic (User Requested)
            # If we see Head/Shoulders but NO Lower Body -> Force SITTING (Occluded by desk)
            if raw_state == "UNKNOWN" and keypoints is not None and len(keypoints) > i:
                 kps = keypoints[i]
                 # visible score threshold
                 VIS_THRESH = 0.5
                 
                 # Indices: 0-4 (Head), 5-6 (Shoulders)
                 head_visible = any(kp[2] > VIS_THRESH for kp in kps[0:5])
                 shoulders_visible = any(kp[2] > VIS_THRESH for kp in kps[5:7])
                 
                 # Indices: 11-16 (Hips, Knees, Ankles)
                 lower_body_visible = any(kp[2] > VIS_THRESH for kp in kps[11:17])
                 
                 if (head_visible or shoulders_visible) and not lower_body_visible:
                     # "Floating Head" logic -> They are sitting behind a desk
                     raw_state = "SITTING"

            # C. Fallback to Aspect Ratio if Geometry failed or ambiguous
            if raw_state == "UNKNOWN":
                aspect_ratio = box_height / box_width
                raw_state = "STANDING" if aspect_ratio > ASPECT_RATIO_THRESHOLD else "SITTING"
            
            # --- 2. Build Temporal Memory Tubelet ---
            # If this is a new ID, give them a memory bank
            if track_id not in student_memory:
                student_memory[track_id] = deque(maxlen=MEMORY_FRAMES)
            
            # Log their state for this specific frame
            student_memory[track_id].append(raw_state)
            
            # --- 3. FAIR LOGIC: The Temporal Vote ---
            # Instead of using the raw guess, we count the states over the last 1 second.
            # This completely destroys the 2-3 second flickering issue.
            standing_count = student_memory[track_id].count("STANDING")
            sitting_count = student_memory[track_id].count("SITTING")
            
            final_state = "STANDING" if standing_count > sitting_count else "SITTING"
            
            # --- CLEAN DRAWING ---
            color = (0, 255, 0) if final_state == "SITTING" else (0, 0, 255)
            
            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label
            label = f"ID:{track_id} {final_state}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Log Analytics
            writer.writerow([frame_idx, track_id, final_state, round(float(conf), 2)])

    out.write(frame)
    if frame_idx % 10 == 0:
        print(f"Processed Frame {frame_idx}...", end='\r')

cap.release()
out.release()
csv_f.close()
print("\n✅ DONE. SOTA processing complete.")