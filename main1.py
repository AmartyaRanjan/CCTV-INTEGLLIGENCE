import os
import cv2
import csv
import math
import torch
import zipfile
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# ==============================
# CONFIG
# ==============================
VIDEO_PATH    = "input2.mp4"
REF_FACES_DIR = "Reference_faces"
OUTPUT_DIR    = "output"

# YOLO11x-POSE — gives person bboxes + 17 COCO keypoints (requires Ultralytics >= 8.3)
YOLO_MODEL  = "yolo11x-pose.pt"
IMG_SIZE    = 1280
CONF_THRESH = 0.25            # low threshold catches people in wide / low-quality shots

# Face recognition
FACE_MATCH_THRESH = 1.10      # cosine-like L2 distance; raise if too few matches
FACE_MAX_ATTEMPTS = 20        # retry this many frames before permanently caching Unknown

# Posture smoothing
POSTURE_FRAMES = 20           # majority-vote window (frames)

# Keypoint visibility threshold (COCO confidence score)
KP_VIS_THRESH = 0.35

# Knee-angle thresholds
KNEE_SIT_ANGLE   = 140        # below this → leg is bent → SITTING
KNEE_STAND_ANGLE = 155        # above this (both legs) → leg straight → candidate STANDING

# Aspect-ratio fallback (when pose has no useful keypoints)
# A standing person at normal distance has bh/bw > ~1.8
ASPECT_STAND_RATIO = 1.9

os.makedirs(OUTPUT_DIR, exist_ok=True)

# COCO keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW,    R_ELBOW    = 7, 8
L_WRIST,    R_WRIST    = 9, 10
L_HIP,      R_HIP      = 11, 12
L_KNEE,     R_KNEE     = 13, 14
L_ANKLE,    R_ANKLE    = 15, 16

# ==============================
# DEVICE
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ==============================
# LOAD MODELS
# ==============================
print("📦 Loading YOLO11x-Pose…")
detector = YOLO(YOLO_MODEL)

print("📦 Loading MTCNN + FaceNet…")
mtcnn   = MTCNN(keep_all=False, device=device, post_process=True,
                min_face_size=20, thresholds=[0.6, 0.7, 0.7])
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# ==============================
# LOAD REFERENCE FACES
# ==============================
known_embeddings: dict[str, torch.Tensor] = {}
print("📚 Loading reference faces…")
for file in sorted(os.listdir(REF_FACES_DIR)):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    name = os.path.splitext(file)[0]
    try:
        img  = Image.open(os.path.join(REF_FACES_DIR, file)).convert("RGB")
        face = mtcnn(img)
        if face is None:
            img  = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
            face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                emb = facenet(face.unsqueeze(0).to(device)).detach().cpu()
            known_embeddings[name] = emb
            print(f"  ✅  {name}")
        else:
            print(f"  ⚠️  {file} — no face detected, skipped")
    except Exception as ex:
        print(f"  ❌  {file}: {ex}")

print(f"\n👥 Enrolled {len(known_embeddings)} faces: {list(known_embeddings.keys())}\n")

# ==============================
# VIDEO IO
# ==============================
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Input video not found: {VIDEO_PATH}")

cap          = cv2.VideoCapture(VIDEO_PATH)
W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS          = cap.get(cv2.CAP_PROP_FPS)
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"📹 {W}×{H} @ {FPS:.1f} fps  |  {TOTAL_FRAMES} frames")

_stem      = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_run_tag   = f"{_stem}_{_timestamp}"   # e.g. input2_20260221_181900
out_video  = os.path.join(OUTPUT_DIR, f"output_video_{_run_tag}.mp4")
csv_path   = os.path.join(OUTPUT_DIR, f"analytics_{_run_tag}.csv")
zip_path   = os.path.join(OUTPUT_DIR, f"results_{_run_tag}.zip")

out      = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
csv_file = open(csv_path, "w", newline="")
writer   = csv.writer(csv_file)

# Enriched schema — designed to give an LLM as many signals as possible
writer.writerow([
    "frame", "track_id", "name",
    "posture",          # SITTING | STANDING
    "posture_method",   # pose_keypoint | aspect_ratio | occluded
    "knee_angle_deg",   # best knee angle (float), -1 if unavailable
    "knees_visible",    # 1 or 0
    "hand_raised",      # 1 if wrist above shoulder (rough raise-hand signal), 0 otherwise
    "det_conf",         # YOLO detection confidence
    "x1","y1","x2","y2",        # bounding box pixels
    "bbox_width","bbox_height",  # convenience
    "aspect_ratio",              # height/width
])

# ==============================
# PER-TRACK STATE
# ==============================
posture_hist  : dict[int, deque] = {}
name_cache    : dict[int, str]   = {}
face_attempts : dict[int, int]   = {}

# ==============================
# HELPERS
# ==============================
def angle_at_joint(p1, p2, p3):
    """Interior angle (degrees) at p2 formed by p1→p2←p3. Returns None if any point missing."""
    if p1 is None or p2 is None or p3 is None:
        return None
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 * n2 < 1e-6:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def kp_xy(kps, idx):
    """Return (x,y) of keypoint idx if confidence >= threshold, else None."""
    if kps is None or len(kps) <= idx:
        return None
    x, y, c = kps[idx]
    return (float(x), float(y)) if c >= KP_VIS_THRESH else None


def classify_posture(kps, bh, bw):
    """
    Returns (posture_str, method_str, best_angle_deg, knees_visible_bool).

    Decision rules (conservative — prefer SITTING on ambiguity):
      1. Pose-keypoint path (when keypoints available):
         a. If BOTH knees invisible (occluded by desk/crowd) → SITTING
         b. If any knee angle < KNEE_SIT_ANGLE              → SITTING
         c. If both knees straight (> KNEE_STAND_ANGLE) AND
            person is tall (bh/bw > ASPECT_STAND_RATIO)    → STANDING
         d. Otherwise                                        → SITTING
      2. Fallback (no useful keypoints):
         - Aspect ratio bh/bw > ASPECT_STAND_RATIO          → STANDING
         - Otherwise                                         → SITTING
    """
    l_hip  = kp_xy(kps, L_HIP);   r_hip  = kp_xy(kps, R_HIP)
    l_knee = kp_xy(kps, L_KNEE);  r_knee = kp_xy(kps, R_KNEE)
    l_ank  = kp_xy(kps, L_ANKLE); r_ank  = kp_xy(kps, R_ANKLE)

    knees_visible = (l_knee is not None) or (r_knee is not None)

    if not knees_visible:
        # Knees hidden → person is behind a desk → SITTING
        return "SITTING", "occluded", -1, False

    l_angle = angle_at_joint(l_hip, l_knee, l_ank) if (l_hip and l_knee and l_ank) else None
    r_angle = angle_at_joint(r_hip, r_knee, r_ank) if (r_hip and r_knee and r_ank) else None

    # Pick the most informative angle
    valid_angles = [a for a in [l_angle, r_angle] if a is not None]
    best_angle   = min(valid_angles) if valid_angles else -1   # min = most bent

    if valid_angles:
        # Any bent knee → sitting
        if min(valid_angles) < KNEE_SIT_ANGLE:
            return "SITTING", "pose_keypoint", best_angle, True
        # Both legs straight AND tall bbox → standing
        if (len(valid_angles) == 2 and
                l_angle > KNEE_STAND_ANGLE and r_angle > KNEE_STAND_ANGLE and
                bh / max(bw, 1) > ASPECT_STAND_RATIO):
            return "STANDING", "pose_keypoint", best_angle, True
        # One visible straight leg + tall bbox → standing
        if (len(valid_angles) == 1 and
                valid_angles[0] > KNEE_STAND_ANGLE and
                bh / max(bw, 1) > ASPECT_STAND_RATIO):
            return "STANDING", "pose_keypoint", best_angle, True
        # Default with keypoints → sitting
        return "SITTING", "pose_keypoint", best_angle, True

    # No angle computable (knee visible but hips/ankles missing)
    ar = bh / max(bw, 1)
    if ar > ASPECT_STAND_RATIO:
        return "STANDING", "aspect_ratio", -1, True
    return "SITTING", "aspect_ratio", -1, True


def hand_raised(kps):
    """Return 1 if either wrist is above the shoulder on the same side."""
    l_sh = kp_xy(kps, L_SHOULDER); r_sh = kp_xy(kps, R_SHOULDER)
    l_wr = kp_xy(kps, L_WRIST);    r_wr = kp_xy(kps, R_WRIST)
    # In image coords Y increases downward, so wrist above shoulder means wrist_y < shoulder_y
    if l_sh and l_wr and l_wr[1] < l_sh[1] - 30:
        return 1
    if r_sh and r_wr and r_wr[1] < r_sh[1] - 30:
        return 1
    return 0


def match_face(emb):
    best_name, best_dist = "Unknown", float("inf")
    for name, ref_emb in known_embeddings.items():
        dist = torch.norm(emb - ref_emb).item()
        if dist < best_dist:
            best_dist, best_name = dist, name
    return ("Unknown" if best_dist >= FACE_MATCH_THRESH else best_name), best_dist


def try_recognize(frame, x1, y1, x2, y2):
    bh = y2 - y1
    for frac in [0.45, 0.55, 0.35]:
        crop_h = int(bh * frac)
        pad    = int((x2 - x1) * 0.05)
        fx1    = max(0, x1 - pad);  fx2 = min(W, x2 + pad)
        crop   = frame[y1: y1 + crop_h, fx1: fx2]
        if crop.size == 0:
            continue
        pil  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        face = mtcnn(pil)
        if face is not None:
            with torch.no_grad():
                emb = facenet(face.unsqueeze(0).to(device)).detach().cpu()
            return match_face(emb)[0]
    return None


def draw_label(img, text, pt, color):
    font = cv2.FONT_HERSHEY_SIMPLEX; scale = 0.55; thick = 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = pt
    cv2.rectangle(img, (x, y - th - bl - 4), (x + tw + 4, y + 2), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, color, thick, cv2.LINE_AA)

# ==============================
# MAIN PROCESSING LOOP
# ==============================
print("🎥 Processing…\n")
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = detector.track(
        frame,
        imgsz   = IMG_SIZE,
        conf    = CONF_THRESH,
        persist = True,
        tracker = "botsort.yaml",
        classes = [0],          # persons only
        verbose = False
    )

    r = results[0]
    if r.boxes.id is None:
        out.write(frame)
        continue

    boxes   = r.boxes.xyxy.cpu().numpy()
    ids     = r.boxes.id.cpu().numpy().astype(int)
    confs   = r.boxes.conf.cpu().numpy()
    kp_data = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None

    for i, (box, tid, conf) in enumerate(zip(boxes, ids, confs)):
        x1, y1, x2, y2 = map(int, box)
        bh = max(1, y2 - y1);  bw = max(1, x2 - x1)
        ar = round(bh / bw, 2)

        kps = kp_data[i] if (kp_data is not None and i < len(kp_data)) else None

        # ── INIT ─────────────────────────────────────
        if tid not in name_cache:
            posture_hist[tid]  = deque(maxlen=POSTURE_FRAMES)
            name_cache[tid]    = None
            face_attempts[tid] = 0

        # ── FACE RECOGNITION ────────────────────────
        if name_cache[tid] is None:
            face_attempts[tid] += 1
            result = try_recognize(frame, x1, y1, x2, y2)
            if result is not None:
                name_cache[tid] = result
            elif face_attempts[tid] >= FACE_MAX_ATTEMPTS:
                name_cache[tid] = "Unknown"

        display_name = name_cache[tid] if name_cache[tid] is not None else "…"
        csv_name     = name_cache[tid] if name_cache[tid] is not None else "Unknown"

        # ── POSTURE ──────────────────────────────────
        raw_posture, method, knee_angle, knees_vis = classify_posture(kps, bh, bw)
        posture_hist[tid].append(raw_posture)
        final_posture = max(set(posture_hist[tid]), key=posture_hist[tid].count)

        # ── HAND RAISED ──────────────────────────────
        h_raised = hand_raised(kps)

        # ── DRAW ─────────────────────────────────────
        color = (255, 80, 0) if final_posture == "STANDING" else (0, 220, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        posture_icon = "▲" if final_posture == "STANDING" else "▼"
        hand_icon    = " ✋" if h_raised else ""
        label        = f"ID{tid}: {display_name} {posture_icon}{hand_icon}"
        draw_label(frame, label, (x1, y1 - 2), color)

        # ── CSV (enriched for LLM pipeline) ──────────
        writer.writerow([
            frame_idx, tid, csv_name,
            final_posture, method,
            round(knee_angle, 1),
            int(knees_vis),
            h_raised,
            round(float(conf), 3),
            x1, y1, x2, y2,
            bw, bh,
            ar,
        ])

    out.write(frame)

    if frame_idx % 30 == 0:
        pct = frame_idx / max(TOTAL_FRAMES, 1) * 100
        print(f"  Frame {frame_idx}/{TOTAL_FRAMES}  ({pct:.1f}%)", end="\r")

# ==============================
# CLEANUP & ZIP
# ==============================
cap.release()
out.release()
csv_file.close()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(out_video, arcname=f"output_video_{_run_tag}.mp4")
    zf.write(csv_path,  arcname=f"analytics_{_run_tag}.csv")

print(f"\n\n✅  DONE")
print(f"📹  Video  → {out_video}")
print(f"📊  CSV    → {csv_path}")
print(f"📦  Zip    → {zip_path}")
