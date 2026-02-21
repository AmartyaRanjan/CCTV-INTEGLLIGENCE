"""
aggregator.py — Layer 2: Frame CSV → Semantic Event CSV
Usage:
    python3 aggregator.py --csv output/analytics_input2_20260221_182024.csv --fps 25 [--session "Math_10AM"]
"""
import os
import csv
import argparse
import json
from collections import defaultdict
from datetime import datetime

# ─── CONFIG ──────────────────────────────────────────────────────────────
MIN_HAND_RAISE_SEC    = 2.0   # hand raised for ≥ this → "question_intent"
MIN_STANDING_SEC      = 5.0   # standing for ≥ this    → "active_participation"
LOW_CONF_THRESH       = 0.45  # det_conf below this = low visibility
LOW_CONF_MAJORITY     = 0.6   # fraction of window that must be low-conf to flag
WINDOW_SEC            = 10.0  # time window for aggregation (seconds)

POSTURE_COL      = "posture"
HAND_RAISED_COL  = "hand_raised"
DET_CONF_COL     = "det_conf"
FRAME_COL        = "frame"
TID_COL          = "track_id"
NAME_COL         = "name"


def load_csv(path: str):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def frame_to_sec(frame: int, fps: float) -> float:
    return frame / fps


def detect_runs(sequence, value, fps, min_sec):
    """Find contiguous runs of `value` in sequence and return list of (start_frame, end_frame, duration_sec)."""
    runs = []
    start = None
    for frame, v in sequence:
        if v == value:
            if start is None:
                start = frame
            end = frame
        else:
            if start is not None:
                dur = frame_to_sec(end - start + 1, fps)
                if dur >= min_sec:
                    runs.append((start, end, dur))
                start = None
    if start is not None:
        dur = frame_to_sec(end - start + 1, fps)
        if dur >= min_sec:
            runs.append((start, end, dur))
    return runs


def aggregate(rows, fps: float, session: str):
    """
    Returns list of event dicts.
    """
    # Group rows by track_id
    by_track = defaultdict(list)
    for row in rows:
        by_track[int(row[TID_COL])].append(row)

    events = []

    for tid, track_rows in by_track.items():
        # Sort by frame
        track_rows.sort(key=lambda r: int(r[FRAME_COL]))

        # Resolve canonical name (most frequent non-Unknown)
        name_counter = defaultdict(int)
        for r in track_rows:
            n = r[NAME_COL]
            if n not in ("Unknown", "…", ""):
                name_counter[n] += 1
        name = max(name_counter, key=name_counter.get) if name_counter else "Unknown"

        frames = [(int(r[FRAME_COL]), r) for r in track_rows]
        first_frame = frames[0][0]
        last_frame  = frames[-1][0]

        # ── Compute per-frame sequences ──────────────────────────────
        posture_seq    = [(f, r[POSTURE_COL]) for f, r in frames]
        hand_seq       = [(f, int(r[HAND_RAISED_COL])) for f, r in frames]
        conf_vals      = [float(r[DET_CONF_COL]) for _, r in frames]
        mean_conf      = round(sum(conf_vals) / len(conf_vals), 3)

        low_conf_frac  = sum(1 for c in conf_vals if c < LOW_CONF_THRESH) / len(conf_vals)
        low_visibility = low_conf_frac > LOW_CONF_MAJORITY

        # ── HAND RAISE → question_intent ─────────────────────────────
        hand_runs = detect_runs([(f, v) for f, v in hand_seq], 1, fps, MIN_HAND_RAISE_SEC)
        for start_f, end_f, dur in hand_runs:
            events.append({
                "student_id":   f"ID_{tid}",
                "name":         name,
                "session":      session,
                "event":        "question_intent",
                "duration_sec": round(dur, 1),
                "confidence":   min(round(mean_conf + 0.05, 3), 1.0),
                "frame_start":  start_f,
                "frame_end":    end_f,
                "session_time_sec": round(frame_to_sec(start_f, fps), 1),
            })

        # ── STANDING RUNS → active_participation ─────────────────────
        stand_runs = detect_runs(posture_seq, "STANDING", fps, MIN_STANDING_SEC)
        for start_f, end_f, dur in stand_runs:
            events.append({
                "student_id":   f"ID_{tid}",
                "name":         name,
                "session":      session,
                "event":        "active_participation",
                "duration_sec": round(dur, 1),
                "confidence":   round(mean_conf, 3),
                "frame_start":  start_f,
                "frame_end":    end_f,
                "session_time_sec": round(frame_to_sec(start_f, fps), 1),
            })

        # ── OVERALL SITTING (passive attentive) ──────────────────────
        total_frames    = len(track_rows)
        sitting_frames  = sum(1 for _, r in frames if r[POSTURE_COL] == "SITTING")
        standing_frames = total_frames - sitting_frames
        sit_ratio       = sitting_frames / max(total_frames, 1)

        events.append({
            "student_id":   f"ID_{tid}",
            "name":         name,
            "session":      session,
            "event":        "session_summary",
            "duration_sec": round(frame_to_sec(last_frame - first_frame + 1, fps), 1),
            "confidence":   round(mean_conf, 3),
            "frame_start":  first_frame,
            "frame_end":    last_frame,
            "session_time_sec": round(frame_to_sec(first_frame, fps), 1),
            # Extra fields for session_summary row
            "sitting_ratio":   round(sit_ratio, 3),
            "standing_frames": standing_frames,
            "hand_raise_count": len(hand_runs),
            "low_visibility":  int(low_visibility),
        })

        # ── LOW VISIBILITY FLAG ───────────────────────────────────────
        if low_visibility:
            events.append({
                "student_id":   f"ID_{tid}",
                "name":         name,
                "session":      session,
                "event":        "low_visibility",
                "duration_sec": round(frame_to_sec(last_frame - first_frame + 1, fps), 1),
                "confidence":   round(mean_conf, 3),
                "frame_start":  first_frame,
                "frame_end":    last_frame,
                "session_time_sec": round(frame_to_sec(first_frame, fps), 1),
            })

    return events


def save_events(events, out_path):
    if not events:
        print("⚠️  No events generated.")
        return

    # Collect all keys across all events for dynamic CSV schema
    all_keys = []
    seen = set()
    for e in events:
        for k in e.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for e in events:
            writer.writerow({k: e.get(k, "") for k in all_keys})

    print(f"✅  Events saved → {out_path}  ({len(events)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Layer 2: Frame CSV → Event CSV")
    parser.add_argument("--csv",     required=True, help="Path to analytics CSV from main1.py")
    parser.add_argument("--fps",     type=float, default=25.0, help="Video FPS (default 25)")
    parser.add_argument("--session", default="Session_1", help="Session label e.g. Math_10AM")
    parser.add_argument("--out",     default=None, help="Output path (auto if not set)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    # Auto-name output alongside the input file
    base     = os.path.splitext(args.csv)[0].replace("analytics_", "events_")
    out_path = args.out or (base + ".csv")

    print(f"📂  Loading {args.csv} …")
    rows = load_csv(args.csv)
    print(f"    {len(rows)} frame rows, {len(set(r[TID_COL] for r in rows))} unique tracks")

    events = aggregate(rows, fps=args.fps, session=args.session)

    # Print summary
    event_types = {}
    for e in events:
        event_types[e["event"]] = event_types.get(e["event"], 0) + 1
    print(f"\n📊  Event breakdown:")
    for etype, cnt in sorted(event_types.items()):
        print(f"    {etype:<30} {cnt}")

    save_events(events, out_path)
    return out_path


if __name__ == "__main__":
    main()
