"""
recommender.py — Layer 4: LLM Profiles → App Recommendations
Usage:
    python3 recommender.py --profiles output/llm_profiles_<tag>.json [--out output/recommendations_<tag>.json]

Rule-based layer. LLM output is advisory input only.
No LLM calls are made here.
"""
import os
import json
import argparse
from datetime import datetime

# ─── THRESHOLDS (tune these per deployment) ───────────────────────────────
ATTENTION_LOW_THRESH        = 0.55   # below → focus nudge
ATTENTION_HIGH_THRESH       = 0.80   # above → positive reinforcement
MIN_CONFIDENCE_FOR_ACTION   = 0.50   # below → "insufficient data", no nudge sent
TEACHER_ALERT_THRESH        = 0.45   # attention very low → alert teacher

# ─── MESSAGE TEMPLATES (positive framing, no surveillance language) ────────
MSG_POSITIVE = (
    "Great session today! You were actively engaged and your participation "
    "stood out. Keep it up — this kind of involvement makes a real difference."
)
MSG_FOCUS = (
    "You showed up today and that matters. There were a few moments where "
    "staying a bit more focused could help you get even more out of the class. "
    "Small improvements add up — you've got this."
)
MSG_QUESTION = (
    "Asking questions shows real intellectual curiosity. Keep that energy — "
    "it helps you and the whole class learn better."
)
MSG_NEUTRAL = (
    "Steady session today. Consistent attendance and attention are the "
    "foundation of great learning. Keep showing up."
)
MSG_INSUFFICIENT = (
    "We didn't have enough data to generate personalized feedback for this "
    "session. Check back after your next class."
)


def recommend(profile: dict) -> dict:
    """Generate one recommendation record from one LLM profile."""
    student_id   = profile.get("student_id", "?")
    name         = profile.get("name", "Unknown")
    session      = profile.get("session", "Session")
    attention    = float(profile.get("attention_score", 0.5))
    participation= profile.get("participation_level", "medium")
    data_conf    = float(profile.get("confidence", 0.0))
    flags        = profile.get("flags", [])
    hints        = profile.get("recommendations_hint", [])

    # Insufficient data guard — do not nudge if we can't trust the data
    if data_conf < MIN_CONFIDENCE_FOR_ACTION:
        return {
            "student_id":    student_id,
            "name":          name,
            "session":       session,
            "nudge_type":    "none",
            "app_message":   MSG_INSUFFICIENT,
            "teacher_alert": False,
            "data_quality":  "insufficient",
            "attention_score": attention,
            "participation":   participation,
            "llm_flags":     flags,
            "llm_hints":     hints,
        }

    # ── Determine nudge type ────────────────────────────────────────
    has_question = "question_intent" in flags
    has_active   = "active_participation" in flags

    if attention >= ATTENTION_HIGH_THRESH or participation == "high":
        nudge_type = "positive"
        message    = MSG_POSITIVE
        if has_question:
            message = MSG_QUESTION + "\n\n" + MSG_POSITIVE
    elif attention < ATTENTION_LOW_THRESH:
        nudge_type = "focus"
        message    = MSG_FOCUS
    else:
        nudge_type = "neutral"
        message    = MSG_NEUTRAL
        if has_question:
            message = MSG_QUESTION + "\n\n" + MSG_NEUTRAL

    # ── Teacher alert ───────────────────────────────────────────────
    teacher_alert = (attention < TEACHER_ALERT_THRESH and data_conf >= MIN_CONFIDENCE_FOR_ACTION)

    return {
        "student_id":    student_id,
        "name":          name,
        "session":       session,
        "nudge_type":    nudge_type,
        "app_message":   message,
        "teacher_alert": teacher_alert,
        "data_quality":  "sufficient",
        "attention_score": round(attention, 3),
        "participation":   participation,
        "llm_flags":     flags,
        "llm_hints":     hints,
    }


def main():
    parser = argparse.ArgumentParser(description="Layer 4: LLM Profiles → App Recommendations")
    parser.add_argument("--profiles", required=True, help="LLM profiles JSON from llm_analyzer.py")
    parser.add_argument("--out",      default=None)
    args = parser.parse_args()

    if not os.path.exists(args.profiles):
        raise FileNotFoundError(f"Profiles not found: {args.profiles}")

    base     = os.path.splitext(args.profiles)[0].replace("llm_profiles_", "recommendations_")
    out_path = args.out or (base + ".json")

    with open(args.profiles) as f:
        profiles = json.load(f)

    print(f"📋  Generating recommendations for {len(profiles)} students…")
    recommendations = [recommend(p) for p in profiles]

    # Print summary
    nudge_counts = {}
    alerts = 0
    for r in recommendations:
        nudge_counts[r["nudge_type"]] = nudge_counts.get(r["nudge_type"], 0) + 1
        if r["teacher_alert"]:
            alerts += 1

    print(f"\n📊  Nudge breakdown:")
    for ntype, cnt in sorted(nudge_counts.items()):
        print(f"    {ntype:<15} {cnt}")
    if alerts:
        print(f"\n🚨  Teacher alerts: {alerts}")

    with open(out_path, "w") as f:
        json.dump(recommendations, f, indent=2)

    print(f"\n✅  Recommendations saved → {out_path}")
    return out_path


if __name__ == "__main__":
    main()
