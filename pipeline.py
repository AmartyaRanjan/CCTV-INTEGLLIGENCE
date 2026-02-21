"""
pipeline.py — Layer 5: Master Pipeline Runner (Layers 2 → 3 → 4)
Usage:
    GEMINI_API_KEY=<key> python3 pipeline.py \
        --csv output/analytics_input2_20260221_182024.csv \
        --fps 25 \
        --session "Math_10AM"

All intermediate and final outputs are saved to the output/ directory
with the same timestamp tag as the input CSV.
"""
import os
import sys
import json
import argparse
from datetime import datetime

# ─── Import our layers ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import aggregator
import llm_analyzer
import recommender

DIVIDER = "─" * 60


def run_pipeline(csv_path: str, fps: float, session: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.dirname(csv_path)  # same folder as input CSV

    # Derive a shared tag from the input filename
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    # e.g. analytics_input2_20260221_182024 → strip "analytics_" prefix
    run_tag   = base_name.replace("analytics_", "") or timestamp

    events_path      = os.path.join(out_dir, f"events_{run_tag}.csv")
    profiles_path    = os.path.join(out_dir, f"llm_profiles_{run_tag}.json")
    reco_path        = os.path.join(out_dir, f"recommendations_{run_tag}.json")
    summary_path     = os.path.join(out_dir, f"pipeline_summary_{run_tag}.txt")

    print(f"\n{'═'*60}")
    print(f"  🏫  Classroom Analytics Pipeline")
    print(f"  Session  : {session}")
    print(f"  Input CSV: {csv_path}")
    print(f"  Run tag  : {run_tag}")
    print(f"{'═'*60}\n")

    # ── LAYER 2: AGGREGATION ──────────────────────────────────────
    print(f"{DIVIDER}")
    print("  LAYER 2 — Event Aggregation")
    print(f"{DIVIDER}")
    rows   = aggregator.load_csv(csv_path)
    events = aggregator.aggregate(rows, fps=fps, session=session)
    aggregator.save_events(events, events_path)

    n_students = len(set(e["student_id"] for e in events))
    n_events   = len([e for e in events if e["event"] != "session_summary"])

    print(f"  👥  {n_students} students  |  {n_events} behavioral events\n")

    # ── LAYER 3: LLM ANALYSIS ─────────────────────────────────────
    print(f"{DIVIDER}")
    print("  LAYER 3 — LLM Behavior Reasoning")
    print(f"{DIVIDER}")
    llm_call = llm_analyzer.get_llm_client()
    profiles = llm_analyzer.analyze_all(events_path, session, llm_call)

    with open(profiles_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"  ✅  {len(profiles)} profiles → {profiles_path}\n")

    # ── LAYER 4: RECOMMENDER ──────────────────────────────────────
    print(f"{DIVIDER}")
    print("  LAYER 4 — Recommender")
    print(f"{DIVIDER}")
    recs = [recommender.recommend(p) for p in profiles]

    nudge_counts = {}
    alerts = []
    for r in recs:
        nudge_counts[r["nudge_type"]] = nudge_counts.get(r["nudge_type"], 0) + 1
        if r["teacher_alert"]:
            alerts.append(r["name"])

    for nt, cnt in sorted(nudge_counts.items()):
        print(f"  {nt:<15} {cnt} students")
    if alerts:
        print(f"\n  🚨  Teacher alerts: {', '.join(alerts)}")

    with open(reco_path, "w") as f:
        json.dump(recs, f, indent=2)
    print(f"\n  ✅  Recommendations → {reco_path}\n")

    # ── PIPELINE SUMMARY (human-readable) ────────────────────────
    print(f"{DIVIDER}")
    print("  PIPELINE SUMMARY")
    print(f"{DIVIDER}")

    lines = []
    lines.append(f"Classroom Analytics Report")
    lines.append(f"Session : {session}")
    lines.append(f"Run     : {run_tag}")
    lines.append(f"Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)
    lines.append(f"Students tracked : {n_students}")
    lines.append(f"Behavioral events: {n_events}")
    lines.append("")
    lines.append("Per-Student Summary:")
    lines.append("-" * 60)

    for p, r in zip(profiles, recs):
        name    = p.get("name", p.get("student_id"))
        attn    = p.get("attention_score", "?")
        part    = p.get("participation_level", "?")
        summary = p.get("behavior_summary", "")
        nudge   = r["nudge_type"]
        quality = r["data_quality"]
        alert   = " 🚨 TEACHER ALERT" if r["teacher_alert"] else ""

        lines.append(f"\n{name}  [{p['student_id']}]")
        lines.append(f"  Attention       : {attn}")
        lines.append(f"  Participation   : {part}")
        lines.append(f"  Data quality    : {quality}")
        lines.append(f"  Nudge type      : {nudge}{alert}")
        lines.append(f"  Summary         : {summary}")
        lines.append(f"  App message     : {r['app_message'][:120]}…" if len(r['app_message']) > 120 else f"  App message     : {r['app_message']}")

    lines.append("\n" + "=" * 60)
    lines.append("Output files:")
    lines.append(f"  Events CSV     : {events_path}")
    lines.append(f"  LLM Profiles   : {profiles_path}")
    lines.append(f"  Recommendations: {reco_path}")
    lines.append(f"  This summary   : {summary_path}")

    summary_text = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\n✅  Pipeline complete. Summary → {summary_path}")

    return {
        "events":       events_path,
        "profiles":     profiles_path,
        "recommendations": reco_path,
        "summary":      summary_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Full Classroom Analytics Pipeline (Layers 2–4)")
    parser.add_argument("--csv",     required=True,        help="analytics_*.csv from main1.py")
    parser.add_argument("--fps",     type=float, default=25.0)
    parser.add_argument("--session", default="Session_1",  help="e.g. Math_10AM")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    run_pipeline(args.csv, args.fps, args.session)


if __name__ == "__main__":
    main()
