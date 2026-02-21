"""
llm_analyzer.py — Layer 3: Event CSV → Per-Student LLM Behavior Profiles
Usage:
    GEMINI_API_KEY=<key> python3 llm_analyzer.py \
        --events output/events_input2_20260221_182024.csv \
        --session "Math_10AM" [--out output/llm_profiles_<tag>.json]
"""
import os
import csv
import json
import argparse
import time
from collections import defaultdict
from datetime import datetime

# ─── LLM SETUP ────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def get_llm_client():
    """Return a callable: prompt_str -> response_str.
    Supports Gemini if API key set, else falls back to a stub for testing."""
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        def call(prompt):
            resp = model.generate_content(prompt)
            return resp.text
        return call
    else:
        # Offline stub — useful for testing the pipeline without an API key
        print("⚠️  GEMINI_API_KEY not set — using offline stub (mock LLM output)")
        def stub(prompt):
            import re
            # Extract student name from prompt
            name_match = re.search(r"Student name: (.+?)\n", prompt)
            name = name_match.group(1).strip() if name_match else "Unknown"
            return json.dumps({
                "behavior_summary": f"{name} was present during the session.",
                "attention_score": 0.70,
                "participation_level": "medium",
                "interpretation": "Insufficient API key to generate real analysis. Set GEMINI_API_KEY.",
                "flags": [],
                "confidence": 0.50,
                "recommendations_hint": ["Provide Gemini API key for real analysis."],
            })
        return stub

# ─── PROMPT TEMPLATE ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an educational behavior analyst reviewing computer vision data from a classroom session.
The data may be noisy. You must:
- Be conservative: if confidence < 0.6 for an event, note uncertainty
- Never fabricate behaviors not in the data
- Output ONLY valid JSON matching the schema below (no markdown fences, no extra text)

Output schema:
{
  "behavior_summary": "1-2 sentence plain English summary",
  "attention_score": <float 0.0-1.0>,
  "participation_level": "<high|medium|low>",
  "interpretation": "brief reasoning on notable events",
  "flags": ["<event_labels that are significant, if any>"],
  "confidence": <float 0.0-1.0 reflecting data quality>,
  "recommendations_hint": ["<1-3 short actionable hints for recommender>"]
}"""


def build_prompt(student_id, name, session, events):
    event_lines = []
    for e in events:
        if e["event"] == "session_summary":
            continue  # summarized separately below
        conf_note = "(low confidence)" if float(e.get("confidence", 1)) < 0.6 else ""
        dur = e.get("duration_sec", "?")
        event_lines.append(f"  - {e['event']}: {dur}s {conf_note}")

    # Session summary stats
    summaries = [e for e in events if e["event"] == "session_summary"]
    summary_txt = ""
    if summaries:
        s = summaries[0]
        summary_txt = (
            f"\nSession stats:\n"
            f"  Total tracked: {s.get('duration_sec', '?')}s\n"
            f"  Sitting ratio: {float(s.get('sitting_ratio', 0))*100:.0f}%\n"
            f"  Hand raises:   {s.get('hand_raise_count', 0)}\n"
            f"  Low visibility: {'yes' if str(s.get('low_visibility','0'))=='1' else 'no'}\n"
        )

    prompt = f"""{SYSTEM_PROMPT}

---
Student ID: {student_id}
Student name: {name}
Session: {session}

Detected events:
{chr(10).join(event_lines) if event_lines else "  (no specific events beyond presence)"}
{summary_txt}
Analyze this student's behavior and respond with JSON only."""
    return prompt


def analyze_all(events_path, session, llm_call):
    rows = []
    with open(events_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Group by student_id
    by_student = defaultdict(list)
    for row in rows:
        by_student[row["student_id"]].append(row)

    print(f"👥  Analyzing {len(by_student)} students via LLM…\n")
    profiles = []

    for student_id, student_events in sorted(by_student.items()):
        # Resolve name
        name_counts = defaultdict(int)
        for e in student_events:
            n = e.get("name", "Unknown")
            if n not in ("Unknown", ""):
                name_counts[n] += 1
        name = max(name_counts, key=name_counts.get) if name_counts else "Unknown"

        print(f"  🔍  {student_id} ({name}) — {len(student_events)} events…", end=" ", flush=True)

        prompt = build_prompt(student_id, name, session, student_events)

        try:
            raw = llm_call(prompt).strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            profile = json.loads(raw)
        except Exception as ex:
            print(f"⚠️  parse error: {ex}")
            profile = {
                "behavior_summary": "Analysis failed.",
                "attention_score": 0.5,
                "participation_level": "unknown",
                "interpretation": str(ex),
                "flags": [],
                "confidence": 0.0,
                "recommendations_hint": [],
            }

        profile["student_id"] = student_id
        profile["name"]       = name
        profile["session"]    = session
        profiles.append(profile)
        print("✅")

        # Rate limit — Gemini free tier: 15 req/min
        time.sleep(2)

    return profiles


def main():
    parser = argparse.ArgumentParser(description="Layer 3: Event CSV → LLM Profiles JSON")
    parser.add_argument("--events",  required=True, help="Path to events CSV from aggregator.py")
    parser.add_argument("--session", default="Session_1", help="Session label")
    parser.add_argument("--out",     default=None)
    args = parser.parse_args()

    # Auto-name
    base     = os.path.splitext(args.events)[0].replace("events_", "llm_profiles_")
    out_path = args.out or (base + ".json")

    llm_call = get_llm_client()
    profiles = analyze_all(args.events, args.session, llm_call)

    with open(out_path, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"\n✅  LLM profiles saved → {out_path}  ({len(profiles)} students)")
    return out_path


if __name__ == "__main__":
    main()
