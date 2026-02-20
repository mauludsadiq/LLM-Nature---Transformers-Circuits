from pathlib import Path

p = Path("llm_nature") / "journey.py"
if not p.exists():
    raise SystemExit("llm_nature/journey.py not found")

s = p.read_text(encoding="utf-8")

need_imports = [
    "from . import stage11_universal_features as stage11_universal_features",
    "from . import stage12_real_world_evidence as stage12_real_world_evidence",
    "from . import stage13_truth_geometry as stage13_truth_geometry",
    "from . import stage14_universal_steering as stage14_universal_steering",
    "from . import stage15_complete_picture as stage15_complete_picture",
]

for line in need_imports:
    if line not in s:
        s = line + "\n" + s

added = False
candidates = [
    "stage10_gpt2_pipeline",
    "stage10",
]
for token in candidates:
    if token in s:
        idx = s.rfind(token)
        tail = s[idx : idx + 400]
        if "]" in tail:
            j = idx + tail.index("]")
            block = "\n    stage11_universal_features,\n    stage12_real_world_evidence,\n    stage13_truth_geometry,\n    stage14_universal_steering,\n    stage15_complete_picture,\n"
            if "stage11_universal_features" not in s:
                s = s[:j] + block + s[j:]
                added = True
        break

if not added:
    p.write_text(s, encoding="utf-8")
    raise SystemExit("Patched imports, but could not locate a stage list to append. Add stages manually in journey.py.")

p.write_text(s, encoding="utf-8")
print("Patched llm_nature/journey.py")
