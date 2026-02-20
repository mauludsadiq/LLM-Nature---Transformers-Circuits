import os, json, hashlib, time
from dataclasses import dataclass
from typing import Any

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_text(path: str, s: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def banner(title: str) -> str:
    line = "=" * 80
    return f"{line}\n{title}\n{line}\n"

def friend_sentence(stage: int, claim: str, model_try: str, correction: str) -> str:
    return (
        f"🧑‍🤝‍🧑 Friend-finishes-your-sentence check (Stage {stage})\n"
        f"   You:      {claim}\n"
        f"   Friend:   {model_try}\n"
        f"   Reality:  {correction}\n"
    )

@dataclass
class RunContext:
    out_dir: str
    seed: int = 42
