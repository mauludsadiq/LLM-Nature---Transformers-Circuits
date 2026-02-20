import argparse, os, json
from .util import sha256_file, write_json

def compute_manifest(out_dir: str) -> dict:
    files = {}
    for root, _, fnames in os.walk(out_dir):
        for fn in fnames:
            if fn.startswith("."):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, out_dir)
            files[rel] = "sha256:" + sha256_file(path)
    return {"out_dir": out_dir, "files": dict(sorted(files.items()))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = compute_manifest(a.out)
    write_json(os.path.join(a.out, "manifest.json"), man)
    print(json.dumps(man, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
