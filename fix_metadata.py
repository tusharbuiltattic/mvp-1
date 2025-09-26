# fix_metadata.py
# pip install tqdm
import os, json, glob
from tqdm import tqdm
from pathlib import PurePosixPath

ROOT = "train"
IMG_DIR = os.path.join(ROOT, "images")
META = os.path.join(ROOT, "metadata.jsonl")

# Expected relative names (POSIX)
expected = set(str(PurePosixPath("images") / os.path.basename(p))
               for p in glob.glob(os.path.join(IMG_DIR, "*.png")))
print("images on disk:", len(expected))

# Count lines for tqdm total
with open(META, "r", encoding="utf-8") as f:
    total = sum(1 for _ in f)

new_lines = []
with open(META, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total, desc="Rewriting metadata", unit="rec"):
        if not line.strip():
            continue
        r = json.loads(line)
        rel = str(PurePosixPath("images") / os.path.basename(r["file_name"]))
        r["file_name"] = rel
        new_lines.append(json.dumps(r, ensure_ascii=False))

have = set(json.loads(l)["file_name"] for l in new_lines)
missing = expected - have
extra = have - expected
print(f"records: {len(new_lines)} | missing files: {len(missing)} | extra records: {len(extra)}")

with open(META, "w", encoding="utf-8") as f:
    for l in tqdm(new_lines, desc="Writing file", unit="rec"):
        f.write(l + "\n")

print("metadata.jsonl rewritten with relative POSIX paths.")
