# ensure_metadata_columns.py
import os, json, glob
from pathlib import PurePosixPath

ROOT = "train"
IMG_DIR = os.path.join(ROOT, "images")
META = os.path.join(ROOT, "metadata.jsonl")

# Map of existing files for fast lookup
disk = {os.path.basename(p): p for p in glob.glob(os.path.join(IMG_DIR, "*.png"))}

fixed = []
with open(META, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)

        # move any 'image' key back to 'file_name'
        if "file_name" not in r and "image" in r:
            r["file_name"] = r.pop("image")

        # enforce relative POSIX path under images/
        base = os.path.basename(r["file_name"])
        rel = str(PurePosixPath("images") / base)
        r["file_name"] = rel

        # keep caption column name 'text'
        if "text" not in r:
            raise ValueError("Missing 'text' in a record.")

        # verify file exists on disk
        if base not in disk:
            raise FileNotFoundError(f"Missing image on disk: {rel}")

        fixed.append(json.dumps(r, ensure_ascii=False))

with open(META, "w", encoding="utf-8") as f:
    for s in fixed:
        f.write(s + "\n")

print(f"OK: {len(fixed)} records with keys: file_name, text; paths like images/xxx.png")
