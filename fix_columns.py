# fix_columns_back.py
import json

meta = "train/metadata.jsonl"
out = []
with open(meta, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        # if it has "image", convert to "file_name"
        if "image" in r and "file_name" not in r:
            r["file_name"] = r.pop("image")
        # keep "text" as is
        out.append(json.dumps(r, ensure_ascii=False))
with open(meta, "w", encoding="utf-8") as f:
    for s in out:
        f.write(s + "\n")
print("metadata.jsonl now uses keys: file_name, text")
