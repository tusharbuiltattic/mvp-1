# quick_check.py
import os, json, itertools
print("has dir:", os.path.isdir("train/images"))
print("num imgs:", len([f for f in os.listdir("train/images") if f.endswith(".png")]))
print("has meta:", os.path.isfile("train/metadata.jsonl"))
print("first 2 lines:")
with open("train/metadata.jsonl","r",encoding="utf-8") as f:
    for line in itertools.islice(f,2):
        j=json.loads(line); print(j.keys(), j["file_name"][:60], j["text"][:80])
