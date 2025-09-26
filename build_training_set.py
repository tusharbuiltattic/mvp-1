# build_training_set.py
# Input: captions.json, data/images/*.png
# Output: train/images/*.png, train/metadata.jsonl

import os, json, shutil, random, glob

CAP_PATH = "captions.json"
SRC_DIR  = "data/images"
DST_IMG  = "train/images"
DST_META = "train/metadata.jsonl"

os.makedirs(DST_IMG, exist_ok=True)

# Arch-AI prompt builder from core principles
def arch_ai_prompt(choices: dict, base_caption: str):
    t = [
        base_caption,
        "detailed architectural floor plan, top-down",
        "clean lines, clear circulation axes, readable labels",  # Circulation (3)
    ]
    c = choices.get("Climate")
    if c in ["Hot & Dry", "Hot & Humid", "Tropical"]:
        t += ["east/south openings with shading", "cross-ventilation", "verandas"]  # Openings (5)
    if c in ["Cold", "Temperate"]:
        t += ["south-facing glazing", "interior thermal mass"]  # Sustainability (7)
    if choices.get("Interior Plan") == "Open":
        t += ["open-plan living dining kitchen"]  # Zoning (2)
    if choices.get("Style"):
        t.append(f"{choices['Style'].lower()} style")
    if choices.get("Roof Type"):
        t.append(f"{choices['Roof Type'].lower()} roof")
    feats = choices.get("Additional Features") or []
    if feats:
        t.append(", ".join(f.lower() for f in feats))
    return ", ".join([x for x in t if x])

SCHEMA = {
    "Style": ["Modernism","Minimalism","International Style","Bauhaus"],
    "Climate": ["Temperate","Hot & Dry","Cold","Tropical"],
    "Roof Type": ["Flat","Gable","Hip"],
    "Interior Plan": ["Open","Linear","Cluster"],
    "Additional Features": ["Verandas","Balconies","Courtyards","Atriums","Terraces"],
}

def main():
    if not os.path.exists(CAP_PATH):
        print("captions.json not found. Run Step 3 first.")
        return

    caps = json.load(open(CAP_PATH, "r", encoding="utf-8"))
    src_paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.png")))
    if not src_paths:
        print("No images in data/images. Run Step 2 first.")
        return

    recs = []
    for i, src in enumerate(src_paths):
        base_cap = caps.get(src, "floor plan, architectural diagram")
        choices = {
            "Style": random.choice(SCHEMA["Style"]),
            "Climate": random.choice(SCHEMA["Climate"]),
            "Roof Type": random.choice(SCHEMA["Roof Type"]),
            "Interior Plan": random.choice(SCHEMA["Interior Plan"]),
            "Additional Features": random.sample(SCHEMA["Additional Features"], k=random.randint(0,2)),
        }
        prompt = arch_ai_prompt(choices, base_cap)
        dst = os.path.join(DST_IMG, os.path.basename(src))
        shutil.copy2(src, dst)
        recs.append({"file_name": dst, "text": prompt})

    os.makedirs(os.path.dirname(DST_META), exist_ok=True)
    with open(DST_META, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    print("Wrote", len(recs), "records")
    print("Images dir:", DST_IMG)
    print("Metadata:", DST_META)

if __name__ == "__main__":
    main()
