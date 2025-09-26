# caption_with_hf.py
# Captions data/images/*.png -> captions.json using BLIP (HF)

import os, glob, json
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

IMG_DIR = "data/images"
OUT_PATH = "captions.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    if not paths:
        print(f"No images in {IMG_DIR}. Run Step 2 first.")
        return

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)

    caps = {}
    for i, p in enumerate(paths, 1):
        try:
            img = Image.open(p).convert("RGB")
            inputs = processor(img, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_new_tokens=30)
            cap = processor.decode(out[0], skip_special_tokens=True)
        except Exception:
            cap = "floor plan, architectural diagram"
        caps[p] = cap
        if i % 100 == 0:
            print(f"captioned {i}/{len(paths)}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(caps, f, indent=2, ensure_ascii=False)
    print(f"Done. Captions: {len(caps)} -> {OUT_PATH}")

if __name__ == "__main__":
    main()
