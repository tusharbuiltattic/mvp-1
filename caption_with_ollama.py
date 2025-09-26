# caption_with_ollama.py
# Requirements: Ollama running locally: `ollama serve` and model pulled: `ollama pull llava`
# Output: captions.json mapping image path -> caption

import os, glob, json, base64, time, requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11500")
MODEL = os.environ.get("OLLAMA_MODEL", "llava")  # change to llava:13b if you have VRAM
IMG_DIR = "data/images"
OUT_PATH = "captions.json"
BATCH_LIMIT = None  # set e.g. 500 for a quick pass

PROMPT = (
    "Describe this floor plan in one concise sentence. "
    "List the key rooms and their connections (e.g., living to dining, kitchen adjacency, bedroom locations)."
)

def ollama_generate_vision(img_path: str) -> str:
    b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": PROMPT, "images": [b64]},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"].strip()

def main():
    # sanity checks
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        tags = [m["name"] for m in r.json().get("models", [])]
        if MODEL not in tags:
            print(f"Model '{MODEL}' not found in Ollama. Run:  ollama pull {MODEL}")
            return
    except Exception as e:
        print("Ollama not reachable. Start it with:  ollama serve")
        return

    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    if not paths:
        print(f"No images in {IMG_DIR}. Run Step 2 first.")
        return
    if BATCH_LIMIT:
        paths = paths[:BATCH_LIMIT]

    caps = {}
    start = time.time()
    for i, p in enumerate(paths, 1):
        try:
            caps[p] = ollama_generate_vision(p)
        except Exception:
            caps[p] = "floor plan, architectural diagram"
        if i % 100 == 0:
            print(f"Captioned {i}/{len(paths)}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(caps, f, indent=2, ensure_ascii=False)

    print(f"Done. Captions: {len(caps)} -> {OUT_PATH}  | time: {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
