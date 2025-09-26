# arch_ai_app.py
# ArchAI: Prompt → Analysis → Rendered Concept (with your local LoRA).
# Fixes Gradio schema crash by deep-normalizing JSON schema,
# uses torch_dtype, random seeds for varied images, CLIP truncation, JSON export.

import os
import re
import glob
import json
from datetime import datetime

import torch
from PIL import Image
import gradio as gr

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
ADAPTER_NAME = "archai_lora"  # handle for your LoRA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
IMG_W, IMG_H = 512, 512
STEPS = 28
GUIDANCE = 7.0
NEG_DEFAULT = (
    "lowres, blurry, overexposed, text, watermark, logo, distorted, misaligned walls, "
    "wrong perspective, bad proportions, artifacts, nsfw"
)

# -----------------------------
# Gradio schema bug patch (robust)
# -----------------------------
def _normalize_json_schema(obj):
    """
    Recursively convert boolean JSON-schema nodes (e.g., additionalProperties: true)
    into {} so gradio_client.utils.* no longer tries `\"const\" in True`.
    """
    if isinstance(obj, bool):
        return {}
    if isinstance(obj, dict):
        return {k: _normalize_json_schema(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_json_schema(v) for v in obj]
    return obj

try:
    import gradio_client.utils as _gcu  # type: ignore

    _ORIG_JSON = getattr(_gcu, "json_schema_to_python_type", None)

    def _safe_json(schema, *args, **kwargs):
        if _ORIG_JSON is None:
            return {}
        try:
            schema = _normalize_json_schema(schema)
            # Newer gradio passes (schema, defs) etc.
            return _ORIG_JSON(schema, *args, **kwargs)
        except TypeError:
            # Older signature: single-arg
            try:
                return _ORIG_JSON(schema)
            except Exception:
                return {}
        except Exception:
            return {}

    if _ORIG_JSON is not None:
        _gcu.json_schema_to_python_type = _safe_json  # type: ignore
except Exception:
    # Non-fatal if patching fails
    pass

# -----------------------------
# Utilities
# -----------------------------
def _find_lora_dir(base_dir: str) -> str | None:
    patterns = [
        "archai_lora", "archiai_lora", "archai-lora",
        "archai_lora/checkpoint-*", "archiai_lora/checkpoint-*"
    ]
    needed = {
        "pytorch_lora_weights.safetensors",
        "pytorch_lora_weights.bin",
        "adapter_model.safetensors",
        "diffusers_lora_config.json",
        "adapter_config.json",
    }
    for pat in patterns:
        for p in glob.glob(os.path.join(base_dir, pat)):
            if os.path.isdir(p):
                files = set(os.listdir(p))
                if files & needed:
                    return os.path.abspath(p)
    return None

def _clip_truncate(text: str, tokenizer):
    ids_full = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    max_len = tokenizer.model_max_length
    ids_trim = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_len)
    was_trunc = len(ids_full) > len(ids_trim)
    txt = tokenizer.decode(ids_trim, skip_special_tokens=True).strip()
    return txt, len(ids_trim), max_len, was_trunc

def _rand_generator():
    # None => new random seed per call (varied images)
    return None

# -----------------------------
# Build pipeline
# -----------------------------
def build_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,     # use correct kwarg
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # memory helpers (6–8 GB)
    try: pipe.enable_attention_slicing()
    except Exception: pass
    try: pipe.vae.enable_tiling()
    except Exception: pass
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass

    lora_dir = _find_lora_dir(os.getcwd())
    if not lora_dir:
        raise FileNotFoundError(
            "LoRA folder not found. Expected one of: archai_lora, archiai_lora, or their checkpoint subfolders."
        )

    pipe.load_lora_weights(lora_dir, adapter_name=ADAPTER_NAME)
    pipe.set_adapters([ADAPTER_NAME])

    pipe.to(DEVICE)
    return pipe

PIPE = build_pipeline()

# -----------------------------
# Prompt parsing → rule-based analysis
# -----------------------------
CLIMATES = ["hot & dry", "hot & humid", "cold", "temperate", "composite", "tropical"]
STYLES = [
    "classical","gothic","renaissance","baroque","neoclassical","victorian","beaux-arts",
    "art nouveau","art deco","modernism","bauhaus","international style","mid-century modern",
    "brutalism","postmodernism","deconstructivism","minimalism","neo-futurism","bohemian",
    "industrial","eco-architecture"
]
MATERIALS = [
    "stone","brick","concrete","steel","glass","wood","bamboo","aluminum","copper","plaster",
    "stucco","adobe","rammed earth","thatch","acrylic","pvc","frp","fiber-reinforced polymers"
]

def _extract_lower(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _pick_from(text: str, choices: list[str]) -> list[str]:
    hits = []
    for c in choices:
        if c.lower() in text:
            hits.append(c)
    return hits

def analyze_design(user_prompt: str) -> dict:
    p = _extract_lower(user_prompt)
    climates = _pick_from(p, CLIMATES)
    styles = _pick_from(p, STYLES)
    mats = _pick_from(p, MATERIALS)

    climate = climates[0] if climates else "composite"
    style = styles[0] if styles else "modernism"
    material = mats[:3] if mats else ["concrete", "glass", "wood"]

    recs = {
        "Site Planning": [
            "Orient primary living to east/south for daylight (Principle 1, 5).",
            "Built:open ~60:40 for ventilation & landscape (Principle 1).",
            "Avoid valley bottoms; allow air drainage; stilted plinth if flood-prone (Principle 1).",
            "Respect neighbors’ right-to-light with green buffers (Principle 1).",
        ],
        "Zoning": [
            "Living/dining as social core; bedrooms quieter wing (Principle 2).",
            "Separate kitchen/toilets from living (Principle 2, 6).",
            "Intimacy gradient: public → semi-private → private (Principle 2).",
            "Thermal zoning: services on cold side, living on warm side (Principle 2).",
            "Open-plan LDK for interaction when feasible (Principle 2).",
        ],
        "Circulation": [
            "Corridors ≥ 900 mm; minimize dead ends (Principle 3).",
            "Accessibility: wheelchair turns, wider landings (Principle 3).",
            "Align axes for clear sightlines (Principle 3).",
            "Kitchen: separate clean/dirty paths (Principle 3).",
        ],
        "Spatial Dimensions": [
            "Living depth 3.6–4.5 m (Principle 4).",
            "Kitchen counter 850–900 mm (Principle 4).",
            "Toilet clear space ≥ 1.5×0.9 m; accessible ≥ 1.5×1.8 m (Principle 4).",
            "Dining ~1.5 m²/seat + 0.9–1.2 m aisles (Principle 4).",
            "Bedrooms ≥ 2.7–3.0 m wide; std heights (Principle 4).",
        ],
        "Openings": [
            "Windows 10–20% of room floor area; egress where required (Principle 5).",
            "Orient living to east/south; add shading (Principle 5).",
            "Verandas/loggias as climate buffers (Principle 5).",
            "Cross-ventilation + high-level vents for night purge (Principle 5).",
        ],
        "Services": [
            "Stack kitchens/baths to shorten plumbing (Principle 6).",
            "Sanitation: sealed drainage; handwash near toilets (Principle 6).",
            "Kitchen work-triangle; hood over cooktop (Principle 6).",
            "Utility corridors 1.5–2.0 m; separate potable/storm/electrical (Principle 6).",
            "Fire safety: detectors, kitchen compartment; sprinklers as required (Principle 6).",
        ],
        "Sustainability & Climate": [
            "Passive solar; south glazing + sized shading (Principle 7).",
            "Rainwater harvesting + cistern for dry spells (Principle 7).",
            "Greywater reuse where permitted (Principle 7).",
            "Cold: compact form + internal thermal mass (Principle 7).",
            "Daylight (clerestories/skylights); energy ≤ ~15 kWh/m²·yr (Principle 7).",
            "Local, low-embodied-energy materials (Principle 7).",
        ],
    }
    climate_notes = {
        "hot & dry": "Deep verandas, high mass, shaded courts, cross-ventilation.",
        "hot & humid": "Stilted floors, high airflow, wide overhangs, moisture-safe finishes.",
        "cold": "Compact form, south glazing with good U-values, thermal mass inside.",
        "temperate": "Balanced glazing with seasonal shading, flexible ventilation.",
        "composite": "Operable shading, mixed seasonal strategies.",
        "tropical": "Elevated floors, cross-ventilation, mold-resistant materials.",
    }

    return {
        "Extracted": {"Climate": climate, "Style": style, "Material(s)": material},
        "Recommendations": recs,
        "Climate Notes": climate_notes.get(climate, ""),
    }

def analysis_to_table(analysis: dict) -> str:
    rows = [
        ("Climate", analysis["Extracted"]["Climate"]),
        ("Style", analysis["Extracted"]["Style"]),
        ("Materials", ", ".join(analysis["Extracted"]["Material(s)"])),
        ("Climate Notes", analysis["Climate Notes"]),
    ]
    md = "| Category | Value |\n|---|---|\n"
    for k, v in rows:
        md += f"| {k} | {v} |\n"
    md += "\n**Key Rules Applied**\n\n"
    for sec, items in analysis["Recommendations"].items():
        md += f"- **{sec}**\n"
        for it in items:
            md += f"  - {it}\n"
    return md

# -----------------------------
# Inference
# -----------------------------
def generate(prompt: str):
    if not isinstance(prompt, str) or not prompt.strip():
        raise gr.Error("Prompt is empty.")

    safe_prompt, used_len, max_len, was_trunc = _clip_truncate(prompt, PIPE.tokenizer)
    safe_negative, n_used_neg, _, was_trunc_neg = _clip_truncate(NEG_DEFAULT, PIPE.tokenizer)

    try:
        PIPE.set_adapters([ADAPTER_NAME])
    except Exception:
        pass

    out = PIPE(
        prompt=safe_prompt,
        negative_prompt=safe_negative,
        width=IMG_W,
        height=IMG_H,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=_rand_generator(),  # None => random noise each call
    )
    img: Image.Image = out.images[0]

    analysis = analyze_design(prompt)
    table_md = analysis_to_table(analysis)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"analysis_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "original_prompt": prompt,
                "used_prompt": safe_prompt,
                "negative_prompt": safe_negative,
                "clip_truncated": {
                    "prompt_truncated": was_trunc,
                    "prompt_tokens_used": used_len,
                    "prompt_max_tokens": max_len,
                    "negative_truncated": was_trunc_neg,
                    "negative_tokens_used": n_used_neg,
                },
                "inference": {
                    "width": IMG_W, "height": IMG_H, "steps": STEPS, "guidance": GUIDANCE,
                    "adapter": ADAPTER_NAME, "model": MODEL_ID,
                },
                "analysis": analysis,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return img, table_md, json_path

# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="ArchAI") as demo:
    gr.Markdown("## 🏛️ ArchAI — Prompt → Analysis → Rendered Concept")
    prompt_in = gr.Textbox(
        label="Architectural Prompt",
        placeholder="e.g., Residential 3BHK in hot & humid climate, open-plan living/dining/kitchen, courtyards, deep verandas, stacked services.",
        lines=3,
    )
    run_btn = gr.Button("Generate")
    out_img = gr.Image(label="Rendered Concept", type="pil")
    out_md = gr.Markdown(label="Analysis")
    out_json = gr.File(label="Download Analysis JSON")

    run_btn.click(
        fn=generate,
        inputs=[prompt_in],
        outputs=[out_img, out_md, out_json],
        concurrency_limit=1,
        api_name="generate",
    )

if __name__ == "__main__":
    # 127.0.0.1 avoids some Windows proxy/hosts issues; change to "0.0.0.0" if you want LAN access.
    demo.queue(max_size=16).launch(server_name="127.0.0.1", server_port=7860, share=False, max_threads=8)
