# infer_ui.py — SD1.5 + ControlNet (MLSD + Lineart) floorplan UI
# Works on 6 GB VRAM with fp16. Produces labeled, flat-color plans.
# Variability controls: seed, denoise (strength), and layout jitter.

import os
import time
import math
import numpy as np
from typing import Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import gradio as gr

from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models import MultiControlNetModel


# -------------------- runtime / model --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
FONT = ImageFont.load_default()

NEG_DEFAULT = (
    "3d perspective, textures, shadows, noisy strokes, blur, watermark, photo, "
    "realistic furniture, perspective grid, low contrast"
)

ROOM_COLORS = {
    "Living / Dining": (232, 222, 210),
    "Kitchen": (220, 206, 190),
    "Bedroom": (219, 210, 201),
    "Bath": (225, 235, 240),
    "Corridor": (235, 233, 230),
    "Garage": (226, 230, 240),
    "Veranda": (230, 233, 237),
    "Store": (234, 230, 226),
    "Home Office": (228, 225, 217),
}

# ------------- tiny drawing helpers (no external assets) -------------
def _fill(d: ImageDraw.ImageDraw, r: Tuple[int, int, int, int], color):
    d.rectangle(r, fill=color, outline=(30, 30, 30), width=3)

def draw_bed(d, r):
    x0, y0, x1, y1 = r
    pad = 6
    d.rectangle((x0 + pad, y0 + pad, x1 - pad, y1 - pad), outline=(60, 60, 60), width=2)
    d.line((x0 + pad, (y0 + y1) // 2, x1 - pad, (y0 + y1) // 2), fill=(60, 60, 60), width=2)

def draw_sofa(d, r):
    x0, y0, x1, y1 = r
    d.rounded_rectangle((x0 + 6, y0 + 8, x1 - 6, y1 - 8), radius=8, outline=(60, 60, 60), width=2)

def draw_table(d, r):
    x0, y0, x1, y1 = r
    d.ellipse((x0 + 8, y0 + 8, x1 - 8, y1 - 8), outline=(60, 60, 60), width=2)

def draw_kitchen(d, r):
    x0, y0, x1, y1 = r
    d.rectangle((x0 + 4, y0 + 4, x0 + 30, y1 - 4), outline=(60, 60, 60), width=2)  # base
    d.rectangle((x1 - 30, y0 + 4, x1 - 4, y1 - 4), outline=(60, 60, 60), width=2)  # tall

def draw_bath(d, r):
    x0, y0, x1, y1 = r
    d.rectangle((x0 + 6, y0 + 6, x0 + 36, y0 + 22), outline=(60, 60, 60), width=2)  # tub
    d.rectangle((x1 - 30, y1 - 22, x1 - 10, y1 - 6), outline=(60, 60, 60), width=2)  # wc


def make_generator(seed: int | None):
    if seed is None or int(seed) < 0:
        return None
    return torch.Generator(device=DEVICE).manual_seed(int(seed))


# -------------------- base plan generator (deterministic + jitter) --------------------
def plan_image(config, pr, W=768, H=576, seed: int = -1, jitter_pct: int = 6):
    """
    Draws a labeled CAD-like base plan. Units: approximate meters via ppm.
    jitter_pct randomly perturbs key spans to avoid identical layouts.
    """
    rng = np.random.default_rng(None if seed is None or seed < 0 else int(seed))

    def J(px, pct=jitter_pct):
        if pct <= 0:
            return int(px)
        return int(round(px * (1.0 + rng.uniform(-pct, pct) / 100.0)))

    ppm = 66  # pixels per meter for nominal scale
    def m(v): return int(round(v * ppm))

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    pad = int(0.035 * min(W, H))
    x0, y0, x1, y1 = pad, pad, W - pad, H - pad
    d.rectangle((x0, y0, x1, y1), outline=(30, 30, 30), width=3)

    veranda = config["Climate"] in {"Hot & Dry", "Hot & Humid", "Tropical"}

    # Service strip (top)
    serv_h = J(m(2.2))
    S = (x0, y0, x1, y0 + serv_h)
    _fill(d, S, ROOM_COLORS["Kitchen"])

    nb = pr["baths"]
    bw = (S[2] - S[0]) // (nb + 1)

    # Baths
    for i in range(nb):
        bx0 = S[0] + i * bw
        r = (bx0 + 3, S[1] + 3, bx0 + bw - 6, S[3] - 3)
        _fill(d, r, ROOM_COLORS["Bath"])
        draw_bath(d, r)
        d.text(((r[0] + r[2]) // 2, (r[1] + r[3]) // 2), f"Bath {i+1}",
               fill=(30, 30, 30), anchor="mm", font=FONT)

    # Kitchen block at right end of service strip
    K = (S[2] - bw + 3, S[1] + 3, S[2] - 3, S[3] - 3)
    _fill(d, K, ROOM_COLORS["Kitchen"])
    draw_kitchen(d, K)
    d.text(((K[0] + K[2]) // 2, (K[1] + K[3]) // 2), "Kitchen",
           fill=(30, 30, 30), anchor="mm", font=FONT)

    # Bedrooms west
    Bx1 = x0 + J(m(3.3))
    By0 = S[3]
    By1 = y1 - (J(m(1.2)) if veranda else 0)
    bh = (By1 - By0) // max(1, pr["beds"])
    for i in range(pr["beds"]):
        r = (x0 + 3, By0 + i * bh + 3, Bx1 - 3, By0 + (i + 1) * bh - 3)
        _fill(d, r, ROOM_COLORS["Bedroom"])
        draw_bed(d, r)
        d.text(((r[0] + r[2]) // 2, (r[1] + r[3]) // 2), f"Bedroom {i+1}",
               fill=(30, 30, 30), anchor="mm", font=FONT)

    # Corridor (≥ 0.9m)
    corridor_w = max(J(m(0.9)), int(0.03 * (x1 - x0)))
    C = (Bx1, By0, Bx1 + corridor_w, By1)
    _fill(d, C, ROOM_COLORS["Corridor"])
    d.text(((C[0] + C[2]) // 2, (C[1] + C[3]) // 2), "Corridor",
           fill=(30, 30, 30), anchor="mm", font=FONT)

    # Living / Dining
    L = (C[2], By0, x1 - 3, By1 - 3)
    _fill(d, L, ROOM_COLORS["Living / Dining"])
    draw_sofa(d, (L[0] + 10, L[1] + 10, L[0] + 120, L[1] + 70))
    draw_table(d, (L[0] + 150, L[1] + 20, L[0] + 230, L[1] + 90))
    d.text(((L[0] + L[2]) // 2, (L[1] + L[3]) // 2), "Living / Dining",
           fill=(30, 30, 30), anchor="mm", font=FONT)

    # Optional office/store
    if pr.get("office", False):
        off = (C[2] + J(m(2.0)), By0 + J(m(0.6)), C[2] + J(m(4.0)), By0 + J(m(2.6)))
        _fill(d, off, ROOM_COLORS["Home Office"])
        d.text(((off[0] + off[2]) // 2, (off[1] + off[3]) // 2), "Home Office",
               fill=(30, 30, 30), anchor="mm", font=FONT)
    if pr.get("store", False):
        st = (K[0] - J(m(1.2)), S[1], K[0], S[1] + J(m(1.8)))
        _fill(d, st, ROOM_COLORS["Store"])
        d.text(((st[0] + st[2]) // 2, (st[1] + st[3]) // 2), "Store",
               fill=(30, 30, 30), anchor="mm", font=FONT)

    # Climate buffer
    if veranda:
        V = (x0 + 3, By1 + 3, x1 - 3, y1 - 3)
        _fill(d, V, ROOM_COLORS["Veranda"])
        d.text(((V[0] + V[2]) // 2, (V[1] + V[3]) // 2), "Veranda",
               fill=(30, 30, 30), anchor="mm", font=FONT)

    return img


def control_images_from_base(base_img: Image.Image):
    """Create lineart + MLSD-like binary edge maps from the base plan."""
    arr = np.array(base_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # binary map of walls/edges
    _, bw = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.dilate(bw, np.ones((3, 3), np.uint8), iterations=1)

    edges = cv2.Canny(255 - bw, 50, 150)
    lineart = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2RGB)

    # a slightly cleaned "mlsd" clone (morphological)
    mlsd = cv2.morphologyEx(lineart, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return Image.fromarray(lineart), Image.fromarray(mlsd)


# -------------------- pipeline --------------------
def build_pipeline():
    # ensure we do NOT send any HF token accidentally
    for k in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        os.environ.pop(k, None)
    cache_dir = os.path.join(os.getcwd(), "hf_cache")

    cn1 = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_mlsd", torch_dtype=DTYPE, cache_dir=cache_dir
    )
    cn2 = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart", torch_dtype=DTYPE, cache_dir=cache_dir
    )
    controlnet = MultiControlNetModel([cn1, cn2])

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=DTYPE,
        controlnet=controlnet,
        safety_checker=None,
        cache_dir=cache_dir,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if DEVICE == "cuda":
        if is_xformers_available():
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        pipe.enable_attention_slicing()
        pipe.vae.enable_tiling()
        pipe.unet.to(memory_format=torch.channels_last)
    pipe.to(DEVICE)
    return pipe


PIPE = build_pipeline()


# -------------------- prompting --------------------
def build_prompt(category, style, material, climate):
    return (
        f"flat 2D architectural floor plan, {category.lower()}, style {style.lower()}, "
        f"material palette {material.lower()}, climate {climate.lower()}, "
        "clean vector ink lines, uniform fills, room labels readable, "
        "simple icon furniture, consistent wall thickness, top-down, scale bar"
    )


# -------------------- generation --------------------
def generate(
    category, style, material, climate,
    beds, baths, want_office, want_store,
    steps, cfg, mlsd_scale, lineart_scale,
    seed, denoise, jitter, width, height
):
    pr = {
        "beds": int(beds),
        "baths": int(baths),
        "office": bool(want_office),
        "store": bool(want_store),
    }
    cfg_map = {"Climate": climate}

    base = plan_image(cfg_map, pr, W=int(width), H=int(height),
                      seed=int(seed), jitter_pct=int(jitter))
    lineart_img, mlsd_img = control_images_from_base(base)

    gen = make_generator(int(seed))
    prompt = build_prompt(category, style, material, climate)

    image = PIPE(
        prompt=prompt,
        negative_prompt=NEG_DEFAULT,
        image=base,
        control_image=[mlsd_img, lineart_img],
        controlnet_conditioning_scale=[float(mlsd_scale), float(lineart_scale)],
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        strength=float(denoise),
        generator=gen,
        width=base.width,
        height=base.height,
    ).images[0]

    used_seed = int(seed if int(seed) >= 0 else int(time.time()) & 0x7FFFFFFF)
    meta = f"seed={used_seed} | steps={steps} | cfg={cfg} | denoise={denoise} | jitter%={jitter}"
    return image, base, lineart_img, mlsd_img, prompt, meta


# -------------------- UI --------------------
with gr.Blocks(title="Arch-AI — SD1.5 ControlNet Floorplan") as demo:
    gr.Markdown("### Arch-AI — SD1.5 LoRA/ControlNet Inference (deterministic + jitter)")
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            category = gr.Dropdown(
                ["Residential", "Commercial", "Mixed Use"], value="Residential", label="Category"
            )
            style = gr.Dropdown(
                ["Modernism", "Minimalism", "Neo-Futurism", "Industrial"],
                value="Modernism", label="Style"
            )
            material = gr.Dropdown(
                ["Concrete", "Brick", "Wood", "Steel", "Glass"],
                value="Concrete", label="Material Used"
            )
            climate = gr.Dropdown(
                ["Temperate", "Hot & Dry", "Hot & Humid", "Tropical", "Cold", "Composite"],
                value="Temperate", label="Climate"
            )
            beds = gr.Slider(1, 4, 2, step=1, label="Bedrooms")
            baths = gr.Slider(1, 3, 2, step=1, label="Baths")
            want_office = gr.Checkbox(False, label="Home Office")
            want_store = gr.Checkbox(False, label="Store / Utility")

            gr.Markdown("**Sampler / Control**")
            steps = gr.Slider(8, 30, 14, step=1, label="Steps")
            cfg = gr.Slider(3.0, 10.0, 5.0, step=0.5, label="CFG")
            denoise = gr.Slider(0.15, 0.75, 0.45, step=0.01, label="Denoise strength")
            jitter = gr.Slider(0, 15, 6, step=1, label="Layout jitter %")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            mlsd_scale = gr.Slider(0.8, 2.0, 1.45, step=0.05, label="MLSD scale")
            lineart_scale = gr.Slider(0.3, 1.2, 0.65, step=0.05, label="Lineart scale")
            width = gr.Slider(640, 896, 768, step=32, label="Width")
            height = gr.Slider(448, 704, 576, step=32, label="Height")

            btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            out = gr.Image(label="Output", interactive=False)
            with gr.Row():
                base_img = gr.Image(label="Base plan (CAD)")
                lineart_v = gr.Image(label="Control: Lineart")
                mlsd_v = gr.Image(label="Control: MLSD")
            used_prompt = gr.Textbox(label="Used prompt")
            meta = gr.Textbox(label="Run meta")

    btn.click(
        fn=generate,
        inputs=[
            category, style, material, climate,
            beds, baths, want_office, want_store,
            steps, cfg, mlsd_scale, lineart_scale,
            seed, denoise, jitter, width, height
        ],
        outputs=[out, base_img, lineart_v, mlsd_v, used_prompt, meta],
        concurrency_limit=1,
        api_name=False,
    )

if __name__ == "__main__":
    # Different local host/port to avoid collisions
    demo.launch(server_name="127.0.0.1", server_port=7865, show_error=True, max_threads=8)
