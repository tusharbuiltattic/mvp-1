# infer_ui.py — fixed for your stack
# Pins: gradio==4.44.1, gradio-client==1.3.0, fastapi==0.113.0, starlette==0.38.6
import os, glob, inspect, warnings
import torch, gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Guard against gradio-client JSON Schema=bool crashes ----
import gradio_client.utils as gu
_ORIG_JSON = gu.json_schema_to_python_type
_ORIG_GET  = gu.get_type
_ORIG_PRIV = getattr(gu, "_json_schema_to_python_type", None)

def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "any"
    return _ORIG_GET(schema)

def _safe_json(schema, defs=None):
    if isinstance(schema, bool):
        return "any"
    # call with the arity the installed version expects
    n = len(inspect.signature(_ORIG_JSON).parameters)
    return _ORIG_JSON(schema, defs) if n >= 2 else _ORIG_JSON(schema)

def _safe_priv(schema, defs=None):
    if isinstance(schema, bool):
        return "any"
    if _ORIG_PRIV is None:
        return _safe_json(schema, defs)
    n = len(inspect.signature(_ORIG_PRIV).parameters)
    return _ORIG_PRIV(schema, defs) if n >= 2 else _ORIG_PRIV(schema)

gu.get_type = _safe_get_type
gu.json_schema_to_python_type = _safe_json
if _ORIG_PRIV is not None:
    setattr(gu, "_json_schema_to_python_type", _safe_priv)
# ----------------------------------------------------------------

def latest_ckpt(root="archai_lora"):
    cs = sorted(glob.glob(os.path.join(root, "checkpoint-*")),
                key=lambda p: int(p.split("-")[-1])) if os.path.isdir(root) else []
    return cs[-1] if cs else root

LORA_DIR = latest_ckpt()
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if DEVICE == "cuda" else torch.float32

# rules → prompt
def arch_ai_prompt(choices, base="detailed architectural floor plan, top-down"):
    t = [base, "clean lines, clear circulation axes, readable labels"]
    c = choices.get("Climate")
    if c in {"Hot & Dry","Hot & Humid","Tropical"}:
        t += ["east/south openings with shading","cross-ventilation","verandas"]
    if c in {"Cold","Temperate"}:
        t += ["south-facing glazing","interior thermal mass"]
    if choices.get("Interior Plan") == "Open":
        t += ["open-plan living dining kitchen"]
    if choices.get("Style"):     t += [choices["Style"].lower()+" style"]
    if choices.get("Roof Type"): t += [choices["Roof Type"].lower()+" roof"]
    feats = choices.get("Additional Features") or []
    if feats: t += [", ".join(f.lower() for f in feats)]
    return ", ".join(t)

SCHEMA = {
"Category":["Residential","Commercial","Industrial","Agricultural","Recreational","Institutional","Mixed Use","Infrastructural"],
"Style":["Classical","Gothic","Renaissance","Baroque","Neoclassical","Victorian","Beaux-Arts","Art Nouveau","Art Deco","Modernism","Bauhaus","International Style","Mid-Century Modern","Brutalism","Postmodernism","Deconstructivism","Minimalism","Neo-Futurism","Bohemian","Industrial","Eco-architecture"],
"Material Used":["Stone","Brick","Concrete","Steel","Glass","Wood","Bamboo","Aluminum","Copper","Plaster & Stucco","Adobe","Rammed Earth","Thatch","Acrylic/PVC","Fiber-reinforced polymers"],
"Soil Type":["Loose","Soft","Firm","Stiff","Hard","Rock"],
"Terrain":["Flat","Sloping","Hilly","Mountainous","Waterfront","Coastal","Plateau","Valley"],
"Climate":["Hot & Dry","Hot & Humid","Cold","Temperate","Composite","Tropical"],
"Roof Type":["Flat","Gable","Hip","Shed","Mansard","Gambrel","Butterfly","Dome","Pyramid","Curved","Sawtooth","Green Roof"],
"Interior Plan":["Open","Closed","Linear","Centralized","Radial","Grid","Cluster","Split-Level"],
"Sustainability":["Passive Solar Design","Green Roofs","Rainwater Harvesting","Greywater Recycling","Natural Ventilation","Thermal Mass","Daylighting","Net-Zero Energy","Low-Carbon Materials","Recycled Materials","Smart Glass","Solar Panels","Wind Energy","Geothermal","Biomimicry","BREEAM Standards","Circular Economy","Adaptive Reuse","Carbon-Neutral Construction"],
"Additional Features":["Balconies","Verandas","Terraces","Patios","Decks","Pergolas","Atriums","Courtyards","Bay Windows","Chimneys","Columns","Arches","Domes","Spires","Ramps","Elevators","Water Features","Fireplaces","Gates","Fences","Driveways","Porches","Staircases"],
"Exterior Finish":["Façade","Cladding","Siding","Stucco","Brickwork","Stone Veneer","Glass Curtain Wall","Timber Exterior","Metal Panels","Green Walls","Canopies","Awnings"]
}

# load base + LoRA
BASE = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(BASE, dtype=DTYPE, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
_lora_loaded = False
try:
    pipe.load_lora_weights(LORA_DIR)
    _lora_loaded = True
except Exception as e:
    LORA_DIR = f"(no LoRA loaded: {e})"
if DEVICE == "cuda":
    pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()
pipe.to(DEVICE)

def generate(category, style, material, soil, terrain, climate, roof, interior,
             sustainability, addl, exterior, user_prompt, use_rules, negative_prompt,
             steps, guidance, seed, width, height):
    try:
        choices = {
            "Category": category, "Style": style, "Material Used": material,
            "Soil Type": soil, "Terrain": terrain, "Climate": climate,
            "Roof Type": roof, "Interior Plan": interior, "Sustainability": sustainability,
            "Additional Features": addl or [], "Exterior Finish": exterior
        }
        rules = arch_ai_prompt(choices)
        up = (user_prompt or "").strip()
        prompt = f"{up}, {rules}" if (use_rules and up) else (rules if use_rules else (up or rules))
        gen = None if int(seed) < 0 else torch.Generator(DEVICE).manual_seed(int(seed))
        out = pipe(prompt,
                   negative_prompt=negative_prompt,
                   num_inference_steps=int(steps),
                   guidance_scale=float(guidance),
                   generator=gen,
                   width=int(width),
                   height=int(height))
        ckpt = f"{LORA_DIR} ({'loaded' if _lora_loaded else 'base only'})"
        return out.images[0], prompt, ckpt
    except Exception as e:
        return None, f"ERROR: {type(e).__name__}: {e}", f"LoRA: {LORA_DIR}"

with gr.Blocks(title="Arch-AI Inference", analytics_enabled=False) as demo:
    gr.Markdown("### Arch-AI — SD1.5 LoRA Inference")
    with gr.Row():
        with gr.Column(scale=1):
            category = gr.Dropdown(SCHEMA["Category"], value="Residential", label="Category")
            style    = gr.Dropdown(SCHEMA["Style"], value="Modernism", label="Style")
            material = gr.Dropdown(SCHEMA["Material Used"], value="Concrete", label="Material Used")
            soil     = gr.Dropdown(SCHEMA["Soil Type"], value="Firm", label="Soil Type")
            terrain  = gr.Dropdown(SCHEMA["Terrain"], value="Flat", label="Terrain")
            climate  = gr.Dropdown(SCHEMA["Climate"], value="Temperate", label="Climate")
            roof     = gr.Dropdown(SCHEMA["Roof Type"], value="Flat", label="Roof Type")
            interior = gr.Dropdown(SCHEMA["Interior Plan"], value="Open", label="Interior Plan")
            sustain  = gr.Dropdown(SCHEMA["Sustainability"], value="Daylighting", label="Sustainability")
            addl     = gr.CheckboxGroup(SCHEMA["Additional Features"], value=["Verandas"], label="Additional Features")
            exterior = gr.Dropdown(SCHEMA["Exterior Finish"], value="Cladding", label="Exterior Finish")
            user_prompt = gr.Textbox(label="Prompt", placeholder="Type your prompt here…")
            use_rules   = gr.Checkbox(value=True, label="Append Arch-AI rules prompt")
            negative    = gr.Textbox(value="low quality, blurry, extra walls, unreadable labels", label="Negative prompt")
            steps    = gr.Slider(10, 50, value=30, step=1, label="Steps")
            guidance = gr.Slider(1.0, 12.0, value=7.0, step=0.5, label="CFG")
            seed     = gr.Slider(-1, 2**31-1, value=42, step=1, label="Seed (-1 random)")
            width    = gr.Slider(384, 768, step=64, value=512, label="Width")
            height   = gr.Slider(384, 768, step=64, value=512, label="Height")
            btn      = gr.Button("Generate")
        with gr.Column(scale=1):
            out_img    = gr.Image(label="Output", interactive=False)
            out_prompt = gr.Textbox(label="Used prompt", interactive=False)
            info       = gr.Textbox(label="LoRA checkpoint", interactive=False)

    btn.click(
        generate,
        inputs=[category, style, material, soil, terrain, climate, roof, interior,
                sustain, addl, exterior, user_prompt, use_rules, negative,
                steps, guidance, seed, width, height],
        outputs=[out_img, out_prompt, info],
        concurrency_limit=1,
    )

# Serve on a different localhost+port
demo.launch(server_name="127.0.0.1", server_port=7861, show_error=True, max_threads=8)
demo.launch(share=True)  # uncomment to get a public link