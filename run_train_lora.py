# run_train_lora.py
import os, subprocess, sys, pathlib

# Paths
ROOT = pathlib.Path(__file__).parent.resolve()
DIFF = ROOT / "diffusers" / "examples" / "text_to_image"
MODEL = "runwayml/stable-diffusion-v1-5"
DATA_DIR = str(ROOT / "train")          # contains images/ and metadata.jsonl
OUT_DIR  = str(ROOT / "archai_lora")    # will be created

# Hyperparams tuned for ~6 GB VRAM
RES          = 512
BATCH        = 1
GRAD_ACCUM   = 8
LR           = 1e-4
MAX_STEPS    = 1500          # raise later for quality
RANK         = 4
SEED         = 42

cmd = [
    sys.executable,
    str(DIFF / "train_text_to_image_lora.py"),
    "--pretrained_model_name_or_path", MODEL,
    "--train_data_dir", DATA_DIR,
    "--image_column", "image",
"--caption_column", "text",


    "--resolution", str(RES),
    "--random_flip",
    "--train_batch_size", str(BATCH),
    "--gradient_accumulation_steps", str(GRAD_ACCUM),
    "--max_train_steps", str(MAX_STEPS),
    "--learning_rate", str(LR),
    "--lr_scheduler", "cosine",
    "--lr_warmup_steps", "100",
    "--mixed_precision", "fp16",
    "--seed", str(SEED),
    "--checkpointing_steps", "300",
    "--output_dir", OUT_DIR,
    "--rank", str(RANK),
    "--gradient_checkpointing",
]

print("Launching training…")
print(" ".join(cmd))
subprocess.run(cmd, check=True)
