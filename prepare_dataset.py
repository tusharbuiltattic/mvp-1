import zipfile, os, glob
from PIL import Image

# Path to your zip
zip_path = r"D:\archi\HouseExpo_png.zip"
# Where to unzip
root = "houseexpo_png"
os.makedirs(root, exist_ok=True)

# 1) Unzip all images
with zipfile.ZipFile(zip_path) as z:
    z.extractall(root)

# 2) Collect image paths
imgs = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
print("Found:", len(imgs), "raw images")

# 3) Normalize all images to 512x512
norm_dir = "data/images"
os.makedirs(norm_dir, exist_ok=True)

for i, p in enumerate(imgs):
    try:
        im = Image.open(p).convert("RGB").resize((512, 512))
        im.save(os.path.join(norm_dir, f"img_{i:07d}.png"))
    except Exception as e:
        print("skip", p, "error:", e)

print("Normalized:", len(os.listdir(norm_dir)), "images written to", norm_dir)
