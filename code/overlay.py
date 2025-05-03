from PIL import Image
import os

# Input KDE image files
KDE_IMAGES = [
    "../output/latest_kde_price_per_sqft.png",  # Replace with your actual file paths
    "../output/oldest_kde_price_per_sqft.png"
] 

# Paths for the label map and output directory
LABEL_IMAGE_PATH = "../output/white_overlay_ready_sg_labels.png"
OUTPUT_DIR = "../output"

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load label image
label_img = Image.open(LABEL_IMAGE_PATH).convert("RGBA")

# Process each KDE image
for kde_path in KDE_IMAGES:
    # Load KDE image
    kde_img = Image.open(kde_path).convert("RGBA")
    
    # Resize label image to match the KDE image dimensions (if needed)
    if kde_img.size != label_img.size:
        label_img_resized = label_img.resize(kde_img.size, Image.Resampling.LANCZOS)
    else:
        label_img_resized = label_img

    # Overlay the label image on the KDE image
    combined = Image.alpha_composite(kde_img, label_img_resized)

    # Generate the output filename
    output_filename = os.path.join(OUTPUT_DIR, f"overlay_{os.path.basename(kde_path)}")
    
    # Save the result
    combined.save(output_filename)

    print(f"Saved overlayed image to {output_filename}")
