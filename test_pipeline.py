import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from dotenv import load_dotenv

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import refactored modules
from src.image_processor import find_stickers_and_swatches, resize_image_for_api
from src.ocr_extractor import extract_fabric_data
from src.excel_generator import generate_fabric_excel, clean_up_temp_images
from src.test_image_generator import create_test_image

# Load environment variables (e.g., GOOGLE_API_KEY) from .env file
load_dotenv()

def run_pipeline_test():
    print("\n--- Starting Digital Fabric Library Pipeline Test ---")

    # 1. Generate synthetic test images
    print("Generating test images...")
    test_images_data = [
        {
            "name": "fabric_swatch_01.png",
            "sticker_text": "Brand: MK\nItem: MK1234\nContent: 80% Cotton 20% Poly"
        },
        {
            "name": "fabric_swatch_02.png",
            "sticker_text": "Brand: Calvin Klein\nItem: CK5678\nContent: 100% Silk\nWeight: 120gsm"
        },
        {
            "name": "fabric_swatch_03.png",
            "sticker_text": "Item: TLL001\nContent: 95% Wool 5% Spandex"
        } # No brand for this one
    ]

    temp_image_files = []
    for img_data in test_images_data:
        base64_img = create_test_image(img_data["sticker_text"], dimensions=(800, 600), sticker_color=(250,250,250), background_color=(150,150,150), text_color=(0,0,0))
        img_path = f"temp_test_{img_data['name']}"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(base64_img))
        temp_image_files.append(img_path)
        print(f"  Generated {img_path}")

    all_processed_swatches = []
    temp_swatch_output_paths = [] # To store paths of temporarily saved swatch images for excel

    # 2. Process each test image through the pipeline
    debug_figs = []
    for image_path in temp_image_files:
        print(f"\nProcessing image: {image_path}")
        img_array = cv2.imread(image_path)
        if img_array is None:
            print(f"Error: Could not load image {image_path}. Skipping.")
            continue

        detected_items, debug_img = find_stickers_and_swatches(img_array)

        # Add debug image to list for display
        debug_figs.append({'name': image_path, 'image': debug_img})

        if not detected_items:
            print(f"No stickers/swatches detected in {image_path}. No data extracted.")
            continue
        
        print(f"Found {len(detected_items)} potential sticker/swatch pairs.")

        for i, item in enumerate(detected_items):
            sticker_crop = item['sticker_crop']
            swatch_crop = item['swatch_crop']

            # Resize sticker for API
            small_sticker = resize_image_for_api(sticker_crop)

            print(f"  Extracting data for swatch {i+1}...")
            fabric_data = extract_fabric_data(small_sticker)
            print(f"    Extracted: Brand={fabric_data.get('Brand')}, Item={fabric_data.get('Item')}, Content={fabric_data.get('Content')}")

            # Save swatch image temporarily for Excel embedding
            swatch_output_path = f"temp_swatch_{os.path.basename(image_path).replace('.png', '')}_{i}.png"
            cv2.imwrite(swatch_output_path, swatch_crop)
            temp_swatch_output_paths.append(swatch_output_path)

            all_processed_swatches.append({
                "Brand": fabric_data.get('Brand'),
                "Item": fabric_data.get('Item'),
                "Content": fabric_data.get('Content'),
                "swatch_image_path": swatch_output_path
            })

    # 3. Generate Excel file
    print("\nGenerating Excel file...")
    output_excel_file = generate_fabric_excel(all_processed_swatches, "Test_Fabric_Library.xlsx")
    print(f"Excel file generated: {output_excel_file}")

    # 4. Cleanup temporary files
    print("Cleaning up temporary image files...")
    for f in temp_image_files:
        if os.path.exists(f): os.remove(f)
    clean_up_temp_images(all_processed_swatches) # Cleans up temp swatch images
    print("Cleanup complete.")

    print("\n--- Pipeline Test Finished Successfully ---")
    print(f"Please check '{output_excel_file}' for the results.")

    # Display debug images
    if debug_figs:
        print("\nDisplaying debug images (close windows to proceed)...")
        for fig_data in debug_figs:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(fig_data['image'], cv2.COLOR_BGR2RGB))
            plt.title(f"Debug View: {fig_data['name']}")
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    run_pipeline_test()
