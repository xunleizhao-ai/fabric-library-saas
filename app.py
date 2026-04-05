import os

import streamlit as st
import sys
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import refactored modules
from src.image_processor import find_stickers_and_swatches, resize_image_for_api
from src.ocr_extractor import extract_fabric_data
from src.excel_generator import generate_fabric_excel, clean_up_temp_images

# --- Authentication Logic ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

TEST_USERS = {
    "tester1": "test123",
    "tester2": "test123"
}

if not st.session_state['authenticated']:
    st.title("Login to Digital Fabric Library")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in TEST_USERS and TEST_USERS[username] == password:
            st.session_state['authenticated'] = True
            st.success("Logged in successfully!")
            st.rerun() # Rerun to hide login and show app content
        else:
            st.error("Invalid username or password")
else:
    # --- Streamlit UI Setup (Existing content goes here) ---
    st.set_page_config(page_title="Digital Fabric Library", layout="centered")
    st.title("✂️ Digital Fabric Library")
    st.markdown("Upload fabric swatch photos, and I'll extract data from stickers and compile it into an Excel file with images.")

    # --- Cropping Parameters in Sidebar ---
    st.sidebar.header("Cropping Parameters")
    white_threshold = st.sidebar.slider("WHITE_THRESHOLD", 0, 255, 200)
    min_sticker_area = st.sidebar.number_input("MIN_STICKER_AREA", 0, 10000, 5000)
    jump_down_pixels = st.sidebar.slider("JUMP_DOWN_PIXELS", 0, 100, 20)
    swatch_square_size = st.sidebar.number_input("SWATCH_SQUARE_SIZE", 100, 1000, 300)

    # --- File Uploader ---
    uploaded_files = st.file_uploader(
        "Upload one or more fabric swatch photos (max 10)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("Please upload a maximum of 10 images.")
            uploaded_files = uploaded_files[:10] # Limit to first 10

        process_button = st.button("Process Photos")

        if process_button:
            st.subheader("Processing Results")
            all_processed_swatches = []
            temp_swatch_output_paths = []

            # Create a temporary directory for output images
            temp_image_dir = "temp_swatches_for_excel"
            os.makedirs(temp_image_dir, exist_ok=True)

            for i, uploaded_file in enumerate(uploaded_files):
                st.write(f"Processing: {uploaded_file.name}")

                # Read image data
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img_array is None:
                    st.error(f"Could not load image {uploaded_file.name}. Skipping.")
                    continue

                # Find stickers and swatches
                detected_items, debug_img = find_stickers_and_swatches(img_array, white_threshold, min_sticker_area, jump_down_pixels, swatch_square_size)

                if not detected_items:
                    st.warning(f"No stickers/swatches detected in {uploaded_file.name}. No data extracted.")
                    continue

                st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), caption=f"Processed: {uploaded_file.name} (Detections)", use_column_width=True)

                for j, item in enumerate(detected_items):
                    sticker_crop = item['sticker_crop']
                    swatch_crop = item['swatch_crop']

                    # Resize sticker for AI API (token optimization)
                    small_sticker = resize_image_for_api(sticker_crop)

                    # Extract fabric data using AI
                    fabric_data = extract_fabric_data(small_sticker)
                    # st.write(f"  - Extracted from {uploaded_file.name} (Swatch {j+1}): Brand={fabric_data.get('Brand')}, Item={fabric_data.get('Item')}")

                    # Save swatch image temporarily for Excel embedding
                    swatch_output_filename = f"swatch_{os.path.basename(uploaded_file.name).replace('.', '_')}_{j}.png"
                    swatch_output_path = os.path.join(temp_image_dir, swatch_output_filename)
                    cv2.imwrite(swatch_output_path, swatch_crop)
                    temp_swatch_output_paths.append(swatch_output_path)

                    all_processed_swatches.append({
                        "Brand": fabric_data.get('Brand'),
                        "Item": fabric_data.get('Item'),
                        "Content": fabric_data.get('Content'),
                        "swatch_image_path": swatch_output_path
                    })

            if all_processed_swatches:
                st.success("All photos processed! Generating Excel file...")


                output_excel_filename = "Digital_Fabric_Library.xlsx"
                output_excel_file_path = generate_fabric_excel(all_processed_swatches, output_excel_filename)

                # Provide download link for the Excel file
                with open(output_excel_file_path, "rb") as f:
                    st.download_button(
                        label="Download Excel Report",
                        data=f.read(),
                        file_name=output_excel_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                st.success("Excel report ready for download.")

                # Clean up temporary swatch images after Excel generation
                clean_up_temp_images(all_processed_swatches) # This will also remove the temp_image_dir itself
                if os.path.exists(temp_image_dir): # Ensure dir is removed if empty
                    os.rmdir(temp_image_dir)

            else:
                st.info("No fabric data could be extracted from the uploaded images.")


