import google.generativeai as genai
import PIL.Image
import json
import time
import cv2
import streamlit as st

# =========================================================================
# AI Setup (using st.secrets for API key)
# =========================================================================
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Gemini API key not found in Streamlit secrets. Please add it to your `secrets.toml` file or Hugging Face Space secrets.")
    st.stop() # Stop the app if API key is not available
genai.configure(api_key=API_KEY)

def get_ai_model():
    """
    Dynamically picks the best available Gemini Flash model.
    """
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    target_models = ['gemini-2.5-flash-lite', 'gemini-2.0-flash-lite', 'gemini-flash-lite-latest', 'gemini-2.0-flash', 'gemini-1.5-flash']
    best_model_name = next((m for target in target_models for m in available_models if target in m), None)
    
    if not best_model_name:
        # Fallback if no specific flash model is found, try any available Gemini model
        best_model_name = next((m for m in available_models if 'gemini' in m), None)
        if not best_model_name:
            raise RuntimeError("No suitable Gemini model found.")

    return genai.GenerativeModel(best_model_name)

model = get_ai_model()

# --- THE SUPER-PROMPT (tuned from Colab) ---
# This prompt is designed to extract specific fabric information.
EXTRACTION_PROMPT = """
You are an expert textile data extractor. Analyze the text on this fabric sticker.
I need you to extract exactly three pieces of information.

1. Brand Name: Check if it says 'MICHAEL KORS', 'CALVIN KLEIN', 'TALLIA', or something similar. If no brand is listed, leave it blank.
2. Item Number: Look for an alphanumeric code (letters and numbers combined, like 'GJW0032' or 'GSW0399').
3. Fabric Content: Look for percentages and fabric types (e.g., '81% Wool \n 11% Silk'). Clean up any obvious typos.

Return ONLY a raw JSON dictionary. Do NOT include markdown blocks. Use these exact keys:
{"Brand": "Extracted Brand", "Item": "Extracted Item Number", "Content": "Extracted Fabric Percentages"}
"""

def extract_fabric_data(sticker_image_array, retries=3, delay_429=30, delay_error=5):
    """
    Extracts fabric data from a sticker image using the configured AI model.

    Args:
        sticker_image_array (np.array): The cropped sticker image (BGR format).
        retries (int): Number of retries for API calls.
        delay_429 (int): Delay in seconds if a 429 (rate limit) error occurs.
        delay_error (int): Delay in seconds for other API/JSON errors.

    Returns:
        dict: A dictionary containing 'Brand', 'Item', and 'Content', or empty strings if extraction fails.
    """
    ai_data = {"Brand": "", "Item": "", "Content": ""}
    try:
        pil_sticker = PIL.Image.fromarray(cv2.cvtColor(sticker_image_array, cv2.COLOR_BGR2RGB))
    except NameError:
        print("[CRITICAL ERROR]: cv2 is not defined inside extract_fabric_data. Please ensure opencv-python-headless is installed and cv2 is properly imported.")
        raise # Re-raise to halt execution and show the error.

    for attempt in range(retries):
        try:
            response = model.generate_content(
                [EXTRACTION_PROMPT, pil_sticker],
                generation_config={"response_mime_type": "application/json"}
            )
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            ai_data = json.loads(clean_text)
            # Basic validation of keys
            for key in ["Brand", "Item", "Content"]:
                if key not in ai_data: ai_data[key] = ""
            return ai_data

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Quota" in error_msg:
                print(f"[!] AI token limit hit (attempt {attempt+1}/{retries}). Pausing for {delay_429} seconds...")
                time.sleep(delay_429)
            else:
                print(f"[!] JSON or API error (attempt {attempt+1}/{retries}): {error_msg}. Retrying in {delay_error} seconds...")
                time.sleep(delay_error)
    
    print(f"[X] Failed to extract data after {retries} attempts.")
    return ai_data # Return empty data on full failure
