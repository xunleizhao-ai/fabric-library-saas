import cv2
import numpy as np

# =========================================================================
# 🛠️ THE CROP DIALS (tuned from Colab)
# =========================================================================
WHITE_THRESHOLD = 200
MIN_STICKER_AREA = 5000
JUMP_DOWN_PIXELS = 20
SWATCH_SQUARE_SIZE = 300

def find_stickers_and_swatches(image_array):
    """
    Detects stickers and crops corresponding fabric swatches from an image.
    Handles multiple stickers per image.

    Args:
        image_array (np.array): The input image as a NumPy array (BGR format).

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries, where each dictionary contains:
                    - 'sticker_crop' (np.array): The cropped image of the sticker.
                    - 'swatch_crop' (np.array): The cropped image of the fabric swatch.
                    - 'debug_rects' (list): List of rectangles (x,y,w,h) for debugging overlay.
            - np.array: A debug image with detected contours and bounding boxes drawn.
    """
    debug_img = image_array.copy()
    img = image_array.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_stickers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_STICKER_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter for typically rectangular stickers (e.g., width significantly greater than height)
            if w > h * 1.5: 
                valid_stickers.append({'x': x, 'y': y, 'w': w, 'h': h})

    valid_stickers.sort(key=lambda b: b['y']) # Sort by Y-coordinate to process top-down

    results = []
    for sticker in valid_stickers:
        sx, sy, sw, sh = sticker['x'], sticker['y'], sticker['w'], sticker['h']

        # Crop sticker with a small margin
        margin = 5
        sticker_crop = img[max(0, sy+margin):min(img.shape[0], sy+sh-margin), 
                           max(0, sx+margin):min(img.shape[1], sx+sw-margin)]
        
        # Determine fabric swatch crop area relative to the sticker
        sticker_bottom_y = sy + sh
        sticker_center_x = sx + (sw // 2)

        fab_y1 = sticker_bottom_y + JUMP_DOWN_PIXELS
        fab_y2 = min(img.shape[0], fab_y1 + SWATCH_SQUARE_SIZE)
        fab_x1 = max(0, sticker_center_x - (SWATCH_SQUARE_SIZE // 2))
        fab_x2 = min(img.shape[1], fab_x1 + SWATCH_SQUARE_SIZE)

        swatch_crop = img[fab_y1:fab_y2, fab_x1:fab_x2]

        if swatch_crop.shape[0] == 0 or sticker_crop.shape[0] == 0:
            continue # Skip if crops are empty

        cv2.rectangle(debug_img, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 3) # Blue for sticker
        cv2.rectangle(debug_img, (fab_x1, fab_y1), (fab_x2, fab_y2), (0, 255, 0), 3) # Green for swatch

        results.append({
            'sticker_crop': sticker_crop,
            'swatch_crop': swatch_crop,
        })
    return results, debug_img

def resize_image_for_api(img_array, max_dim=512):
    """
    Resizes an image (NumPy array) to fit within max_dim while maintaining aspect ratio,
    to reduce token usage for API calls.
    """
    h, w = img_array.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img_array
