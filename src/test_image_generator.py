import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

def create_test_image(text="TEST", dimensions=(600, 400), background_color=(200, 200, 200), sticker_color=(255, 255, 255), text_color=(0, 0, 0)):
    """
    Generates a synthetic test image with a fabric-like background and a readable sticker.
    Returns the image as a base64 encoded string.
    """
    img = np.full((*dimensions, 3), background_color, dtype=np.uint8)

    # Simulate fabric texture (simple noise)
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int8)
    img = cv2.add(img, noise.astype(np.uint8))
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Draw a white sticker at the top-center
    sticker_w = int(dimensions[1] * 0.4)  # 40% of width
    sticker_h = int(dimensions[0] * 0.1)  # 10% of height
    sticker_x = (dimensions[1] - sticker_w) // 2
    sticker_y = int(dimensions[0] * 0.05) # 5% from top
    cv2.rectangle(img, (sticker_x, sticker_y), (sticker_x + sticker_w, sticker_y + sticker_h), sticker_color, -1)

    # Add border to sticker
    cv2.rectangle(img, (sticker_x, sticker_y), (sticker_x + sticker_w, sticker_y + sticker_h), (100, 100, 100), 2)

    # Add text to sticker
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = sticker_x + (sticker_w - text_size[0]) // 2
    text_y = sticker_y + (sticker_h + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Encode to base64
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

if __name__ == "__main__":
    # Example usage: Generate a test image with specific sticker text
    sample_text = "Brand: TestCo\nItem: TST001\nContent: 100% Cotton"
    base64_image = create_test_image(sample_text)
    print(f"Generated base64 image (first 100 chars): {base64_image[:100]}...")
    
    # To save and view the image:
    # with open("test_fabric_swatch.png", "wb") as f:
    #     f.write(base64.b64decode(base64_image))
