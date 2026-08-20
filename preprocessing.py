import cv2
import os
import numpy as np
def preprocess_image(image_path):
    # Check image exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Read image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur image
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Apply threshold
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # Resize character
    resized = cv2.resize(
        thresh,
        (200, 200),
        interpolation=cv2.INTER_NEAREST
    )
    # Create white canvas
    canvas = np.ones((800, 800), dtype=np.uint8) * 255
    # Center character
    x = (800 - 200) // 2
    y = (800 - 200) // 2
    canvas[y:y+200, x:x+200] = resized
    return canvas
# ==========================
# MAIN PROGRAM
# ==========================
image_path = input("Enter image path: ")
processed_image = preprocess_image(image_path)
# Full Screen Window
cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Processed Image",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)
cv2.imshow("Processed Image", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()