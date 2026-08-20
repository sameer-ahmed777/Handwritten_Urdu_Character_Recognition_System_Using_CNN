import tensorflow as tf
import cv2
import numpy as np
# Load Model
model = tf.keras.models.load_model("Models/urdu_model.keras")
# Load Classes
with open("Models/class_names.txt", "r") as f:
    classes = [line.strip() for line in f]
# Input Image
image_path = input("Enter image path: ").strip().replace('"', '')
# Read Image
img = cv2.imread(image_path)
if img is None:
    print("❌ Image not found!")
    exit()
# Show Original Image
cv2.imshow("Uploaded Image", img)
# Preprocess
img_resized = cv2.resize(img, (64, 64))
img_resized = img_resized.astype("float32") / 255.0
img_resized = np.expand_dims(img_resized, axis=0)
# Predict
prediction = model.predict(img_resized, verbose=0)
class_index = np.argmax(prediction)
confidence = np.max(prediction) * 100
# Result
print("\n======================")
print("Prediction Result")
print("======================")
print("Class Name :", classes[class_index])
print(f"Confidence : {confidence:.2f}%")
print("======================")
cv2.waitKey(0)
cv2.destroyAllWindows()