import tensorflow as tf
from tensorflow.keras.models import load_model
# Load Model
model = load_model("Models/urdu_model.keras")
# Load Test Data
test_data = tf.keras.utils.image_dataset_from_directory(
    "Dataset/characters_test_set",
    image_size=(64, 64),
    batch_size=32
)
# Evaluate Model
loss, accuracy = model.evaluate(test_data)
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Loss: {loss:.4f}")