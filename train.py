import tensorflow as tf
import os
# ==========================
# DATASET PATHS
# ==========================
train_dir = "Dataset/characters_train_set"
test_dir = "Dataset/characters_test_set"
# ==========================
# LOAD DATASET
# ==========================
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(64, 64),
    batch_size=32,
    shuffle=True
)
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(64, 64),
    batch_size=32,
    shuffle=False
)
# ==========================
# SAVE CLASS NAMES
# ==========================
os.makedirs("Models", exist_ok=True)
with open("Models/class_names.txt", "w") as f:
    for name in train_data.class_names:
        f.write(name + "\n")
print("Classes Saved Successfully!")
# ==========================
# CNN MODEL
# ==========================
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(
        len(train_data.class_names),
        activation="softmax"
    )
])
# ==========================
# COMPILE MODEL
# ==========================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
print("\nTraining Started...\n")
# ==========================
# TRAIN MODEL
# ==========================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=20
)
# ==========================
# SAVE MODEL
# ==========================
model.save("Models/urdu_model.keras")
print("\n✅ Model Saved Successfully!")
print("✅ Training Completed!")
# ==========================
# FINAL RESULTS
# ==========================
loss, accuracy = model.evaluate(test_data)
print("\n=========================")
print("FINAL RESULT")
print("=========================")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Loss     : {loss:.4f}")
import matplotlib.pyplot as plt
import os
os.makedirs("Results", exist_ok=True)
# Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend([
    "Training Accuracy",
    "Validation Accuracy"
])
plt.savefig("Results/accuracy_graph.png")
plt.close()
# Loss Graph
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend([
    "Training Loss",
    "Validation Loss"
])
plt.savefig("Results/loss_graph.png")
plt.close()
print("✅ Graphs Saved Successfully!")