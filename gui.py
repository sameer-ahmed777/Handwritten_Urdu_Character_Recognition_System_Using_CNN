import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
# =========================
# LOAD MODEL
# =========================
model = load_model("Models/urdu_model.keras")
with open("Models/class_names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
image_path = ""
# =========================
# FUNCTIONS
# =========================
def upload_image():
    global image_path
    image_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not image_path:
        return
    img = Image.open(image_path)
    img = img.resize((450, 450))
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)
    image_label.image = photo
    history_list.insert(tk.END, image_path.split("/")[-1])
def predict_image():
    global image_path
    if image_path == "":
        messagebox.showwarning(
            "Warning",
            "Please Upload Image First!"
        )
        return
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (64, 64))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img, verbose=0)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        result_label.config(
            text=f"""
Predicted Character : {classes[class_index]}
Confidence : {confidence:.2f} %
Status : Successfully Recognized
""",
            fg="#16a34a"
        )
    except Exception as e:
        result_label.config(
            text=f"Error: {e}",
            fg="red"
        )
def clear_image():
    global image_path
    image_path = ""
    image_label.config(image="")
    image_label.image = None
    result_label.config(
        text="Upload Image To Start",
        fg="#2563eb"
    )
# =========================
# MAIN WINDOW
# =========================
root = tk.Tk()
root.title("Urdu Character Recognition Dashboard")
root.geometry("1400x800")
root.configure(bg="#0f172a")
# =========================
# SIDEBAR
# =========================
sidebar = tk.Frame(
    root,
    bg="#111827",
    width=260
)
sidebar.pack(
    side="left",
    fill="y"
)
menu_label = tk.Label(
    sidebar,
    text="MENU",
    font=("Arial", 20, "bold"),
    bg="#111827",
    fg="white"
)
menu_label.pack(pady=20)
upload_btn = tk.Button(
    sidebar,
    text="📁 Upload New",
    width=22,
    bg="#2563eb",
    fg="white",
    command=upload_image
)
upload_btn.pack(pady=8)
predict_btn = tk.Button(
    sidebar,
    text="🔍 Prediction",
    width=22,
    bg="#16a34a",
    fg="white",
    command=predict_image
)
predict_btn.pack(pady=8)
clear_btn = tk.Button(
    sidebar,
    text="🗑 Clear",
    width=22,
    bg="#f59e0b",
    fg="white",
    command=clear_image
)
clear_btn.pack(pady=8)
history_title = tk.Label(
    sidebar,
    text="Upload History",
    font=("Arial", 14, "bold"),
    bg="#111827",
    fg="white"
)
history_title.pack(pady=20)
history_list = tk.Listbox(
    sidebar,
    width=25,
    height=15,
    font=("Arial", 10)
)
history_list.pack(pady=5)
exit_btn = tk.Button(
    sidebar,
    text="❌ Exit",
    width=22,
    bg="#dc2626",
    fg="white",
    command=root.destroy
)
exit_btn.pack(
    side="bottom",
    pady=20
)
# =========================
# MAIN CONTENT
# =========================
content = tk.Frame(
    root,
    bg="#0f172a"
)
content.pack(
    side="right",
    fill="both",
    expand=True
)
title = tk.Label(
    content,
    text="Urdu Handwritten Character Recognition System",
    font=("Arial", 24, "bold"),
    bg="#0f172a",
    fg="white"
)
title.pack(pady=20)
# =========================
# IMAGE PREVIEW
# =========================
preview_frame = tk.Frame(
    content,
    bg="white",
    bd=4,
    relief="ridge"
)
preview_frame.pack(pady=15)
image_label = tk.Label(
    preview_frame,
    text="IMAGE PREVIEW",
    bg="white",
    width=55,
    height=18,
    font=("Arial", 12, "bold")
)
image_label.pack()
# =========================
# RESULT CARD
# =========================
result_frame = tk.Frame(
    content,
    bg="white",
    bd=4,
    relief="ridge"
)
result_frame.pack(pady=20)
heading = tk.Label(
    result_frame,
    text="Prediction Result",
    font=("Arial", 18, "bold"),
    bg="white"
)
heading.pack(pady=10)
result_label = tk.Label(
    result_frame,
    text="Upload Image To Start",
    font=("Arial", 16, "bold"),
    bg="white",
    fg="#2563eb",
    justify="center",
    width=45,
    height=5
)
result_label.pack(
    padx=20,
    pady=15
)
# =========================
# FOOTER
# =========================
footer = tk.Label(
    content,
    text="Developed By Sameer Ahmed",
    bg="#0f172a",
    fg="white",
    font=("Arial", 12, "bold")
)
footer.pack(
    side="bottom",
    pady=10
)
root.mainloop()