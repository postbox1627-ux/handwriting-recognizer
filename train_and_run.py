import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import numpy as np
import os

# ===============================
# 1. LOAD EMNIST DATASET
# ===============================

print("Loading dataset...")

(ds_train, ds_test), ds_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

def ds_to_numpy(ds):
    images = []
    labels = []
    for img, label in tfds.as_numpy(ds):
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

x_train, y_train = ds_to_numpy(ds_train)
x_test, y_test = ds_to_numpy(ds_test)

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# FIX EMNIST orientation (correct version)
x_train = np.transpose(x_train, (0, 2, 1, 3))
x_test = np.transpose(x_test, (0, 2, 1, 3))

x_train = np.flip(x_train, axis=2)
x_test = np.flip(x_test, axis=2)

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ===============================
# 2. MODEL
# ===============================

import os
from tensorflow.keras.models import load_model

MODEL_FILE = "alphanumeric_model.h5"

if os.path.exists(MODEL_FILE):
    print("Loading saved model...")
    model = load_model(MODEL_FILE)

# ===============================
# 3. BUILD and TRAIN MODEL
# ===============================

else:
    print("Training model...")

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),

        layers.Dense(62, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=20,
              validation_data=(x_test, y_test))

    model.save(MODEL_FILE)
    print("Model saved!")


# ===============================
# 4. DRAWING WINDOW
# ===============================

import tkinter as tk
from PIL import Image, ImageDraw

classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

class Drawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Alphanumeric Recognizer")

        self.canvas = tk.Canvas(root, width=350, height=350, bg='black')
        self.canvas.pack()

        self.result = tk.Label(root, text="Prediction:",
                               font=("Arial", 24))
        self.result.pack(pady=20)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Predict",
                  command=self.predict).grid(row=0, column=0, padx=10)

        tk.Button(btn_frame, text="Clear",
                  command=self.clear).grid(row=0, column=1, padx=10)

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new("L", (350, 350), 'black')
        self.draw_img = ImageDraw.Draw(self.image)

    def draw(self, event):
        x1, y1 = event.x - 10, event.y - 10
        x2, y2 = event.x + 10, event.y + 10

        self.canvas.create_oval(x1, y1, x2, y2,
                                fill='white', outline='white')

        self.draw_img.ellipse([x1, y1, x2, y2], fill='white')

    def clear(self):
        self.canvas.delete("all")
        self.draw_img.rectangle([0, 0, 350, 350], fill='black')
        self.result.config(text="Prediction:")

    def predict(self):
        bbox = self.image.getbbox()

        if bbox:
            img = self.image.crop(bbox)
        else:
            img = self.image

        img = img.resize((28, 28))

        img = np.fliplr(img)

        # Make strokes thinner
        img = img / 255.0

        #  orientation fix as training
       
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        index = np.argmax(prediction)
        char = classes[index]
        char = char.upper()
        confidence = np.max(prediction) * 100

        self.result.config(
            text=f"Prediction: {char} ({confidence:.2f}%)"
        )


# Run GUI
root = tk.Tk()
app = Drawer(root)
root.mainloop()
