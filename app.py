import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.title("Handwritten Digit Recognizer")

# ⭐ Load model ONLY once
model = load_model("alphanumeric_model.h5")

classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

st.write("Draw a character below:")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:

    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
    img = img.convert('L')
    img = img.resize((28, 28))

    img = np.array(img)

    # ⭐ Check if canvas is empty
    if np.sum(img) < 1000:   # threshold
        st.warning("Draw something ✏️")
        st.stop()

    # ⭐ Flip if needed (mirror fix)
    img = np.fliplr(img)

    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    char = classes[index]
    char = char.upper()
    confidence = np.max(prediction) * 100

    st.subheader(f"Prediction: {char}")

    st.write(f"Confidence: {confidence:.2f}%")
