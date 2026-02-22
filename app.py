import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

st.title("Handwritten Digit Recognizer")

# Load model ONLY once
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
    display_toolbar=False,   # HIDE ICONS
    key=f"canvas_{st.session_state.canvas_key}",
)

col1, col2 = st.columns(2)

with col1:
    predict_clicked = st.button("🔍 Predict")

with col2:
    clear_clicked = st.button("🧹 Clear")

if clear_clicked:
    st.session_state.canvas_key += 1
    st.rerun()

# ⭐ Predict button
if predict_clicked:

    if canvas_result.image_data is None:
        st.warning("Draw something first ✏️")
    else:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        img = img.convert('L')
        img = img.resize((28, 28))

        img = np.array(img)

        # ⭐ Check if canvas is empty
        if np.sum(img) < 1000:   # threshold
            st.warning("Draw something ✏️")
            st.stop()
    
        # Flip if needed (mirror fix)
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
