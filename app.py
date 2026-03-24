import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import joblib
import os

st.set_page_config(page_title="Handwritten Digit Recognizer", layout="centered", page_icon="🧠")

if "canvas_id" not in st.session_state:
    st.session_state.canvas_id = 0

def overline(text):
    st.markdown(f"<p style='font-size: 12px; font-weight: bold; color: #808080; margin-bottom: -15px; margin-top: 10px;'>{text.upper()}</p>", unsafe_allow_html=True)
    st.divider()

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff !important; border: 1px solid #ececec !important; padding: 15px; border-radius: 8px; }
    [data-testid="stMetricLabel"] > div { color: #555555 !important; }
    [data-testid="stMetricValue"] > div { color: #1a1a1a !important; }
    [data-testid="stCanvas"] { border: 2px solid #1a1a1a; border-radius: 4px; }
    hr { margin-top: 1rem; margin-bottom: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

if 'model' not in st.session_state:
    if os.path.exists('mnist_model.joblib'):
        st.session_state.model = joblib.load('mnist_model.joblib')
        st.session_state.model.warm_start = False 
    else:
        st.error("System Error: 'mnist_model.joblib' core not found.")
        st.stop()

st.title("MNIST Digit Classifier")
st.caption("Formal Inference Lab & Online Learning System")
st.write("---")

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    overline("Digit Input")
    canvas_result = st_canvas(
        stroke_width=20, 
        stroke_color="#FFFFFF", 
        background_color="#000000",
        height=300, 
        width=300, 
        drawing_mode="freedraw", 
        key=f"canvas_key_{st.session_state.canvas_id}",
        display_toolbar=False
    )
    
    if st.button("Clear Input Field", use_container_width=True):
        st.session_state.canvas_id += 1 
        st.rerun()

features = None

with col_output:
    overline("System Classification")

    if canvas_result.image_data is not None and np.max(cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)) > 50:
        
        img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
        coords = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(coords)
        crop = img[y:y+h, x:x+w]
        pad = max(w, h) + 60
        square = np.zeros((pad, pad), dtype=np.uint8)
        square[(pad-h)//2:(pad-h)//2+h, (pad-w)//2:(pad-w)//2+w] = crop
        img_28 = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        features = img_28.reshape(1, -1) / 255.0

        probs = st.session_state.model.predict_proba(features)[0]
        prediction = np.argmax(probs)
        confidence = probs[prediction]

        st.metric(label="Predicted Digit", value=prediction)
        st.metric(label="Confidence Level", value=f"{confidence*100:.2f}%")
        
        overline("AI Internal View")
        st.image(img_28, width=80)
    else:
        st.info("Awaiting input on drawing surface.")

st.write("---")
overline("System Calibration")

correct_digit = st.radio(
    "If the AI was incorrect, select the actual digit:",
    options=range(10),
    horizontal=True
)

if st.button("Commit Correction & Update Brain", type="primary", use_container_width=True):
    if features is not None:
        all_classes = np.arange(10)
        st.session_state.model.partial_fit(features, np.array([int(correct_digit)]), classes=all_classes)
        joblib.dump(st.session_state.model, 'mnist_model.joblib')
        st.toast(f"Model updated: Image recorded as {correct_digit}", icon="💾")
    else:
        st.warning("Action Denied: No input data detected to calibrate.")