import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Defect descriptions
defect_info = {
    "damged": "Physical damage found on PCB component — may cause circuit failure",
    "Short_circuit": "Two conductors are unintentionally connected — can burn the board",
    "lack_of_part": "A required component is missing from the PCB",
    "miss_welding": "Solder joint is incomplete or missing — poor electrical connection",
    "redundant": "Extra unwanted component found on PCB",
    "slug": "Foreign material or excess solder detected on board",
    "spillover": "Solder has spread beyond intended area"
}

st.title("AI-AOI Inspection Dashboard")
st.write("Upload a PCB image to detect defects using AI!")

uploaded_file = st.file_uploader("Choose a PCB image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    img_array = np.array(image)

    results = model.predict(img_array, conf=0.05)

    result_img = results[0].plot()
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    defect_count = len(results[0].boxes)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image)
    with col2:
        st.subheader("AI Detection Result")
        st.image(result_img_rgb)

    st.subheader("Inspection Result:")
    if defect_count == 0:
        st.success("PASS - No defects found!")
    else:
        st.error(f"FAIL - {defect_count} defect(s) found!")

    st.metric("Total Defects Found", defect_count)

    if defect_count > 0:
        st.subheader("Defect Details:")
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            reason = defect_info.get(label, "Unknown defect detected")
            st.error(f"Defect Type: **{label}**")
            st.write(f"Confidence: {conf:.0%}")
            st.write(f"Reason: {reason}")
            st.write("---")