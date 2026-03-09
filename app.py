import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np
from predict import predict_identity

st.set_page_config(page_title="Face Recognition ", layout="centered")

st.title("Face Recognition Access Control")
st.write("Upload an image of a person to check if they are in the database and verify their identity.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = "temp_uploaded_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Processing...")

    # Run the prediction
    result = predict_identity(temp_path)
    
    if result is None:
        st.error("Model is untrained or image is invalid. Please train the model first.")
    else:
        closest_person, out_img = result
        
        if closest_person == "Unknown":
            st.error(f"❌ ACCESS DENIED: Image completely unknown or not in trained dataset.")
            st.error("The model restricts access to untrained persons.")
        else:
            st.success(f"✅ Identity Verified: **{closest_person}**")
            
            
            # Display Output Image
            out_img_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
            st.image(out_img_rgb, caption="Processed Result", use_container_width=True)

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
