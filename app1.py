import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from fpdf import FPDF
import os
import uuid
from PIL import Image

# Fix for rerun


# Load pre-trained embeddings and labels
embeddings = np.load('embeddings.npy')
labels_df = pd.read_csv('labels.csv')

# Initialize FaceAnalysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Streamlit App Config and Title
st.set_page_config(page_title="AI Detail Extraction", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Face Detection and Detail Extraction using AI</h1>", unsafe_allow_html=True)

# Refresh button


# File uploader
uploaded_file = st.file_uploader("📤 Upload an Image", type=['jpg', 'jpeg', 'png'])

# For storing matched students
matched_faces = []

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # BGR format

    st.markdown("### 📷 Uploaded Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    faces = app.get(img)

    if len(faces) == 0:
        st.error("❌ No face detected. Please try another image.")
    else:
        for i, face in enumerate(faces):
            st.markdown(f"<hr><h4 style='color:#2196F3;'>👤 Face {i+1} Result</h4>", unsafe_allow_html=True)

            embedding = face.normed_embedding.reshape(1, -1)
            similarities = cosine_similarity(embedding, embeddings)[0]
            max_sim = np.max(similarities)
            max_index = np.argmax(similarities)

            # Crop face
            x1, y1, x2, y2 = map(int, face.bbox)
            face_img = img[y1:y2, x1:x2]
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(face_img_rgb, caption="Detected Face", use_container_width=True)

            with col2:
                if max_sim > 0.4:
                    full_info = labels_df.iloc[max_index]['USN']
                    parts = full_info.split("_")
                    usn_part = parts[0] if len(parts) > 0 else "Unknown"
                    name_part = parts[1] if len(parts) > 1 else "Unknown"
                    dept_part = parts[2] if len(parts) > 2 else "Unknown"

                    st.success(
                        f"✅ **Match Found**\n\n"
                        f"**USN:** {usn_part}\n\n"
                        f"**Name:** {name_part}\n\n"
                        f"**Department:** {dept_part}\n\n"
                        #f"**Similarity Score:** {max_sim:.2f}"
                    )

                    # Save face for PDF report
                    matched_faces.append({
                        "usn": usn_part,
                        "name": name_part,
                        "dept": dept_part,
                        "similarity": max_sim,
                        "face_image": face_img_rgb
                    })
                else:
                    st.warning(
                        f"⚠️ **Relavent Data Not Available(Person may be out of your organization)**\n\n"
                        #f"**Similarity Score:** {max_sim:.2f}"
                    )

        # Generate PDF if there are matches
        if matched_faces:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            for i, match in enumerate(matched_faces):
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(200, 10, txt=f"Matched Face {i+1}", ln=True, align="L")
                pdf.cell(200, 10, txt=f"USN: {match['usn']}", ln=True)
                pdf.cell(200, 10, txt=f"Name: {match['name']}", ln=True)
                pdf.cell(200, 10, txt=f"Department: {match['dept']}", ln=True)
                #pdf.cell(200, 10, txt=f"Similarity Score: {match['similarity']:.2f}", ln=True)

                # Save temp image
                temp_img_path = f"{uuid.uuid4()}.png"
                Image.fromarray(match["face_image"]).save(temp_img_path)
                pdf.image(temp_img_path, x=10, y=60, w=60)
                os.remove(temp_img_path)

            # Save PDF to a bytes buffer
            pdf_output_path = "/tmp/report.pdf"
            pdf.output(pdf_output_path)

            with open(pdf_output_path, "rb") as f:
                st.download_button("📄 Download Final Report (PDF)", f, file_name="match_report.pdf", mime="application/pdf")
