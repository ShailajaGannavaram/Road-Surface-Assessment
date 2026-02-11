import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import tempfile
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Intelligent Road Surface Quality Assessment",
    layout="centered",
)

# ---------------- SESSION STATE ----------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "results" not in st.session_state:
    st.session_state.results = None
if "annotated" not in st.session_state:
    st.session_state.annotated = None
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

# ---------------- CUSTOM CSS (BLUE / GREEN THEME) ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e3f2fd, #e8f5e9);
}
.block-container {
    background-color: #ffffff;
    padding: 2.5rem;
    border-radius: 22px;
    box-shadow: 0px 12px 30px rgba(0,0,0,0.12);
}
h1 { color: #0d47a1; text-align: center; }
h2, h3 { color: #1b5e20; }
.low { color: #2e7d32; font-weight: bold; }
.moderate { color: #ef6c00; font-weight: bold; }
.severe { color: #c62828; font-weight: bold; }
.report-box {
    background-color: #f1f8ff;
    padding: 18px;
    border-radius: 15px;
    margin-top: 20px;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🚧 Intelligent Road Surface Quality Assessment System")
st.write("""
An **AI-powered decision support system** that detects road surface defects,  
evaluates severity, and provides **reconstruction & maintenance guidance**.
""")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()  # Stop app if model not loaded

# ---------------- HELPER FUNCTIONS ----------------
def assess_severity(conf):
    if conf < 0.40:
        return "Low", "low", 0.3, "Minor surface damage with no immediate safety risk."
    elif conf < 0.70:
        return "Moderate", "moderate", 0.6, "Damage may worsen and affect ride quality."
    else:
        return "Severe", "severe", 0.9, "Critical damage posing serious safety risks."

def recommendation(label, severity):
    actions = {
        "crack": {
            "Low": "Seal cracks to prevent water penetration.",
            "Moderate": "Crack filling followed by surface overlay.",
            "Severe": "Immediate resurfacing required."
        },
        "pothole": {
            "Low": "Temporary patch repair recommended.",
            "Moderate": "Permanent patch repair with compaction.",
            "Severe": "Urgent full-depth reconstruction required."
        },
        "manhole": {
            "Low": "Routine inspection advised.",
            "Moderate": "Level adjustment and reinforcement needed.",
            "Severe": "Immediate structural correction required."
        }
    }
    return actions.get(label, {}).get(severity, "Standard maintenance recommended.")

def reconstruction_steps(severity):
    return {
        "Low": """
**Reconstruction Process (Low Severity)**  
1. Visual inspection  
2. Cleaning debris  
3. Crack sealing / minor patching  
4. Periodic monitoring  
""",
        "Moderate": """
**Reconstruction Process (Moderate Severity)**  
1. Detailed site inspection  
2. Marking damaged zones  
3. Crack filling / patch repair  
4. Surface overlay  
5. Compaction and leveling  
""",
        "Severe": """
**Reconstruction Process (Severe Severity)**  
1. Traffic diversion and safety barricades  
2. Removal of damaged layers  
3. Structural base repair  
4. Full resurfacing  
5. Quality and load inspection  
"""
    }[severity]

# ---------------- IMAGE UPLOAD & ANALYSIS ----------------
uploaded_file = st.file_uploader(
    "📤 Upload a road surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Resize large images for YOLO
        max_size = (1024, 1024)
        image.thumbnail(max_size)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyze Road Condition"):
            if model is None:
                st.error("Model not loaded.")
            else:
                with st.spinner("Running intelligent analysis..."):
                    try:
                        # Save temp image
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            image.save(tmp.name)
                            img_path = tmp.name

                        # Run YOLO inference
                        st.session_state.results = model(img_path)
                        st.session_state.annotated = Image.fromarray(
                            st.session_state.results[0].plot()
                        )
                        st.session_state.analysis_done = True
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")

    except UnidentifiedImageError:
        st.error("⚠️ Uploaded file is not a valid image. Please upload JPG, JPEG, or PNG.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# ---------------- DISPLAY RESULTS ----------------
if st.session_state.analysis_done:
    st.success("✅ Analysis Complete")
    st.image(
        st.session_state.annotated,
        caption="Detected Road Defects",
        use_container_width=True
    )

    st.subheader("🧠 Detailed Damage Assessment")

    issues = []
    boxes = st.session_state.results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            severity, css, score, explanation = assess_severity(conf)
            action = recommendation(label, severity)

            issues.append(
                f"{label.upper()} | Severity: {severity}\n"
                f"Condition: {explanation}\n"
                f"Action: {action}"
            )

            st.markdown(f"""
            <div class="report-box">
            <b>Damage {i+1}</b><br>
            <b>Type:</b> {label.upper()}<br>
            <b>Confidence:</b> {conf:.2f}<br>
            <b>Severity:</b> <span class="{css}">{severity}</span><br>
            <b>Condition:</b> {explanation}<br>
            <b>Recommended Action:</b> {action}
            </div>
            """, unsafe_allow_html=True)

            st.write("**Severity Level**")
            st.progress(score)

            with st.expander(f"🛠 Reconstruction Details – Damage {i+1}"):
                st.markdown(reconstruction_steps(severity))

    else:
        st.success("✅ No road damage detected.")

    # ---------------- REPORT DOWNLOAD ----------------
    st.session_state.report_text = f"""
ROAD CONDITION ANALYSIS REPORT
------------------------------
Date & Time: {datetime.now()}

Detected Issues:
----------------
""" + "\n\n".join(issues)

    st.download_button(
        "⬇ Download Final Analysis Report",
        st.session_state.report_text,
        file_name="road_surface_analysis_report.txt"
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center><small>AI-based Road Monitoring & Decision Support System | Internship Project</small></center>",
    unsafe_allow_html=True
)
