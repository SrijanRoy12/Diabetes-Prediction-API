import streamlit as st
import numpy as np
import pickle
import base64
import time

# 🌌 Background from local file
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    bg_img_style = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        animation: fadein 1.5s ease-in;
    }}
    @keyframes fadein {{
        from {{ opacity: 0; }}
        to   {{ opacity: 1; }}
    }}
    [data-testid="stHeader"], [data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0);
    }}
    [data-testid="stToolbar"] {{
        right: 2rem;
    }}
    .css-1aumxhk {{
        animation: fadein 2s ease-in-out;
    }}
    </style>
    """
    st.markdown(bg_img_style, unsafe_allow_html=True)

# 🖼️ Set the uploaded image as background
set_background("medical.jpg")

# 🎯 Load model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# 🧬 Page configuration
st.set_page_config(page_title="Diabetes Predictor", page_icon="🧬", layout="centered")

# 🌟 Stylish headers
st.markdown("""
<h1 style='text-align: center; color: #FFD700;'>💉 Smart Diabetes Risk Predictor</h1>
<h4 style='text-align: center; color: #FFFFFF;'>Machine Learning meets Medical Insight</h4>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:2px solid #4CAF50;'>", unsafe_allow_html=True)

# 🌈 Input Fields with bold and helpful tooltips
st.markdown("### 🌟 Enter Your Health Parameters", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("👶 **Pregnancies**", min_value=0,
        help="Number of times the patient has been pregnant.")
    
    glucose = st.number_input("🩸 **Glucose Level (mg/dL)**", min_value=0,
        help="Plasma glucose concentration after 2 hours of oral test.")
    
    bp = st.number_input("💓 **Blood Pressure (mm Hg)**", min_value=0,
        help="Diastolic blood pressure in mm Hg.")
    
    skin = st.number_input("📏 **Skin Thickness (mm)**", min_value=0,
        help="Triceps skin fold thickness in millimeters.")

with col2:
    insulin = st.number_input("💉 **Insulin Level (mu U/ml)**", min_value=0,
        help="2-hour serum insulin in micro units per mL.")
    
    bmi = st.number_input("📊 **BMI (Body Mass Index)**", min_value=0.0, format="%.1f",
        help="Body mass index = weight / height² (kg/m²).")
    
    dpf = st.number_input("🧬 **Diabetes Pedigree Function**", min_value=0.0, format="%.2f",
        help="Probability of diabetes based on family history.")
    
    age = st.number_input("🎂 **Age**", min_value=1,
        help="Age of the person in years.")

# 🚀 Custom Predict Button Style
btn_css = """
<style>
div.stButton > button:first-child {
    background-color: #FF4B4B;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    padding: 10px 24px;
    transition: 0.3s ease-in-out;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
div.stButton > button:first-child:hover {
    background-color: #ff2222;
    transform: scale(1.05);
}
</style>
"""
st.markdown(btn_css, unsafe_allow_html=True)

# 🔍 Predict Action
if st.button("🔍 Predict My Diabetes Risk"):
    with st.spinner("Analyzing your data..."):
        time.sleep(1.5)
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.markdown("<hr style='border:2px dashed #2196F3;'>", unsafe_allow_html=True)

        if prediction == 1:
            st.error("⚠️ **High Risk Detected!**")
            st.markdown("🔴 Please consult a healthcare professional as soon as possible.")
        else:
            st.success("✅ **Low Risk Detected!**")
            st.markdown("🟢 You're doing great! Stay healthy and maintain regular checkups.")

        st.markdown("<hr style='border:2px dashed #2196F3;'>", unsafe_allow_html=True)

# 🔗 Social Footer & Floating Icons
footer_css = """
<style>
.floating-icons {
    position: fixed;
    bottom: 30px;
    right: 25px;
    z-index: 1000;
}
.floating-icons a {
    margin: 6px;
    text-decoration: none;
    font-size: 28px;
    transition: transform 0.2s ease-in-out;
    display: inline-block;
    border-radius: 50%;
    padding: 10px;
    color: white;
}
.floating-icons a:hover {
    transform: scale(1.3);
    background: rgba(255, 255, 255, 0.2);
}
.fab.fa-linkedin      { background: #0e76a8; }
.fab.fa-github-square { background: #171515; }
.fa-envelope          { background: #D44638; }
.fab.fa-x-twitter     { background: #000000; }
.fab.fa-instagram     { background: #E1306C; }

.footer-banner {
    margin-top: 100px;
    padding: 12px;
    text-align: center;
    font-weight: bold;
    font-size: 14px;
    border-radius: 12px;
    background: rgba(0,0,0,0.4);
    backdrop-filter: blur(8px);
    color: white;
}
</style>
"""

footer_html = f"""
{footer_css}
<div class="floating-icons">
    <a href="https://www.linkedin.com/in/srijan-roy-iemians/" target="_blank"><i class="fab fa-linkedin"></i></a>
    <a href="https://github.com/SrijanRoy12" target="_blank"><i class="fab fa-github-square"></i></a>
    <a href="https://mail.google.com/mail/?view=cm&fs=1&to=roysrijan53@gmail.com" target="_blank"><i class="fa fa-envelope"></i></a>
    <a href="https://x.com/home" target="_blank"><i class="fab fa-x-twitter"></i></a>
    <a href="https://www.instagram.com/its_ur_roy123/" target="_blank"><i class="fab fa-instagram"></i></a>
</div>

<div class="footer-banner">
    <span style="color: #FF9933;">© 2️⃣0️⃣2️⃣5️⃣ 𝙎𝙧𝙞𝙟𝙖𝙣❜𝙨 𝙋𝙤𝙧𝙩𝙛𝙤𝙡𝙞𝙤 |</span>
    <span style="color: #FFFFFF;"> 𝘾𝙧𝙚𝙖𝙩𝙞𝙣𝙜 𝙬𝙞𝙩𝙝 𝙘𝙤𝙣𝙫𝙞𝙘𝙩𝙞𝙤𝙣, 𝙘𝙤𝙙𝙞𝙣𝙜 𝙬𝙞𝙩𝙝 𝙥𝙪𝙧𝙥𝙤𝙨𝙚 |</span>
    <span style="color: #138808;"> 𝘼𝙡𝙡 𝙧𝙞𝙜𝙝𝙩𝙨 𝙧𝙚𝙨𝙚𝙧𝙫𝙚𝙙</span>
</div>

<!-- Font Awesome -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
"""

st.markdown(footer_html, unsafe_allow_html=True)
