import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# 🌈 PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Rainfall Intelligence System",
    page_icon="🌧️",
    layout="wide"
)

# =====================================================
# 🎨 CUSTOM COLORFUL STYLE
# =====================================================
st.markdown("""
<style>
.main {
    background-color: #f0f8ff;
}
h1 {
    color: #0a4d8c;
}
.stButton>button {
    background-color: #007acc;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stMetric {
    background-color: #e6f2ff;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 📂 BASE DIRECTORY (SRC IMAGES)
# =====================================================
BASE_DIR = os.path.dirname(__file__)
RAIN_IMG = os.path.join(BASE_DIR, "Rainfall.jpg")
SUNNY_IMG = os.path.join(BASE_DIR, "sunny.jpg")
UMBRELLA_IMG = os.path.join(BASE_DIR, "umbrella.jpg")

# =====================================================
# 🤖 TRAIN MODEL
# =====================================================
np.random.seed(42)
X = pd.DataFrame({
    "Temperature": np.random.randint(20, 40, 400),
    "Humidity": np.random.randint(30, 100, 400),
    "WindSpeed": np.random.randint(1, 20, 400),
    "Pressure": np.random.randint(980, 1030, 400)
})
y = (X["Humidity"] > 70).astype(int)

model = RandomForestClassifier()
model.fit(X, y)

# =====================================================
# 🌍 WEATHER API
# =====================================================
API_KEY = "d2f41e80da8b28303d10ebf3bc468373"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()
    return None

# =====================================================
# 📌 SIDEBAR
# =====================================================
menu = st.sidebar.radio("🌈 Navigation Menu", [
    "🏠 Home",
    "📍 Weather & Suggestion",
    "📊 Trends & Analytics",
    "🧠 Model Validation"
])

# =====================================================
# 🏠 HOME
# =====================================================
if menu == "🏠 Home":

    st.title("🌧️ Rainfall Intelligence System")

    if os.path.exists(RAIN_IMG):
        st.image(RAIN_IMG, width=700)

    st.markdown("""
    ### 👩‍🎓 Developed By  
    **Daksh Yadav**  
    (year-2026)
    B.Tech – computer science and design 
                
                

    ---
    ## 🎯 Project Objective
    To predict rainfall using Machine Learning and Live Weather API.

    ## 🚀 Key Features
    ✔ Real-time weather dashboard  
    ✔ ML-based rainfall prediction  
    ✔ Trend visualization  
    ✔ Smart weather recommendation  
    """)

# =====================================================
# 📍 WEATHER + SUGGESTION IN SAME SLIDE
# =====================================================
elif menu == "📍 Weather & Suggestion":

    st.header("🌍 Live Weather Dashboard")

    city = st.text_input("Enter City Name")

    if st.button("Fetch Weather Data") and city:

        weather = get_weather(city)

        if weather:
            temp = weather["main"]["temp"]
            humidity = weather["main"]["humidity"]
            pressure = weather["main"]["pressure"]
            wind = weather["wind"]["speed"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🌡 Temp (°C)", temp)
            col2.metric("💧 Humidity (%)", humidity)
            col3.metric("🌬 Wind", wind)
            col4.metric("📉 Pressure", pressure)

            st.session_state["data"] = [temp, humidity, wind, pressure]

        else:
            st.error("Invalid City or API Key")

    # 🔥 Suggestion button inside same slide
    if "data" in st.session_state:

        if st.button("Generate Smart Suggestion"):

            temp, humidity, wind, pressure = st.session_state["data"]

            input_df = pd.DataFrame(
                [[temp, humidity, wind, pressure]],
                columns=["Temperature", "Humidity", "WindSpeed", "Pressure"]
            )

            prediction = model.predict(input_df)[0]

            if prediction == 1:
                st.success("🌧 Rain Expected")
                if os.path.exists(UMBRELLA_IMG):
                    st.image(UMBRELLA_IMG, width=300)
                st.write("☔ Recommendation: Carry an umbrella.")

            else:
                st.success("☀ No Rain Expected")
                if os.path.exists(SUNNY_IMG):
                    st.image(SUNNY_IMG, width=300)
                st.write("😎 Recommendation: Enjoy your day!")

# =====================================================
# 📊 TRENDS & ANALYTICS
# =====================================================
elif menu == "📊 Trends & Analytics":

    st.subheader("📈 15-Day Temperature Trend")

    days = np.arange(1, 16)
    temp_trend = np.random.randint(20, 35, 15)

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(days, temp_trend)
    ax.set_xlabel("Day")
    ax.set_ylabel("Temperature")
    st.pyplot(fig)

    st.subheader("🌧 Rain Probability Distribution")

    rain_prob = np.random.randint(0, 100)

    fig2, ax2 = plt.subplots(figsize=(4,4))
    ax2.pie([rain_prob, 100-rain_prob],
            labels=["Rain", "No Rain"],
            autopct="%1.1f%%")
    st.pyplot(fig2)

# =====================================================
# 🧠 MODEL VALIDATION
# =====================================================
elif menu == "🧠 Model Validation":

    st.subheader("📊 Model Performance")

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    st.metric("Accuracy", f"{acc*100:.2f}%")

    st.write("Model Used: Random Forest Classifier")
    st.write("Rain prediction primarily influenced by humidity feature.")
