import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
rf_classifier = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Images dir
images = {
    "rice": "static/rice.jpg",
    "maize": "static/maize.jpg",
    "chickpea": "static/chickpea.jpg",
    "kidneybeans": "static/kidneybean.jpg",
    "pigeonpeas": "static/pigeonpea.webp",
    "mothbeans": "static/mothbean.webp",
    "mungbean": "static/mungbean.jpg",
    "blackgram": "static/blackgram.jpg",
    "lentil": "static/lentil.jpg",
    "pomegranate": "static/pomegranate.jpg",
    "banana": "static/banana.jpg",
    "mango": "static/mango.jpg",
    "grapes": "static/grapes.jpg",
    "watermelon": "static/watermelon.jpg",
    "muskmelon": "static/muskmelon.jpg",
    "apple": "static/apple.jpg",
    "orange": "static/orange.jpg",
    "papaya": "static/papaya.jpg",
    "coconut": "static/coconut.jpg",
    "cotton": "static/cotton.jpg",
    "jute": "static/jute.jpg",
    "coffee": "static/coffee.jpg"
}

links = {
    "rice": "https://en.wikipedia.org/wiki/Rice",
    "maize": "https://en.wikipedia.org/wiki/Maize",
    "chickpea": "https://en.wikipedia.org/wiki/Chickpea",
    "kidneybeans": "https://en.wikipedia.org/wiki/Kidney_bean",
    "pigeonpeas": "https://en.wikipedia.org/wiki/Pigeon_pea",
    "mothbeans": "https://en.wikipedia.org/wiki/Vigna_aconitifolia",
    "mungbean": "https://en.wikipedia.org/wiki/Mung_bean",
    "blackgram": "https://en.wikipedia.org/wiki/Vigna_mungo",
    "lentil": "https://en.wikipedia.org/wiki/Lentil",
    "pomegranate": "https://en.wikipedia.org/wiki/Pomegranate",
    "banana": "https://en.wikipedia.org/wiki/Banana",
    "mango": "https://en.wikipedia.org/wiki/Mango",
    "grapes": "https://en.wikipedia.org/wiki/Grape",
    "watermelon": "https://en.wikipedia.org/wiki/Watermelon",
    "muskmelon": "https://en.wikipedia.org/wiki/Cantaloupe",
    "apple": "https://en.wikipedia.org/wiki/Apple",
    "orange": "https://en.wikipedia.org/wiki/Orange_(fruit)",
    "papaya": "https://en.wikipedia.org/wiki/Papaya",
    "coconut": "https://en.wikipedia.org/wiki/Coconut",
    "cotton": "https://en.wikipedia.org/wiki/Cotton",
    "jute": "https://en.wikipedia.org/wiki/Jute",
    "coffee": "https://en.wikipedia.org/wiki/Coffea"   
}

# Streamlit app
st.set_page_config(
    page_title="LUPA | Land Utilization and Production Advisor",
    page_icon="🍃",
    initial_sidebar_state="expanded",
    menu_items={
        #'Weather Forecast': 'https://weather.com',
        'Get Help': 'https://www.soilcropadvisor.site/help',
        'Report a bug': "https://www.soilcropadvisor.site/bug",
        'About': "Enter the soil and environmental conditions to predict the best crop."
    }
)

#st.link_button("Go back to the Old Dashboard", "https://www.soilcropadvisor.site")
st.image("static/logo.png")
st.title("Crop Recommendation")
st.write("Enter the soil and environmental conditions to predict the best crop.")

# Ensure the sidebar has content
with st.sidebar:
    st.header("Navigation")
    st.page_link("https://weather.com", label="🌦️ Weather Forecast")
    st.page_link("https://www.soilcropadvisor.site/help", label="📖 Help & Guides")

# Input fields
N = st.number_input("Nitrogen (N) (mg/kg)", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
P = st.number_input("Phosphorus (P) (mg/kg)", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
K = st.number_input("Potassium (K) (mg/kg)", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
ph = st.slider("pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=200.0, step=1.0)

# Prediction
#if st.button("Predict Crop"):
#    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
#    predicted_label = rf_classifier.predict(input_data)
#    predicted_crop = label_encoder.inverse_transform(predicted_label)
#
#    st.info(f"Recommended Crop: {predicted_crop[0]}")
#
#    with st.expander("See more about " + predicted_crop[0]):
#        # Display associated image if available
#        selected_image = images.get(predicted_crop[0])
#        if selected_image:
#            st.image(selected_image, caption=predicted_crop[0], use_container_width=True)

# Prediction
if st.button("Predict Crop", use_container_width=True):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Get probabilities for each class (crop)
    predicted_probs = rf_classifier.predict_proba(input_data)

    # Get the indices of the top 5 crops with the highest probabilities
    top_5_indices = np.argsort(predicted_probs[0])[-5:][::-1]

    # Get the corresponding crop names and their probabilities
    top_5_crops = label_encoder.inverse_transform(top_5_indices)
    top_5_probs = predicted_probs[0][top_5_indices]

    col1, col2 = st.columns(2)
    with col1:
        # Display the associated image for the top predicted crop
        predicted_crop = top_5_crops[0]
        st.info(f"Recommended Crop: {predicted_crop}")

        with st.expander("See more about " + predicted_crop):
            # Display associated image if available
            selected_image = images.get(predicted_crop)
            if selected_image:
                st.image(selected_image, use_container_width=True)

            selected_link = links.get(predicted_crop) 
            if selected_link :
                st.page_link(selected_link, label="Learn More")

    with col2:
        container = st.container(border=True)
        container.subheader("5 Recommended Crops:")
        for crop, prob in zip(top_5_crops, top_5_probs):
            container.write(f"{crop}: {prob * 100:.2f}%")

