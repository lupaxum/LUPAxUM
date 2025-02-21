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
    "pigeonpeas": "static/pigeonpea.png",
    "mothbeans": "static/mothbean.jpg",
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
    initial_sidebar_state="collapsed",
    menu_items={
        #'Weather Forecast': 'https://weather.com',
        'Get Help': 'https://www.soilcropadvisor.site/help',
        'Report a bug': "https://www.soilcropadvisor.site/bug",
        'About': "Enter the soil and environmental conditions to predict the best crop."
    }
)

#st.link_button("Go back to the Old Dashboard", "https://www.soilcropadvisor.site")
st.image("static/logo.png")
st.title("CROP RECOMMENDATION")
st.write("Enter the soil and environmental conditions to predict the best crop.")

# Ensure the sidebar has content
with st.sidebar:
    st.header("> MORE")
    st.page_link("https://weather.com", label="🌦️ Weather Forecast")

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


crop_name_mapping = {
    "rice" : "Rice",
    "chickpea" : "Chick Peas",
    "kidneybeans": "Kidney Beans",
    "maize": "Maize (Corn)",
    "pigeonpeas": "Pigeon Peas",
    "mothbeans": "Moth Beans",
    "mungbean": "Mung Beans",
    "blackgram": "Black Gram",
    "lentil": "Lentils",
    "jute": "Jute",
    "cotton": "Cotton",
    "coconut": "Coconut",
    "pomegranate" : "Pomegranate",
    "banana" : "Banana",
    "mango" : "Mango",
    "grapes" : "Grapes",
    "watermelon" : "Watermelon",
    "muskmelon" : "Muskmelon",
    "apple" : "Apple",
    "orange" : "Orange",
    "papaya" : "Papaya",
    "coffee" : "Coffee"
}

def get_readable_crop_name(crop_key):
    return crop_name_mapping.get(crop_key, crop_key)

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
        readable_predicted_crop = get_readable_crop_name(top_5_crops[0])
        predicted_crop = top_5_crops[0]
        st.markdown(
            f"""
            <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #155724; padding-bottom: 5px;">
                <h5 style="margin: 0; padding-bottom: 2px;">Recommended Crop</h5>
                <p style="font-size: 24px; font-weight: bold; margin: 0;">{readable_predicted_crop}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display associated image if available
        selected_image = images.get(predicted_crop)
        if selected_image:
            st.image(selected_image, use_container_width=True)

        # Display "Learn More" as a button
        selected_link = links.get(predicted_crop)
        if selected_link:
            st.markdown(
                f"""
                <a href="{selected_link}" target="_blank" 
                style="display: inline-block; padding: 8px 12px; background-color: #d4edda; 
                        color: black; text-decoration: none; border-radius: 5px; font-weight: bold;">
                    Learn More
                </a>
                """,
                unsafe_allow_html=True
            )

    #with col2:
    #    container = st.container(border=True)
     #   container.subheader("5 Recommended Crops")
#
 #       for crop, prob in zip(top_5_crops, top_5_probs):
  #          readable_crop = get_readable_crop_name(crop)  # Convert shortcut name
#
 #           # Determine text color based on probability
  #          percentage = prob * 100
   #         if percentage >= 50:
    #            color = "darkgreen"
     #       elif 20 <= percentage < 50:
      #          color = "darkgoldenrod"  # Dark Yellow
       #     else:
        #        color = "darkred"

            # Format output with bold text and colored percentage
         #   container.markdown(
          #      f"{readable_crop}: <span style='color:{color}; font-weight:bold;'>{percentage:.2f}%</span>", 
           #     unsafe_allow_html=True
            #)

    with col2:
        container = st.container(border=True)
        container.subheader("5 Recommended Crops")

        classifications = {
            "Good for Planting ✅": {"range": (80, 100), "color": "darkgreen", "description": 
                "The soil is in excellent condition and supports healthy crop growth with minimal maintenance required."},
            "Fair/Manageable 🌤️": {"range": (50, 79), "color": "goldenrod", "description": 
                "The land is suitable for planting but requires minor adjustments for optimal yield."},
            "Needs Improvement ⚠️": {"range": (16, 49), "color": "darkorange", "description": 
                "The soil has potential but requires intervention such as fertilizers, compost, or irrigation adjustments to be viable."},
            "Not Recommended 🚨": {"range": (0, 15), "color": "darkred", "description": 
                "The soil is highly unsuitable for planting and needs major rehabilitation, such as pH correction and organic matter addition."}
        }

        categorized_crops = {key: [] for key in classifications.keys()}

        for crop, prob in zip(top_5_crops, top_5_probs):
            readable_crop = get_readable_crop_name(crop)  
            percentage = prob * 100  

            for category, details in classifications.items():
                if details["range"][0] <= percentage <= details["range"][1]:
                    categorized_crops[category].append((readable_crop, percentage, details["color"]))
                    break  

        for category, crops in categorized_crops.items():
            if crops:
                container.markdown(f"<h3 style='color:black; margin-bottom:5px;'>{category}</h3>", unsafe_allow_html=True)
                
                for crop, percentage, color in crops:
                    container.markdown(
                        f"<p style='font-size:18px; font-weight:bold; color:{color}; margin:3px 0; line-height:1.2;'>{crop}: {percentage:.2f}%</p>", 
                        unsafe_allow_html=True
                    )

                container.markdown(f"<p style='font-size:15px; color:black; margin-top:2px;'>{classifications[category]['description']}</p>", unsafe_allow_html=True)
