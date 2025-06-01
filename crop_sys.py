import streamlit as st
import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load("best_crop_model.pkl")
le = joblib.load("label_encoder.pkl")

st.title("Smart Crop Recommendation System 🌱")
st.write("Enter soil and climate details manually or upload a CSV file to get the best crop recommendation for your farm.")

# Upload CSV file
uploaded_file = st.file_uploader(
    "📁 Upload CSV file with columns: Nitrogen, Phosphorus, Potassium, temperature, humidity, ph, rainfall", 
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Read uploaded CSV
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("File uploaded successfully! ✅")
        st.dataframe(df_uploaded)

        # Map possible column names to required columns
        col_name_map = {
            'n': 'Nitrogen',
            'nitrogen': 'Nitrogen',
            'p': 'Phosphorus',
            'phosphorus': 'Phosphorus',
            'k': 'Potassium',
            'potassium': 'Potassium',
            'temp': 'temperature',
            'temperature': 'temperature',
            'humid': 'humidity',
            'humidity': 'humidity',
            'ph': 'ph',
            'rainfall': 'rainfall',
            'rain': 'rainfall'
        }

        # Rename columns to standard names for model
        df_uploaded.columns = [col_name_map.get(col.lower(), col) for col in df_uploaded.columns]

        required_cols = ['Nitrogen','Phosphorus','Potassium','temperature','humidity','ph','rainfall']
        missing_cols = [col for col in required_cols if col not in df_uploaded.columns]
        if missing_cols:
            st.error(f"❌ CSV missing columns: {missing_cols}")
        else:
            df_filtered = df_uploaded[required_cols]
            # Predict crop for each row
            for idx, row in df_filtered.iterrows():
                input_data = pd.DataFrame([row], columns=required_cols)
                prediction = model.predict(input_data)
                crop_name = le.inverse_transform(prediction)[0]
                st.success(f"🌱{idx+1} → Recommended crop: **{crop_name}**")

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")

# Manual input section
st.subheader("Or enter values manually:")

Nitrogen = st.number_input("Nitrogen content in the soil (N):", min_value=0.0, step=1.0)
Phosphorus = st.number_input("Phosphorus content in the soil (P):", min_value=0.0, step=1.0)
Potassium = st.number_input("Potassium content in the soil (K):", min_value=0.0, step=1.0)
Temperature = st.number_input("Temperature (°C):", min_value=-60.0, step=0.1)
Humidity = st.number_input("Humidity (%):", step=0.1)
Ph = st.number_input("Soil pH level:", min_value=0.0, max_value=14.0, step=0.1)
Rainfall = st.number_input("Rainfall (mm):", min_value=0.0, step=0.1)

if st.button("Predict Best Crop 🔍"):
    # Validate inputs before predicting
    if 0 in [Nitrogen, Phosphorus, Potassium]:
        st.error("❌ Please complete all input fields.")
    elif Nitrogen < 0:
        st.error("❌ Nitrogen must be positive.")
    elif Phosphorus < 0:
        st.error("❌ Phosphorus must be positive.")
    elif Potassium < 0:
        st.error("❌ Potassium must be positive.")
    elif Humidity < 0 or Humidity > 100:
        st.error("❌ Humidity must be 0-100.")
    elif Ph < 0 or Ph > 14:
        st.error("❌ pH must be 0-14.")
    elif Rainfall < 0:
        st.error("❌ Rainfall must be positive.")
    else:
        # Make prediction
        input_data = pd.DataFrame([[Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]],
                                  columns=['Nitrogen','Phosphorus','Potassium','temperature','humidity','ph','rainfall'])
        prediction = model.predict(input_data)
        crop_name = le.inverse_transform(prediction)[0]
        st.success(f"✅ Optimal crop: **{crop_name}**")