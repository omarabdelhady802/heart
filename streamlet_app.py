import streamlit as st

st.title("Heart Disease Prediction")

# Flask URL (ensure Flask is running)
flask_url = "http://127.0.0.1:5000"

# Embed the Flask-rendered page
st.components.v1.iframe(flask_url, height=600, scrolling=True)
