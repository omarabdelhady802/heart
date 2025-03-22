from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure all form fields are captured
        expected_features = ['cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        features = [float(request.form.get(f, 0)) for f in expected_features]
        
        # Convert input to NumPy array and reshape for prediction
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)
        
        # Interpret prediction
        result = "⚠️ High risk of heart disease! Consult a doctor." if prediction[0] == 1 else "✅ Low risk of heart disease. Maintain a healthy lifestyle!"
    except Exception as e:
        result = f"Error: {str(e)}"
    
    return render_template("index.html", prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
