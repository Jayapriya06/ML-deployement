from flask import Flask, request, jsonify
import joblib


# Load the trained Logistic Regression model
loaded_model = joblib.load('logistic_regression_model.joblib')

# Load the fitted TF-IDF vectorizer
loaded_vectorizer = joblib.load('tfidf_vectorizer.joblib')

print("Logistic Regression model and TF-IDF vectorizer loaded successfully.")
# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
loaded_model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text preprocessing function
def preprocess_text_for_prediction(text):
    return vectorizer.transform([text])

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text in request body'}), 400

    input_text = data['text']

    processed_features = preprocess_text_for_prediction(input_text)

    prediction_numerical = loaded_model.predict(processed_features)[0]

    prediction_label = 'true' if prediction_numerical == 1 else 'fake'

    return jsonify({'prediction': prediction_label})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
