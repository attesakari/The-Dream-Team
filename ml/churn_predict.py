import joblib
from flask import Flask, request, jsonify

# Load the model and dv from the file
# dv, model = joblib.load('churn_model.pkl')

# Create a Flask application
app = Flask('churn')

# Define an endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get customer data from the request in JSON format
    customer = request.get_json()

    # Transform the customer data using the loaded data vectorizer
    # X = dv.transform([customer])

    # Make predictions using the loaded model
    # y_pred = model.predict_proba(X)[0, 1]

    # Random y_pred for testing purposes
    y_pred = 0.6
    
    # Determine churn status based on the prediction probability threshold
    churn = y_pred >= 0.5

    # Prepare the result in JSON format
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    # Return the result as a JSON response
    return jsonify(result)

# Run the Flask application
if __name__ == "__main__":
    # Run the app in debug mode on all available network interfaces
    app.run(debug=True, host='0.0.0.0', port=9696)