import requests
# URL for the prediction endpoint
url = 'http://localhost:9696/predict'

# Unique identifier for the customer
customer_id = 'user_x'

# Customer data for making predictions
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 24,
    "monthlycharges": 29.85,
    "totalcharges": (24 * 29.85)
}

# Make a POST request to the prediction endpoint with customer data
response = requests.post(url, json=customer).json()

# Print the prediction response
print(response)

# Check if the predicted churn status is True and take action
if response['churn'] == True:
    print('This user will churn. Send promotional email to user with ID: %s' % customer_id)
else:
    print('This user will continue using our service. ID: %s' % customer_id)