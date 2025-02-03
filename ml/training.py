# Importing the Pandas library for data manipulation
import pandas as pd

# Importing the NumPy library for numerical operations
import numpy as np

# Importing the train_test_split function for splitting datasets
from sklearn.model_selection import train_test_split

# Importing the KFold class for K-Fold cross-validation
from sklearn.model_selection import KFold

# Importing the DictVectorizer class for feature extraction
from sklearn.feature_extraction import DictVectorizer

# Importing the LogisticRegression class for logistic regression modeling
from sklearn.linear_model import LogisticRegression

# Importing the roc_auc_score function for evaluating model performance
from sklearn.metrics import roc_auc_score


# Reading the CSV file into a DataFrame and converting column names to lowercase with underscores
telecom_data = pd.read_csv("data.csv")
telecom_data.columns = telecom_data.columns.str.lower().str.replace(' ', '_')

# Identifying categorical columns
categorical_columns = list(telecom_data.dtypes[telecom_data.dtypes == 'object'].index)

# Cleaning and preprocessing categorical data
for column in categorical_columns:
    telecom_data[column] = telecom_data[column].str.lower().str.replace(' ', '_')

# Converting 'totalcharges' to numeric, handling errors, and filling NaN values with 0
telecom_data.totalcharges = pd.to_numeric(telecom_data.totalcharges, errors='coerce')
telecom_data.totalcharges = telecom_data.totalcharges.fillna(0)

# Converting 'churn' to binary (0 or 1)
telecom_data.churn = (telecom_data.churn == 'yes').astype(int)

# Splitting the dataset into training and testing sets
X_train_full, X_test = train_test_split(telecom_data, test_size=0.3, random_state=1)
y_test = X_test["churn"]

# Defining numerical and categorical features
numerical_features = ['tenure', 'monthlycharges', 'totalcharges']

categorical_features = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

# Function to train the logistic regression model
def train_model(X_train, y_train, C=1.0):
    dicts = X_train[categorical_features + numerical_features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train_transformed = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train_transformed, y_train)

    return dv, model

# Function to predict churn using the trained model
def predict_churn(df, dv, model):
    dicts = df[categorical_features + numerical_features].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred_prob = model.predict_proba(X)[:, 1]

    return y_pred_prob

# Model training and evaluation using K-Fold cross-validation
C_value = 1.0
num_splits = 5
kfold = KFold(n_splits=num_splits, shuffle=True, random_state=1)

auc_scores = []

for train_idx, val_idx in kfold.split(X_train_full):
    X_train = X_train_full.iloc[train_idx]
    X_val = X_train_full.iloc[val_idx]

    y_train = X_train["churn"]
    y_val = X_val["churn"]

    dv, model = train_model(X_train, y_train, C=C_value)
    y_pred_prob = predict_churn(X_val, dv, model)

    auc = roc_auc_score(y_val, y_pred_prob)
    auc_scores.append(auc)

# Displaying the mean and standard deviation of AUC scores for different folds
print('C=%s %.3f +- %.3f' % (C_value, np.mean(auc_scores), np.std(auc_scores)))

# Training the final model on the full training set and evaluating on the test set
dv, model = train_model(X_train_full, X_train_full["churn"], C=C_value)
y_pred_prob_test = predict_churn(X_test, dv, model)

# Calculating and displaying the AUC score for the test set
auc_test = roc_auc_score(y_test, y_pred_prob_test)
print("AUC on Test Set: %.3f" % auc_test)

