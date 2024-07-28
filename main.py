import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset (you'll need to provide the actual data file)
data = pd.read_csv('parkinsons_data.csv')

# Separate features and target
X = data.drop(['status', 'name'], axis=1)
y = data['status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model
svm_model = svm.SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Function to predict for new data
def predict_parkinsons(new_data):
    new_data_scaled = scaler.transform(new_data)
    prediction = svm_model.predict(new_data_scaled)
    return "Parkinson's detected" if prediction[0] == 1 else "No Parkinson's detected"
