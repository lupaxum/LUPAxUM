import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
file_path = 'Crop_recommendation.csv'
dataset = pd.read_csv(file_path)

# Prepare the features and target
def prepare_data(dataset):
    X = dataset[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = dataset['label']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded, label_encoder

X, y_encoded, label_encoder = prepare_data(dataset)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate model performance
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the trained model and label encoder
joblib.dump(rf_classifier, 'model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')