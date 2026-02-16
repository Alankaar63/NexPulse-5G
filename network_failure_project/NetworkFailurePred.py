import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset
data_cleaned = pd.read_csv('signal_metrics.csv')

# Create 'Failure' column
data_cleaned['Failure'] = (
    (data_cleaned['Signal Strength (dBm)'] < -93) |
    (
        (data_cleaned['BB60C Measurement (dBm)'] < -93) &
        (data_cleaned['srsRAN Measurement (dBm)'] < -93) &
        (data_cleaned['BladeRFxA9 Measurement (dBm)'] < -93) &
        (data_cleaned['Latency (ms)'] > 90)
    )
).astype(int)

# Convert date columns to timestamps
for col in data_cleaned.select_dtypes(include=['object']).columns:
    try:
        data_cleaned[col] = pd.to_datetime(data_cleaned[col])
        data_cleaned[col] = data_cleaned[col].astype(int) // 10**9  # Convert to seconds
    except Exception as e:
        logging.warning(f"Skipping column {col}: {e}")

# Handle missing values in 'Network Type'
data_cleaned['Network Type'] = data_cleaned['Network Type'].fillna('Unknown')

# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_nt = ohe.fit_transform(data_cleaned[['Network Type']])
encoded_nt_df = pd.DataFrame(encoded_nt, columns=ohe.get_feature_names_out(['Network Type']))
data_cleaned = pd.concat([data_cleaned, encoded_nt_df], axis=1)
data_cleaned.drop(columns=['Network Type'], inplace=True)

# Drop unnecessary columns
drop_cols = ['Sr.No.', 'Locality']
data_cleaned = data_cleaned.drop(columns=[col for col in drop_cols if col in data_cleaned.columns])

# Define only the features we need for training
selected_features = ["Signal Strength (dBm)", "BB60C Measurement (dBm)", "srsRAN Measurement (dBm)", 
                     "BladeRFxA9 Measurement (dBm)", "Latency (ms)"] + list(encoded_nt_df.columns)

# Use only the selected features
X = data_cleaned[selected_features]
y = data_cleaned['Failure']



# Split dataset
X = data_cleaned.drop(columns=['Failure'])
y = data_cleaned['Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
logging.info("Model Performance:")
logging.info(classification_report(y_test, y_pred))
logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Hyperparameter tuning
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
logging.info(f"Best Parameters: {best_params}")

# Save model
import joblib
import logging

# Save the model
joblib.dump(model, "model.pkl")

# Save the encoder
joblib.dump(ohe, "ohe.pkl")

selected_features = ["Signal Strength (dBm)", "BB60C Measurement (dBm)", "srsRAN Measurement (dBm)", 
                     "BladeRFxA9 Measurement (dBm)", "Latency (ms)"] + list(ohe.get_feature_names_out(['Network Type']))

joblib.dump(selected_features, "features.pkl")

