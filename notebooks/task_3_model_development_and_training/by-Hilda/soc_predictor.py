import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Feature Selection 
features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Sulfur', 'pH', 'Area', 'Land class', 'Soil type']
target_soc = 'SOC'

# Encode categorical variables: 'Area', 'Land class', 'Soil type'
label_encoders = {}
for col in ['Area', 'Land class', 'Soil type']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features and target
X = data[features]
y = data[target_soc]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, 'soc_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)  # MAE Calculation
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")

# Example Prediction
# Replace these values with real inputs: Nitrogen, Phosphorus, Potassium, Sulfur, pH, Area, Land class, Soil type
example_input = pd.DataFrame([[1.0, 10.5, 15.2, 5.3, 6.5, 0, 1, 2]],
                             columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Sulfur', 'pH', 
                                      'Area', 'Land class', 'Soil type'])

# Scale the input using the trained scaler
example_input_scaled = scaler.transform(example_input)

# Predict SOC using the trained model
predicted_soc = model.predict(example_input_scaled)
print(f"Predicted SOC: {predicted_soc[0]:.2f}")

# Generate Recommendations
def generate_recommendations(soc):
    recommendations = []
    if soc < 2.0:
        recommendations.append("Add organic matter (e.g., compost) to improve SOC levels.")
    elif soc < 3.5:
        recommendations.append("Maintain current practices; consider crop rotation.")
    else:
        recommendations.append("SOC levels are sufficient; monitor periodically.")
    return recommendations

# Generate recommendations for the prediction
recommendations = generate_recommendations(predicted_soc[0])
print("Recommendations:")
for rec in recommendations:
    print(f"- {rec}")
