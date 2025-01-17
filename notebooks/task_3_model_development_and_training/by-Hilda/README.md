## Model Overview by Hilda Posada
This folder includes my work on predicting **Soil Organic Carbon (SOC)** values using a Random Forest model. The goal was to experiment with regression techniques to predict SOC based on soil properties and provide recommendations to improve soil quality.

---
## Folder Contents
- `random_forest_model.ipynb`: Jupyter Notebook implementing the Random Forest model for SOC prediction.
- `data.csv`: The original dataset used for model training and evaluation.
- `soc_model.pkl`: Trained Random Forest model saved using `joblib`.
- `scaler.pkl`: StandardScaler object used to scale the features.
- `README.md`: Documentation of the work, including model details, preprocessing steps, and results.

---

## Models Tested

### Model 1: Random Forest
- **Libraries Used**: Scikit-learn
- **Hyperparameters**:
  - `n_estimators`: 100
  - `random_state`: 42
- **Description**: 
  - Random Forest was used as the primary model for predicting SOC due to its robustness and ability to handle non-linear relationships in the data.

---

## Data and Preprocessing
- **Dataset**: The dataset contains various soil attributes, including Nitrogen, Phosphorus, Potassium, Sulfur, pH, and categorical features like Area, Land class, and Soil type.
- **Preprocessing Steps**:
  1. **Train/Test Split**: 80% training, 20% testing.
  2. **Handling Missing Values**: No missing values were present.
  3. **Categorical Encoding**: Used `LabelEncoder` to encode `Area`, `Land class`, and `Soil type`.
  4. **Feature Scaling**: Applied `StandardScaler` to standardize numerical features.
- **Features Used**:
  - Nitrogen, Phosphorus, Potassium, Sulfur, pH, Area, Land class, Soil type.

---

## Results

| **Model**            | **Data Used** | **Preprocessing**          | **MSE** | **MAE** | **R-Squared** |
|-----------------------|---------------|----------------------------|---------|---------|---------------|
| Random Forest         | Original      | Normalization (StandardScaler) | 0.09    | 0.19    | 0.80          |

---

## Instructions to Reproduce
1. **Install Required Libraries**:
   Ensure you have the necessary libraries installed:
   ```bash
   pip install scikit-learn pandas numpy joblib
   ```

2. **Place Files in Folder**:
   - Save the dataset as `data.csv` in the same directory as the notebook/script.

3. **Run the Code**:
   - Open and run `random_forest_model.ipynb` or execute the script to reproduce the results:
     ```bash
     python random_forest_model.py
     ```

4. **Evaluate Model**:
   - The script will output Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) values.
   - Example predictions and recommendations will also be generated.

---

## Recommendations Based on SOC Predictions
- **If SOC < 2.0**: Add organic matter (e.g., compost) to improve SOC levels.
- **If SOC is between 2.0 and 3.5**: Maintain current practices; consider crop rotation.
- **If SOC > 3.5**: SOC levels are sufficient; monitor periodically.

--- 
