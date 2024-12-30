# SOC Prediction Model Analysis

## 1. Folder Contents
- `model-for-soc-prediction.ipynb`: Main Jupyter notebook containing the SOC prediction models
- `plots`: Folder containing plots for visualizing the outcomes of each model as well as contributions of each parameter to the prediction
- `README.md`: Documentation of the modeling work and results

## 2. Overview
This folder contains the work on predicting SOC (Soil Organic Carbon) values using regression models. The goal was to predict SOC values using various machine learning approaches, compare their performance, and analyze feature importance using SHAP values.

## 3. Models Tested
Four different regression models were implemented and evaluated:

### Model 1: Linear Regression
- Libraries: Scikit-learn
- Hyperparameters:
  - `fit_intercept`: [True, False]
  - `positive`: [True, False]

### Model 2: Random Forest
- Libraries: Scikit-learn
- Hyperparameters:
  - `n_estimators`: Range(50-300)
  - `max_depth`: Range(5-31)
  - `min_samples_split`: Range(2-11)
  - `min_samples_leaf`: Range(1-5)

### Model 3: XGBoost
- Libraries: XGBoost
- Hyperparameters:
  - `n_estimators`: Range(50-300)
  - `max_depth`: Range(3-10)
  - `learning_rate`: Uniform(0.01-0.3)
  - `subsample`: Uniform(0.6-1.0)
  - `colsample_bytree`: Uniform(0.6-1.0)

### Model 4: Support Vector Regression (SVR)
- Libraries: Scikit-learn
- Hyperparameters:
  - `C`: Uniform(0.1-10)
  - `epsilon`: Uniform(0.01-0.1)
  - `gamma`: ['scale', 'auto']

## 4. Data and Preprocessing
- Data Split: 80% training, 20% testing (random_state = 42)
- Preprocessing Steps:
  - Dropped unnecessary fields: Area, Boron, Zinc, longitude, latitude
  - Numerical features: StandardScaler normalization
  - Categorical features: OneHotEncoder encoding
  - Handle unknown categories: 'ignore' strategy
  - Pipeline implementation for reproducible preprocessing

## 5. Results

### Model Comparison Table

| Model             | RMSE     | MAE      | R² Score | CV RMSE  | CV RMSE Std |
|-------------------|----------|----------|----------|----------|-------------|
| Random Forest     | 0.254917 | 0.168785 | 0.825433 | 0.265363 | 0.015952    |
| XGBoost           | 0.257968 | 0.173745 | 0.821230 | 0.267518 | 0.013767    |
| SVR               | 0.262604 | 0.172951 | 0.814746 | 0.272334 | 0.014795    |
| Linear Regression | 0.272831 | 0.182821 | 0.800036 | 0.267765 | 0.015321    |

Key Findings:
- Random Forest achieved the best performance across all metrics
- Tree-based models (Random Forest and XGBoost) outperformed both SVR and Linear Regression 
- Cross-validation results show consistent performance across different data splits
- All models demonstrate strong R² scores, indicating good prediction capability
- Across all models, Nitrogen was the most important parameter for predicting SOC

Additional Analysis:
- SHAP (SHapley Additive exPlanations) values were calculated for each model to understand feature importance
- Feature importance analysis across all models to identify key predictors
- Visualization of SHAP summary plots for feature impact interpretation
