# Model development by Chanathip

This folder contains the file used for training models to predict Boron for iPage.

---

## 1. **Folder Contents**  

- `model-training-for-boron-prediction.ipynb`: Code for training models to predict Boron.
- `README.md`: Documentation for my work.  

---

## 2. **Overview**  

This folder includes my work on the Boron prediction. I experimented with different models like Random Forest and XGBoost to predict Boron values using the provided dataset. I conducted experiment by using 10-fold cross validation.

---

## 3. **Models Tested**  

### **Model 1: Ridge Regression**  
- Libraries: Scikit-learn  
- Hyperparameters:  
  - `alpha`: [1, 5, 10]  
  - `max_iter`: [100, 500, 1000]  

### **Model 2: Lasso Regression**  
- Libraries: Scikit-learn  
- Hyperparameters:  
  - `alpha`: [1, 5, 10]  
  - `max_iter`: [100, 500, 1000]  

### **Model 3: Elastic Net Regression**  
- Libraries: Scikit-learn  
- Hyperparameters:  
  - `alpha`: [1, 5, 10]  
  - `max_iter`: [100, 500, 1000]  
  - `l1_ratio`: [0.3, 0.5, 0.7, 1.0]  

### **Model 4: AdaBoost Regressor**  
- Libraries: Scikit-learn  
- Hyperparameters:  
  - `n_estimators`: [100, 300, 500]  
  - `learning_rate`: [0.1, 0.5, 1.0, 5.0]  
  - `loss`: ['linear', 'square']  

### **Model 5: Bagging Regressor**  
- Libraries: Scikit-learn  
- Hyperparameters:  
  - `n_estimators`: [100, 300, 500]  

### **Model 6: Random Forest Regressor**  
- Libraries: Scikit-learn  
- Hyperparameters:  
  - `n_estimators`: [100, 300, 500]  
  - `min_samples_split`: [2, 3, 5, 7]  
  - `min_samples_leaf`: [1, 3, 5, 7]  

### **Model 7: XGBoost Regressor**  
- Libraries: Scikit-learn, XGBoost  
- Hyperparameters:  
  - `n_estimators`: [100, 300, 500]  
  - `learning_rate`: [0.1, 0.5, 1.0, 5.0]  

---

## 4. **Data and Preprocessing**  

- Data Split: 80% training, 20% testing   (randomly split)
- Preprocessing:
  - use OneHotEncoder to encode categorical variables
  - use StandardScaler to scale all numerical variables into same range

---

## 5. **Results**  

### **Model Comparison Table:**  

| Model                 | Data Used    | Preprocessing        | MSE   | MAE  | R-Squared |  
|-----------------------|--------------|----------------------|-------|------|-----------|  
| XGBoost Regressor     | merged_v2.csv | as explained in Section 4.                 | 0     | 0.04 | 0.7       |  
| RandomForest Regressor| merged_v2.csv | as explained in Section 4.                 | 0     | 0.04 | 0.72      |  
| Bagging               | merged_v2.csv | as explained in Section 4.                 | 0     | 0.04 | 0.7       |  
| Ridge                 | merged_v2.csv | as explained in Section 4.                 | 0     | 0.04 | 0.7       |  













