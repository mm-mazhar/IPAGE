# EDA Findings


## Problems in Data
### Numerical Feature Issues

The dataset contains several anomalies in numerical features that may affect analysis:

- Invalid Numeric Formats:
    - 0.044
    - 13..8
    - .0.01
    - 0.0.1
- String Entries: A very low occurrence of string values in numeric columns.

## Range of Soil Nutrients
![alt text](images/dist_soil_nutrients.png)


| Soil Nutrients | Range |
| --- | --- |
pH Levels  | 4.3 - 6.1
Soil Organic Carbon | 0.58 - 3.3
Nitrogen | 0.04 - 0.16
Potassium | 0.0 < 0.01 - 0.52
Phosphorus | 0.06 - 80.0
Sulfur | 1.0 - 33.7
Boron | 0.06 - 0.63
Zinc | 0.08 - 1.91

## Distribution Analysis of Categorical Features
The analysis of categorical features indicates an imbalanced distribution of data, which could impact model training and generalization. 
![alt text](images/Untitled-1.png)

## Soil Nutrient Levels by Soil Type and Land Class
There are observable variations in soil nutrient levels depending on land_class and soil_type. However, due to the imbalanced data in categorical features, it may be premature to draw strong conclusions about causality.
### Feature Columns
![alt text](images/Untitled-1-4.png)
![alt text](images/Untitled-1-5.png)
![alt text](images/Untitled-1-6.png)
![alt text](images/Untitled-1-7.png)
![alt text](images/Untitled-1-8.png)

### Target Columns
![alt text](images/Untitled-1-2.png)
![alt text](images/Untitled-1-3.png)
![alt text](images/Untitled-1-1.png)

## Correlation between Soil Nutrients and Target
The correlation analysis highlights relationships between soil nutrients and target variables, supported by findings from relevant literature:

**"Prediction of Soil Reaction (PH) and Soil Nutrients Using Multivariate Statistics Techniques For Agricultural Crop and Soil Management"**:
- K and SOC show a positive correlation with PH.
- N displays a negative correlation with PH.
- Cu negatively correlates with PH, while Mn, Zn, Fe, and S correlate positively with PH.

**"Meta Analysis on The Effects of Types and Levels of N, P, K Fertilization"**:
- SOC levels are highest with balanced NPK and decline at very high NPK levels.

**"Effect of Boron, Zinc, and Sulphur on Soil Fertility and Yield of Fababean"**:
- Organic carbon levels improved slightly with the application of B, Zn, and S.
- Nitrogen levels increased with Zn and S additions.
- Available phosphorus improved with B but decreased with higher Zn and S levels.
- Potassium content decreased with B and Zn but increased with S application.

Some findings from these studies are not yet observable in the current IPAGE dataset, likely due to the limited data available at this time.

### Recommendations for Feature Engineering
Given the correlation observed between certain soil nutrients and target variables, consider combining features during the feature engineering phase. This may enhance the predictive power of models by leveraging the relationships identified in the analysis.

![alt text](images/Untitled-1-9.png)
