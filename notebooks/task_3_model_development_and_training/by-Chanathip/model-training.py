# %%
import os

# os.environ["OPENBLAS_NUM_THREADS"] = "4"

# import seaborn as sns
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from autosklearn.regression import AutoSklearnRegressor
# from collections import  Counter
# import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %%
df = pd.read_csv('../../../../soil-prediction/iPAGE SoilData.csv')

# df = df.drop('Data Collection Year', axis=1)
df = df.fillna('UNK')

## filter out outliers (get this from visualization)

df = df[df['SOC (%)'] < 10]
df = df[df['Nitrogen N (%)'] < 0.3]
df = df[df['Potassium K (meq/100)'] < 10]

target_cols = ['SOC (%)', 'Boron B (ug/g)', 'Zinc Zn (ug/g)']

# %%
df.columns

# %%
num_cols = ['pH', 'Nitrogen N (%)', 'Potassium K (meq/100)', 'Phosphorus P (ug/g)', 'Sulfur S (ug/g)']
cat_cols = ['Area', 'soil group', 'Land class', 'knit (surface)']

# %%
# df.head()
train_df = df[df['Data Collection Year']<2016]
test_df = df[df['Data Collection Year']>=2016]

train_df = train_df.drop('Data Collection Year', axis=1)
test_df = test_df.drop('Data Collection Year', axis=1)

print('total train samples:', len(train_df))
print('total test samples:', len(test_df))

train_labels = train_df[target_cols]
train_features = train_df.drop(target_cols, axis=1)

test_labels = test_df[target_cols]
test_features = test_df.drop(target_cols, axis=1)

# %%
num_transformer = Pipeline(
    steps=[
        ('scaler', StandardScaler())
    ]
)

cat_transformer = Pipeline(
    steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
)

col_transformer = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        remainder= 'passthrough'
    )

# %%
col_transformer.fit(train_df)

# col_names = list(col_transformer.get_feature_names_out())

# col_names = [s.replace('cat__','').replace('num__','').replace('remainder__','') for s in col_names]

train_data = col_transformer.transform(train_df).toarray()
test_data = col_transformer.transform(test_df).toarray()

# train_df = pd.DataFrame(data=train_data, columns=col_names)
# test_df = pd.DataFrame(data=test_data, columns=col_names)

# %%
train_labels.columns

# %%
automl = AutoSklearnRegressor(
    time_left_for_this_task=60,
    per_run_time_limit=30,
    tmp_folder="/tmp/autosklearn_regression_example_tmp",
    memory_limit = 512,
)

automl.fit(train_data, train_labels['SOC (%)'])

# %%


# %%



