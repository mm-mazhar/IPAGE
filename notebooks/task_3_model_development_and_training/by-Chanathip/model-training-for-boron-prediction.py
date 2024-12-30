# %%
import argparse

import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import Ridge
from xgboost import  XGBRegressor

import mlflow

import dagshub
dagshub.init(repo_owner='Omdena', repo_name='IPage', mlflow=True)

#%%

parser = argparse.ArgumentParser()

## model name is one of the following: ridge, xgb, rf, bagging
parser.add_argument('--model_name', type = str, required=True)

args = parser.parse_args()

model_name = args.model_name

# %%

df = pd.read_csv('../../../data/merged_v4.csv')


# %%
df = df.drop(['SOC', 'Zinc', 'longitude', 'latitude'], axis=1)

## drop outlier of Boron
# df = df[df['Boron'] <= 5]

print('total data:', len(df))

# %%
num_cols = ['pH', 'Nitrogen', 'Potassium', 'Phosphorus', 'Sulfur', 'Sand', 'Silt', 'Clay']
cat_cols = ['Area', 'Soil group', 'Land class', 'Soil type']

# %%
labels = df['Boron']
features = df.drop('Boron', axis=1)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)


# %%

models = {
    'ridge': Ridge(alpha = 10, max_iter=100, random_state=0), 
    'bagging': BaggingRegressor(n_estimators=500, random_state=0), 
    'rf': RandomForestRegressor(n_estimators=500, min_samples_leaf=7, min_samples_split = 2, random_state=0), 
    'xgb': XGBRegressor(n_estimators = 100, learning_rate = 0.1, random_state=0)
}

# %%

model = models[model_name]

print('fitting model', model_name)

mlflow.set_experiment('Chanathip_merged-v4_{}_Boron_Best-model-from-grid-search_20241230'.format(model_name))

with mlflow.start_run():

    mlflow.autolog()

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

    col_transformer.fit(x_train)

    train_data = col_transformer.transform(x_train).toarray()
    test_data = col_transformer.transform(x_test).toarray()
        
    model.fit(train_data, y_train)

    pred = model.predict(test_data)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)

    mlflow.log_metric('R2-test', r2)
    mlflow.log_metric('MAE-test', mae)
    mlflow.log_metric('MSE-test', mse)

    print('R2:', round(r2,2))
    print('MAE:', round(mae,2))
    print('MSE:', round(mse,2))

print('-'*30)


# %%



