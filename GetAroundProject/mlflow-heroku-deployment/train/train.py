import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import mlflow
import mlflow.sklearn


mlflow.set_tracking_uri("https://getaround-app-cb43022eb5de.herokuapp.com")

data = pd.read_csv('get_around_pricing_project.csv')
X = data.drop(columns=['rental_price_per_day'])
y = data['rental_price_per_day']


numerical_features = ['mileage', 'engine_power']
categorical_features = [col for col in X.columns if col not in numerical_features]


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    'XGBRegressor': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    'LGBMRegressor': lgb.LGBMRegressor(random_state=42),
    'CatBoostRegressor': CatBoostRegressor(verbose=0, random_state=42),
    'SVR': SVR(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'ElasticNet': ElasticNet()
}


cv = KFold(n_splits=5, shuffle=True, random_state=42)


mlflow.set_experiment("GetAround Pricing Prediction - Models Comparison")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        
        mlflow.log_param("model_name", model_name)
        
        train_mse, test_mse = [], []
        train_mae, test_mae = [], []
        train_r2, test_r2 = [], []
        
        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            
            
            pipeline.fit(X_train, y_train)
            
            
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            
            
            train_mse.append(mean_squared_error(y_train, y_pred_train))
            train_mae.append(mean_absolute_error(y_train, y_pred_train))
            train_r2.append(r2_score(y_train, y_pred_train))
            
            test_mse.append(mean_squared_error(y_test, y_pred_test))
            test_mae.append(mean_absolute_error(y_test, y_pred_test))
            test_r2.append(r2_score(y_test, y_pred_test))
        
        
        mlflow.log_metric("train_mse_mean", np.mean(train_mse))
        mlflow.log_metric("train_mse_std", np.std(train_mse))
        mlflow.log_metric("test_mse_mean", np.mean(test_mse))
        mlflow.log_metric("test_mse_std", np.std(test_mse))
        
        mlflow.log_metric("train_mae_mean", np.mean(train_mae))
        mlflow.log_metric("train_mae_std", np.std(train_mae))
        mlflow.log_metric("test_mae_mean", np.mean(test_mae))
        mlflow.log_metric("test_mae_std", np.std(test_mae))
        
        mlflow.log_metric("train_r2_mean", np.mean(train_r2))
        mlflow.log_metric("train_r2_std", np.std(train_r2))
        mlflow.log_metric("test_r2_mean", np.mean(test_r2))
        mlflow.log_metric("test_r2_std", np.std(test_r2))
        
        
        mlflow.sklearn.log_model(pipeline, model_name)
        
        print(f"{model_name} logged with MLflow.")
