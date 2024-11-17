import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("https://getaround-app-cb43022eb5de.herokuapp.com")


data = pd.read_csv('get_around_pricing_project.csv')
data = data.iloc[:, 1:]
X = data.drop(columns=['rental_price_per_day'])
y = data['rental_price_per_day']


numerical_features = ['mileage', 'engine_power']
categorical_features = [col for col in X.columns if col not in numerical_features]


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


param_grid = {
    'model__n_estimators': [50, 100, 150],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

def perform_grid_search_rf():
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="RandomForest Grid Search"):
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        mlflow.log_params(best_params)
        
        best_model = grid_search.best_estimator_

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "test_mse": mean_squared_error(y_test, y_test_pred),
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "test_mae": mean_absolute_error(y_test, y_test_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred)
        }
        mlflow.log_metrics(metrics)
        
        mlflow.sklearn.log_model(best_model, "RandomForest_Best_Model")

        print("RandomForestRegressor best parameters and model logged with MLflow.")
        print("Metrics logged:", metrics)


mlflow.set_experiment("GetAround Pricing Prediction - RandomForest Hyperparameter Tuning")

perform_grid_search_rf()
