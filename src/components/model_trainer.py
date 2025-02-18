import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

class ModelTraining:
    def __init__(self, output_dirs ="artifacts"):
        self.output_dirs = output_dirs
        self.model_file_path = os.path.join(output_dirs, "model.pkl")
        os.makedirs(self.output_dirs, exist_ok=True)
        

    def train_model(self, train_arr, test_arr):
        try:
            X_train, X_test, y_train, y_test =(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )

            logging.info("Splitting data complete")

            models_params = {
            "Random Forest": (
             RandomForestRegressor(),
             {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_features': ['sqrt', 'log2', None],
            'n_estimators': [8, 16, 32, 64, 128, 256]
             }
             ),
            "Decision Tree": (
            DecisionTreeRegressor(),
            {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'splitter': ['best', 'random'],
            'max_features': ['sqrt', 'log2'],
            }
            ),
            "Gradient Boosting": (
            GradientBoostingRegressor(),
            {
            'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
            'learning_rate': [0.1, 0.01, 0.05, 0.001],
            'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            'criterion': ['squared_error', 'friedman_mse'],
            'max_features': ['auto', 'sqrt', 'log2'],
            'n_estimators': [8, 16, 32, 64, 128, 256]
            }
            ),
            "Linear Regression": (LinearRegression(), {}),
            "XGBRegressor": (
            XGBRegressor(),
            {
            'learning_rate': [0.1, 0.01, 0.05, 0.001],
            'n_estimators': [8, 16, 32, 64, 128, 256]
            }
            ),
            "CatBoosting Regressor": (
            CatBoostRegressor(verbose=False),
            {
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [30, 50, 100]
            }
            ),
            "AdaBoost Regressor": (
            AdaBoostRegressor(),
            {
            'learning_rate': [0.1, 0.01, 0.5, 0.001],
            'loss': ['linear', 'square', 'exponential'],
            'n_estimators': [8, 16, 32, 64, 128, 256]
            }
            ),
            }


            logging.info("Starting model evaluation")
            best_model, r2_score = evaluate_model(X_train=X_train,X_test=X_test, y_train=y_train, y_test=y_test, models= models_params)
            logging.info("Model Training Complete")

            predicted =best_model.predict(X_test)
            R2 = r2_score(y_test, predicted)
            logging.info("R2 score for {best_model} is {R2}")

            save_object(file_path=self.model_file_path, obj=best_model)
            logging.info("Saved model as a pickle file")



            return R2

        except Exception as e:
            raise CustomException(e, sys.exc_info())
        


