import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score



def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved to {file_path}")

    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")
        raise CustomException(e, sys.exc_info())

def evaluate_model(X_train, X_test, y_train, y_test, models:dict):
    try:
        model_scores ={}
        best_model = {}
        for name, (model, param) in models.items():
            gs = GridSearchCV(model,param, cv=3)
            gs.fit(X_train, y_train)
            best_model_param = gs.best_estimator_
            best_model_param.fit(X_train, y_train)

            y_pred = best_model_param.predict(X_test)
            R2 = r2_score(y_test, y_pred)

            model_scores[name] = R2
            best_model[name] = best_model_param

            best_model_name = max(model_scores, key=model_scores.get)
            best_models = best_model[best_model_name]
            best_R2 = model_scores[best_model_name]

            logging.info(f"Best Model: {best_model_name} with R² = {best_R2:.4f}")

        return best_model, model_scores
    
    except Exception as e:
        raise CustomException(e, sys.exc_info())


def load_object(file_path):
    try:
        with open(file_path, 'rb') as obj:
            pickle.load(obj)

    except Exception as e:
        raise CustomException(e, sys.exc_info())