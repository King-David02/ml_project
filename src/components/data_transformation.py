import sys
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

logging.info("Data Transformation Started")
class DataTransformation:
    logging.info("Inside DataTransformation Constructor")
    def __init__(self, out_dir='artifacts'):
        logging.info("after DataTransformation Constructor")
        logging.info("Inside DataTransformation Constructor")
        self.out_dir = out_dir
        self.preprocessor_path = os.path.join(self.out_dir, "preprocessor.pkl")
        logging.info(f"Creating directory: {self.out_dir}")
        os.makedirs(self.out_dir, exist_ok=True)
        logging.info("resume")
        logging.info(f"Directory {self.out_dir} created or already exists.")


    def data_transformation_initiation(self):
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoding", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical variables are: {categorical_columns}")
            logging.info(f"Numerical variables are: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ("numerical transformation", num_pipeline, numerical_columns),
                ("categorical transformation", cat_pipeline, categorical_columns)
            ])

            return preprocessor
    
        except Exception as e:
            raise CustomException(e, sys.exc_info())

    def data_transformation(self, train_path: str, test_path: str) -> tuple:
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data successfully")

            target_column_name = "math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            preprocessing = self.data_transformation_initiation()

            input_feature_train_arr = preprocessing.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Preprocessing complete")

            save_object(file_path=self.preprocessor_path, obj=preprocessing)
            logging.info(f"File saved to {self.preprocessor_path}")
            
            return (
                train_arr,
                test_arr,
                self.preprocessor_path
            )

        except Exception as e:
            raise CustomException(e, sys.exc_info())
