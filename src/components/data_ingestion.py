import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTraining
class DataIngestion:
    def __init__(self, data_path="notebook/data/Stud.csv", output_dir="artifacts"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.raw_data_path = os.path.join(self.output_dir, "data.csv")
        self.train_data_path = os.path.join(self.output_dir, "train.csv")
        self.test_data_path = os.path.join(self.output_dir, "test.csv")

        os.makedirs(self.output_dir, exist_ok=True)

    def ingest_data(self):
        try:   
            df = pd.read_csv(self.data_path)
            logging.info("Data read successfully from data path")

            
            df.to_csv(self.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to %s", self.raw_data_path)

            
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train and test data splitting completed")

            
            train_data.to_csv(self.train_data_path, index=False, header=True)
            logging.info("Train data saved to %s", self.train_data_path)

            test_data.to_csv(self.test_data_path, index=False, header=True)
            logging.info("Test data saved to %s", self.test_data_path)

            logging.info("Data ingestion complete")

            return self.train_data_path, self.test_data_path
        except Exception as e:
            logging.error("Error during data ingestion: %s", str(e))
            raise CustomException(e, sys.exc_info())
  

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.ingest_data()   
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.data_transformation(train_data, test_data)
    model_trainer = ModelTraining()
    model_trainer.train_model(train_arr, test_arr)