import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.config_entity import config
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation,DataTransformationConfig
class DataIngestion:
    def __init__(self):
        self.ingestion_config = config.DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
        This function is responsible for data ingestion
        
        '''
        logging.info('Entered the data ingestion components')

        try:
            df = pd.read_csv('Notebook/data/StudentsPerformance.csv')
            print(df.head())
            logging.info('Read the dataset as Dataframes')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            #training and test data
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data .to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initial_data_transformation(train_data,test_data)

        