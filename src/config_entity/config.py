import os
from dataclasses import dataclass # defining variables only

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str =  os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')