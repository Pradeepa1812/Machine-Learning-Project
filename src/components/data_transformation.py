#NumericalFeatures ,Categorical Features handling ,Handle Missing Values
import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging
@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_objects(self):
        '''
        This function is responsible for data transformation based on different types of data

        '''
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            num_pipeline = Pipeline(steps=[
                ('impute',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

            ])

            cat_pipeline = Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info(f'Numerical columns: {numerical_columns}')
            logging.info(f'categorical columns: {categorical_columns}')

            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initial_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading training and test data completed')

            logging.info('Obtaining preprocessing objects')

            preprocessing_obj = self.get_data_transformer_objects()
            
            print(preprocessing_obj)

            target_column_name ='math score'
            
            input_train_features = train_df.drop(columns = [target_column_name],axis=1)
            print(input_train_features.head())

            input_test_features = test_df.drop(columns = [target_column_name],axis = 1)
            logging.info(f'test_df:::: {test_df.head()}')

            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and test dataframe")

            input_train_df_arr = preprocessing_obj.fit_transform(input_train_features)
            input_test_df_arr = preprocessing_obj.transform(input_test_features)

            logging.info('Transformation for training and test data completed')

            train_arr = np.c_[input_train_df_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_test_df_arr,np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.{train_arr[0]}")

            save_object(

                file_path= self.data_transformation_config.preprocesser_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                #self.data_transformation_config.preprocesser_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

