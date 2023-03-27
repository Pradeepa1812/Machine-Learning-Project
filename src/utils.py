import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        print('OBJ ::::::',obj)
        
        dir_path = os.path.dirname(file_path)
        logging.info(f'Paths in utils {dir_path,file_path}')

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)


    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        logging.info('Evaluate model started')
        report = {}

        for i in range(len(models)):

            model = list(models.values())[i]
            print(list(models.keys())[i])
            params = param[list(models.keys())[i]]

            grid_result = GridSearchCV(model,params,cv=3)
            grid_result.fit(X_train,y_train)
            
            model.set_params(**grid_result.best_params_)
            model.fit(X_train,y_train)

            #fit and predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #r square 
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        print(report)
        return report

    except Exception as e:
        raise CustomException(e, sys)