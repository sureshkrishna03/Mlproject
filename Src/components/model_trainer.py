import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from Src.exception import CustomException
from Src.logger import logging

from Src.utils import save_object, Evaluate_model

@dataclass

class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts","model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()



    def initiate_model_trainer(self,train_array,test_array):

            try:
                logging.info("Splitting training and test input data")
                X_train,y_train,X_test,y_test=(
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
                )

                models={
                    "Random_forest": RandomForestRegressor(),
                    "Decision_tree":DecisionTreeRegressor(),
                    "Gradient_boosting":GradientBoostingRegressor(),
                    "Linear_regression":LinearRegression(),
                    "KNN":KNeighborsRegressor(),
                    "XGBRegressor":XGBRegressor(),
                    "CatBoosting":CatBoostRegressor(verbose=False),
                    "AdaBoost":AdaBoostRegressor()
                }

                params={
                    "Decision_tree": {
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                    },
                    "Random_forest":{
                        'n_estimators':[8,16,32,64,128,256]
                    },
                    "Gradient_boosting":{
                        'learning_rate':[.1,.01,.05,.001],
                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                        'n_estimators':[8,16,32,64,128,256]
                    },
                    "Linear_regression":{},
                    "KNN":{
                        'n_neighbors':[5,7,9,11],
                        'weights':['uniform','distance']
                    },
                    "XGBRegressor":{
                        'learning_rate':[.1,.01,.05,.001],
                        'n_estimators':[8,16,32,64,128,256]
                    },
                    "CatBoosting":{
                        'depth': [6,8,10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                    },
                    "AdaBoost":{
                        'learning_rate':[.1,.01,.05,.001],
                        'n_estimators':[8,16,32,64,128,256]
                    }
                }

                model_report: dict= Evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                   models=models, params=params)

                
                ## to get best model score from dicts

                best_model_score=max(sorted(model_report.values()))

                ## to get best model name form dict

                best_model_name=max(model_report,key=model_report.get)
                
                best_model=models[best_model_name]

                if best_model_score<0.6:
                    raise CustomException("No best model found")
                
                logging.info("Best found model on both training and testing dataset is {}".format(best_model))

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model)

                predicted=best_model.predict(X_test)

                r2=r2_score(y_test,predicted)

                return r2
        
            except Exception as e:
                raise CustomException(e,sys)    
 