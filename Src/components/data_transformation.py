import sys
import os 
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from Src.exception import CustomException
from Src.logger import logging
from Src.utils import save_object

class Datatransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=Datatransformationconfig()
    
    def get_data_transformer_object(self):

        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline=Pipeline( steps=[('imputer',SimpleImputer(strategy='median')),
             ("Scalar",StandardScaler())])
            
            cat_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
            ("OneHotEncoder",OneHotEncoder()),("Scalar",StandardScaler(with_mean=False))])

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer([("num_feature",num_pipeline,numerical_columns),(
                "Cat_feature",cat_pipeline,categorical_columns)])
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()

            target_column='math_score'

            input_feature_train_data=train_df.drop(columns=[target_column])
            target_feature_train_data=train_df[target_column]

            input_feature_test_data=test_df.drop(columns=[target_column])
            target_feature_test_data=test_df[target_column]

            logging.info("Applying preprocessing object on training and testing data")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_data)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_data)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_data)]

            logging.info("Saved preprocessing object")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)

            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)


        except Exception as e:
            raise CustomException(e,sys)







