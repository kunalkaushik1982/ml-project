#Feature Engineering, Data Cleaning, Convert catagorical features into numerical features

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    '''
    @dataclass: Automatically generates special methods like __init__() for the class.
    preprocessor_obj_file_path: Stores the file path where the preprocessor object (like a trained scaler) will be saved.
    '''
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        '''__init__: Initializes the DataTransformation class by creating an instance of DataTransformationConfig.
        '''
        
    def get_data_transformer_object(self):
        '''
        This Function is responsible for data transformation        
        numerical_columns and categorical_columns: Lists of columns that will be treated as numerical and categorical, respectively.
        num_pipeline: A pipeline for processing numerical columns:
        SimpleImputer: Fills in missing values with the median of the column.
        StandardScaler: Scales the data so that it has a consistent scale (centering disabled due to sparse matrices).
        
        '''
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )   
            
            '''cat_pipeline: A pipeline for processing categorical columns:
                SimpleImputer: Fills in missing values with the most frequent category.
                OneHotEncoder: Converts categorical variables into a one-hot numeric array.
                StandardScaler: Scales the one-hot encoded values.
            '''         
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            '''
            logging.info: Logs information about the columns being processed.
            ColumnTransformer: Combines both pipelines (num_pipeline and cat_pipeline) so they can be applied to their respective columns.
            return preprocessor: Returns the complete preprocessor object that can be applied to your data.
            CustomException: If an error occurs, it's wrapped in a custom exception for better debugging.
            '''
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            preprocessor=ColumnTransformer(
                [("num_pipeline",num_pipeline,numerical_columns),("cat_pipelines",cat_pipeline,categorical_columns)]
            )
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        '''
        Loading Data:

        train_df = pd.read_csv(train_path) and test_df = pd.read_csv(test_path): Loads the training and test datasets from CSV files.
        logging.info: Logs the progress.
        Processing Features:

        preprocessing_obj = self.get_data_transformer_object(): Retrieves the preprocessor object created earlier.
        target_column_name: The name of the column you want to predict (math_score).
        input_feature_train_df and input_feature_test_df: Separates the input features from the target variable in both training and test data.
        Applying Transformations:

        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df): Fits the preprocessor on the training data and transforms it.
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df): Applies the same transformation to the test data.
        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]: Concatenates the processed features with the target variable.
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]: Does the same for the test data.
        Saving the Preprocessor:

        save_object: Saves the preprocessor object to a file so you can use it later on new data.
        Returning Results:

        return: Returns the processed training and test arrays along with the path to the saved preprocessor.
        '''
        
        try:
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training dataframe and test dataframe.")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )      
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
            