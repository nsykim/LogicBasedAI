import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import logging

class DataProcessing:
    """
    DataProcessing class is used to process and clean data retrieved from a csv file.
    """
    def __init__(self, file_path):
        """
        Constructor for DataProcessing class
        Initializes the file_path, data, label_encoders, and scaler attributes.
        
        Input: file_path (str) - path to the csv file
        """
        self.file_path = file_path
        self.data = None
        self.label_encoders = {} # dictionary to store label encoders for each column 
        self.scaler = StandardScaler() #normalizes the data to have a mean of 0 and a standard deviation of 1

    def load_data(self):
        """
        Loads the data from the csv file into a pandas dataframe.

        Output: data (pd.DataFrame) - dataframe containing the data from the csv file
        """
        try:
            self.data = pd.read_csv(self.file_path)
        except FileNotFoundError:
            logging.error("File not found. Please check the file path.")
            raise
        except pd.errors.EmptyDataError:
            logging.error("File is empty. Please check the file.")
            raise
        except pd.errors.ParserError:
            logging.error("Error parsing the csv file. Please check the format.")
            raise
        return self.data


    def detect_categorical_columns(self):
        """
        Detects the categorical columns in the dataframe.

        Output: categorical_cols (list) - list of column names that are categorical
        """
        categorical_cols = []
        for column in self.data.columns:
            if pd.api.types.is_object_dtype(self.data[column]) or self.data[column].nunique() < 21:
                categorical_cols.append(column)
        return categorical_cols

    def preprocess(self):
        """
        Preprocesses the data by cleaning, encoding, and scaling.
        Automatically detects non-catergorical (non-numeric) columns 
        """
        categorical_cols = self.detect_categorical_columns() # detect the categorical columns

        # clean the data (fill missing values)
        for column in self.data.columns: # for each column in the dataframe
            if column in categorical_cols: # if the column is categorical
                self.data[column].fillna('Unknown', inplace=True) # fill missing values with 'Unknown'
            else:
                # if numerical and empty, fill with column mean
                self.data[column].fillna(self.data[column].mean(),  inplace=True)

        for column in categorical_cols:
            le = LabelEncoder() # create a label encoder
            self.data[column] = le.fit_transform(self.data[column]) # fit and transform the column
            self.label_encoders[column] = le # store the label encoder in the dictionary


        # normalize the numeric columns
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])

        return self.data

    def split_data(self, target_col, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.

        Input:
        - target_col (str) - name of the target column
        - test_size (float) - proportion of the data to include in the test split from 0-1
        - random_state (int) - random seed

        Output:
        - X_train (pd.DataFrame) - training data
        - X_test (pd.DataFrame) - testing data
        - y_train (pd.Series) - training labels
        - y_test (pd.Series) - testing labels
        """
        if target_col not in self.data.columns:
            raise ValueError("Target column not found. Please check the column name")
        X = self.data.drop(target_col, axis=1)
        y = self.data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

    def balance_data(self, target_col):
        """
        Balances the data by oversampling the minority class.

        Input:
        - target_col (str) - name of the target column
        """
        if target_col not in self.data.columns:
            raise ValueError("Target column not found. Please check the column name")
        
        majority_class = self.data[self.data[target_col] == self.data[target_col].mode()[0]]
        minority_class = self.data[self.data[target_col] != self.data[target_col].mode()[0]]

        minority_class_oversampled = resample(minority_class, 
                                            replace=True, 
                                            n_samples=len(majority_class),
                                            random_state=42)
        
        self.data = pd.concat([majority_class, minority_class_oversampled])
        return self.data