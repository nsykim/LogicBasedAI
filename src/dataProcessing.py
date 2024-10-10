import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import logging
from io import StringIO

class DataProcessing:
    """
    DataProcessing is a class for handling and preprocessing data from a CSV file. It provides methods to load data, detect categorical columns, preprocess the data, split the data into training and testing sets, and balance the dataset.
    Methods:
        __init__(file_path):
        load_data():
        detect_categorical_columns():
        preprocess():
            Preprocesses the data by filling missing values, encoding categorical columns, and normalizing numerical columns.
        split_data(target_col, test_size=0.2, random_state=42):
        balance_data(target_col):
    """

    def __init__(self, file_path):
        """
        Initializes the DataProcessing class with the given file path.

        Args:
            file_path (str): The path to the data file.

        Attributes:
            file_path (str): The path to the data file.
            data (None): Placeholder for the data to be loaded.
            label_encoders (dict): Dictionary to store label encoders for each column.
            scaler (StandardScaler): Scaler to normalize the data to have a mean of 0 and a standard deviation of 1.
        """
        self.file_path = file_path
        self.data = None
        self.label_encoders = {} # dictionary to store label encoders for each column 
        self.scaler = StandardScaler() #normalizes the data to have a mean of 0 and a standard deviation of 1

    def load_data(self):
        """
        Loads data from a CSV file specified by the file path.

        This method attempts to read a CSV file into a DataFrame. If the file is not found,
        empty, or cannot be parsed, appropriate error messages are logged and exceptions are raised.

        Returns:
            pd.DataFrame: The data loaded from the CSV file.

        Raises:
            FileNotFoundError: If the file specified by the file path does not exist.
            pd.errors.EmptyDataError: If the file is empty.
            pd.errors.ParserError: If there is an error parsing the CSV file.
        """
        try:
            if isinstance(self.file_path, StringIO):
                self.file_path.seek(0)
                self.data = pd.read_csv(self.file_path)
            else:
                self.data = pd.read_csv(self.file_path)
            self.data.columns = self.data.columns.str.strip() # remove leading/trailing whitespaces from column names
        except FileNotFoundError as e:
            logging.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")
        return self.data


    def detect_categorical_columns(self):
        """
        Detects and returns a list of categorical columns in the dataset.
        This method iterates through all columns in the dataset and identifies
        columns that are either of object data type or have fewer than 21 unique values.
        Returns:
            list: A list of column names that are considered categorical.
        """

        categorical_cols = []
        for column in self.data.columns:
            if pd.api.types.is_object_dtype(self.data[column]) or self.data[column].nunique() < 21 and not pd.api.types.is_numeric_dtype(self.data[column]):
                categorical_cols.append(column)
        return categorical_cols

    def preprocess(self):
        """
        Preprocesses the data by performing the following steps:
        1. Detects categorical columns.
        2. Fills missing values:
           - For categorical columns, fills with 'Unknown'.
           - For numerical columns, fills with the column mean.
        3. Encodes categorical columns using Label Encoding.
        4. Normalizes numerical columns using the provided scaler.
        Returns:
            pd.DataFrame: The preprocessed data.
        """

        categorical_cols = self.detect_categorical_columns() # detect the categorical columns

        # clean the data (fill missing values)
        for column in self.data.columns: # for each column in the dataframe
            if column in categorical_cols: # if the column is categorical
                self.data[column] = self.data[column].fillna('Unknown') # fill missing values with 'Unknown'
            else:
                # if numerical and empty, fill with column mean
                self.data[column] = self.data[column].fillna(self.data[column].mean())

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
        Splits the dataset into training and testing sets.

        Parameters:
        target_col (str): The name of the target column in the dataset.
        test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split. Default is 42.

        Returns:
        tuple: A tuple containing four elements:
            - X_train (DataFrame): The training input samples.
            - X_test (DataFrame): The testing input samples.
            - y_train (Series): The training target values.
            - y_test (Series): The testing target values.

        Raises:
        ValueError: If the target column is not found in the dataset.
        """
        if target_col not in self.data.columns:
            raise ValueError("Target column not found. Please check the column name")
        X = self.data.drop(target_col, axis=1)
        y = self.data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

    def balance_data(self, target_col):
        """
        Balances the dataset by oversampling the minority class to match the size of the majority class.
        Parameters:
        target_col (str): The name of the target column to balance.
        Returns:
        pd.DataFrame: A new DataFrame with balanced classes.
        Raises:
        ValueError: If the target column is not found in the DataFrame.
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