import pandas as pd
from sklearn.model_selection import train_test_split
from sk.preprocessing import StandardScaler, LabelEncoder

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
        self.data = pd.read_csv(self.file_path)
        return self.data

    def preprocess(self):
        """
        Preprocesses the data by cleaning, encoding, and scaling.
        """
        # clean the data (fill missing values)
        self.data.fillna(self.data.mean(),  inplace=True) # fill missing values with the mean of the column

        # encode the categorical columns
        for column in self.data.columns: # for each column in the dataframe
            if self.data[column].dtype == type(object): # if the column is of type object (not useful to model)
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
        - test_size (float) - proportion of the data to include in the test split
        - random_state (int) - random seed

        Output:
        - X_train (pd.DataFrame) - training data
        - X_test (pd.DataFrame) - testing data
        - y_train (pd.Series) - training labels
        - y_test (pd.Series) - testing labels
        """
        X = self.data.drop(target_col, axis=1)
        y = self.data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test