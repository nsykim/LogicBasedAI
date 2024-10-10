import unittest
import pandas as pd
from io import StringIO
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from dataProcessing import DataProcessing
class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # Sample data similar to the "Adult" dataset
        self.csv_data = StringIO("""
        age,workclass,education,marital-status,occupation,relationship,race,sex,hours-per-week,income
        39,State-gov,Bachelors,Never-married,Adm-clerical,Not-in-family,White,Male,40,<=50K
        50,Self-emp-not-inc,Bachelors,Married-civ-spouse,Exec-managerial,Husband,White,Male,13,<=50K
        38,Private,HS-grad,Divorced,Handlers-cleaners,Not-in-family,White,Male,40,<=50K
        53,Private,11th,Married-civ-spouse,Handlers-cleaners,Husband,Black,Male,40,<=50K
        28,Private,Bachelors,Married-civ-spouse,Prof-specialty,Wife,Black,Female,40,>50K
        """)
        self.data_processing = DataProcessing(self.csv_data)

    def test_load_data(self):
        self.data_processing.load_data()
        self.assertIsInstance(self.data_processing.data, pd.DataFrame)
        self.assertEqual(len(self.data_processing.data), 5)

    def test_detect_categorical_columns(self):
        self.data_processing.load_data()
        categorical_columns = self.data_processing.detect_categorical_columns()
        expected_categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'income']
        self.assertListEqual(categorical_columns, expected_categorical_columns)

    def test_preprocess(self):
        self.data_processing.load_data()
        preprocessed_data = self.data_processing.preprocess()
        self.assertIsInstance(preprocessed_data, pd.DataFrame)
        self.assertFalse(preprocessed_data.isnull().values.any())
        self.assertTrue(all(preprocessed_data.dtypes != 'object'))

    def test_split_data(self):
        self.data_processing.load_data()
        self.data_processing.preprocess()
        X_train, X_test, y_train, y_test = self.data_processing.split_data(target_col='income')
        self.assertEqual(len(X_train) + len(X_test), len(self.data_processing.data))
        self.assertEqual(len(y_train) + len(y_test), len(self.data_processing.data))

    def test_balance_data(self):
        self.data_processing.load_data()
        self.data_processing.preprocess()
        balanced_data = self.data_processing.balance_data(target_col='income')
        self.assertIsInstance(balanced_data, pd.DataFrame)
        self.assertTrue('income' in balanced_data.columns)

if __name__ == '__main__':
    unittest.main()