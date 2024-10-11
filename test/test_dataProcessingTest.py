import unittest
import pandas as pd
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from dataProcessing import DataProcessing

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # Path to the Titanic dataset
        self.file_path = 'datasets/titanic/train.csv'
        self.categorical_columns = ['Pclass']  # Specify Pclass as a categorical column
        self.data_processing = DataProcessing(self.file_path, categorical_columns=self.categorical_columns)

    def test_load_data(self):
        data = self.data_processing.load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)  # Ensure data is loaded
        self.assertListEqual(
            list(data.columns),
            ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
        )

    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            DataProcessing('non_existent_file.csv').load_data()

    def test_load_data_empty_file(self):
        empty_file_path = 'datasets/titanic/empty.csv'
        with open(empty_file_path, 'w') as f:
            pass
        with self.assertRaises(pd.errors.EmptyDataError):
            DataProcessing(empty_file_path).load_data()
        os.remove(empty_file_path)

    def test_load_data_parser_error(self):
        invalid_file_path = 'datasets/titanic/invalid.csv'
        with open(invalid_file_path, 'w') as f:
            f.write('invalid,data\n1,2\n3,4,5')
        with self.assertRaises(pd.errors.ParserError):
            DataProcessing(invalid_file_path).load_data()
        os.remove(invalid_file_path)

    def test_detect_categorical_columns(self):
        self.data_processing.load_data()
        categorical_columns = self.data_processing.detect_categorical_columns()
        expected_categorical_columns = ['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
        self.assertListEqual(sorted(categorical_columns), sorted(expected_categorical_columns))

    def test_preprocess(self):
        self.data_processing.load_data()
        preprocessed_data = self.data_processing.preprocess()
        self.assertIsInstance(preprocessed_data, pd.DataFrame)
        self.assertFalse(preprocessed_data.isnull().values.any())
        self.assertTrue(all(preprocessed_data.dtypes != 'object'))

    def test_preprocess_missing_values(self):
        # Create a DataFrame with missing values
        data = {
            'Pclass': [1, 2, None],
            'Name': ['Name1', 'Name2', None],
            'Sex': ['male', None, 'female'],
            'Age': [22, None, 24],
            'SibSp': [1, 0, None],
            'Parch': [0, None, 1],
            'Ticket': ['A/5 21171', None, 'STON/O2. 3101282'],
            'Fare': [7.25, None, 8.05],
            'Cabin': [None, 'C85', None],
            'Embarked': ['S', None, 'C']
        }
        df = pd.DataFrame(data)
        self.data_processing.data = df
        preprocessed_data = self.data_processing.preprocess()
        self.assertFalse(preprocessed_data.isnull().values.any())

    def test_split_data(self):
        self.data_processing.load_data()
        X_train, X_test, y_train, y_test = self.data_processing.split_data(target_col='Survived')
        self.assertEqual(len(X_train) + len(X_test), len(self.data_processing.data))
        self.assertEqual(len(y_train) + len(y_test), len(self.data_processing.data))
        self.assertIn('Survived', y_train.name)
        self.assertIn('Survived', y_test.name)

    def test_split_data_no_target(self):
        self.data_processing.load_data()
        with self.assertRaises(ValueError):
            self.data_processing.split_data(target_col='NonExistentColumn')

    def test_balance_data(self):
        self.data_processing.load_data()
        balanced_data = self.data_processing.balance_data(target_col='Survived')
        self.assertIsInstance(balanced_data, pd.DataFrame)
        self.assertTrue('Survived' in balanced_data.columns)
        self.assertEqual(balanced_data['Survived'].value_counts().iloc[0], balanced_data['Survived'].value_counts().iloc[1])

    def test_balance_data_no_target(self):
        self.data_processing.load_data()
        with self.assertRaises(ValueError):
            self.data_processing.balance_data(target_col='NonExistentColumn')

    def test_end_to_end_workflow(self):
        # Load data
        self.data_processing.load_data()
        self.assertIsInstance(self.data_processing.data, pd.DataFrame)

        # Detect categorical columns
        categorical_columns = self.data_processing.detect_categorical_columns()
        expected_categorical_columns = ['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
        self.assertListEqual(sorted(categorical_columns), sorted(expected_categorical_columns))

        # Preprocess data
        preprocessed_data = self.data_processing.preprocess()
        self.assertIsInstance(preprocessed_data, pd.DataFrame)
        self.assertFalse(preprocessed_data.isnull().values.any())
        self.assertTrue(all(preprocessed_data.dtypes != 'object'))

        # Split data
        X_train, X_test, y_train, y_test = self.data_processing.split_data(target_col='Survived')
        self.assertEqual(len(X_train) + len(X_test), len(self.data_processing.data))
        self.assertEqual(len(y_train) + len(y_test), len(self.data_processing.data))

        # Balance data
        balanced_data = self.data_processing.balance_data(target_col='Survived')
        self.assertIsInstance(balanced_data, pd.DataFrame)
        self.assertTrue('Survived' in balanced_data.columns)
        self.assertEqual(balanced_data['Survived'].value_counts().iloc[0], balanced_data['Survived'].value_counts().iloc[1])

if __name__ == '__main__':
    unittest.main()