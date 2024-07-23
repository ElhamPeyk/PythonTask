import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
from  TaskPython import DataProcessor, CSVLoader, TrainingDataLoader, IdealFunctionLoader, TestDataLoader
from sqlalchemy import create_engine, MetaData, Table, Column, Float, String



class TestCSVLoader(unittest.TestCase):

    def setUp(self):
        self.loader = CSVLoader('test.csv')

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_load(self, mock_read_csv, mock_exists):
        mock_exists.return_value = True
        self.loader.load()
        mock_read_csv.assert_called_once_with('test.csv')

    def test_get_dataframe_without_load(self):
        with self.assertRaises(ValueError):
            self.loader.get_dataframe()


class TestTrainingDataLoader(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_load(self, mock_read_csv):
        mock_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9],
            'D': [10, 11, 12],
            'E': [13, 14, 15]
        })
        mock_read_csv.return_value = mock_df
        loader = TrainingDataLoader('test.csv')
        loader.load()
        self.assertEqual(list(loader.df.columns), ['x', 'y1', 'y2', 'y3', 'y4'])


class TestIdealFunctionLoader(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_load(self, mock_read_csv):
        mock_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9],
            # Add more columns as needed
        })
        mock_read_csv.return_value = mock_df
        loader = IdealFunctionLoader('test.csv')
        loader.load()
        expected_columns = ['x'] + [f'ideal_y{i}' for i in range(1, len(mock_df.columns))]
        self.assertEqual(list(loader.df.columns), expected_columns)


class TestTestDataLoader(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_load(self, mock_read_csv):
        mock_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        mock_read_csv.return_value = mock_df
        loader = TestDataLoader('test.csv')
        loader.load()
        self.assertEqual(list(loader.df.columns), ['x', 'y'])


class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.db_folder = 'TestDatabase'
        self.db_name = 'TestPythonTaskDataBase.db'
        self.processor = DataProcessor(db_folder=self.db_folder, db_name=self.db_name)

    def tearDown(self):
        db_path = os.path.join(self.db_folder, self.db_name)
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(self.db_folder):
            os.rmdir(self.db_folder)

    @patch('os.makedirs')
    def test_ensure_db_directory(self, mock_makedirs):
        DataProcessor(db_folder='NewDatabase')
        mock_makedirs.assert_called_once_with('NewDatabase')

    def test_create_tables(self):
        with patch.object(self.processor.metadata, 'create_all') as mock_create_all:
            self.processor.create_tables()
            mock_create_all.assert_called_once()

    @patch('pandas.DataFrame.to_sql')
    def test_save_to_db(self, mock_to_sql):
        df = pd.DataFrame({'x': [1, 2, 3], 'y1': [4, 5, 6], 'y2': [7, 8, 9]})
        self.processor.save_to_db(df, 'TestTable')
        mock_to_sql.assert_called_once_with('TestTable', self.processor.engine, if_exists='replace', index=False)

    def test_find_best_fit(self):
        training_df = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1, 2, 3],
            'y2': [2, 4, 6]
        })
        ideal_df = pd.DataFrame({
            'x': [1, 2, 3],
            'ideal_y1': [1, 2, 3],
            'ideal_y2': [2, 4, 6],
            'ideal_y3': [3, 6, 9]
        })
        best_fit = self.processor.find_best_fit(training_df, ideal_df)
        expected_best_fit = {'y1': 'ideal_y1', 'y2': 'ideal_y2'}
        self.assertEqual(best_fit, expected_best_fit)

    def test_calculate_deviation(self):
        test_df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [1, 2, 3]
        })
        ideal_df = pd.DataFrame({
            'x': [1, 2, 3],
            'ideal_y1': [1, 2, 3],
            'ideal_y2': [2, 4, 6]
        })
        best_fit_functions = {'y': 'ideal_y1'}
        deviations = self.processor.calculate_deviation(test_df, ideal_df, best_fit_functions)
        expected_deviations = [
            [1, 1, 'ideal_y1', 0],
            [2, 2, 'ideal_y1', 0],
            [3, 3, 'ideal_y1', 0]
        ]
        self.assertEqual(deviations, expected_deviations)


if __name__ == '__main__':
    unittest.main()

