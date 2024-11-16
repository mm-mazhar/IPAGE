"""Module for orchestrating data unit tests.
"""
import os
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock
from tests.utils import (
    setup_test_folder, setup_test_dataset,
    teardown_folder, teardown_dataset
)
from src.data.load_data import verify_file, load_data, load_csv, load_excel

SUPPORTED_FORMATS = ['.csv', '.xls', '.xlsx']


class TestLoadCSVExcel(unittest.TestCase):
    """
    Test `load_csv` function for loading CSV file to a Pandas Dataframe.
    """

    @patch("pandas.read_csv", new_callable=MagicMock)
    def test_load_csv(self, mock_read_csv):
        """Test `load_csv` function by using a mock on `pandas.read_csv`"""
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_csv.return_value = mock_df

        filepath = 'dummy_path.csv'
        result = load_csv(filepath)

        mock_read_csv.assert_called_once_with(filepath)

        self.assertIs(result, mock_df)
        self.assertEqual(result.shape, (2, 2))

    @patch("pandas.read_excel", new_callable=MagicMock)
    def test_load_excel(self, mock_read_excel):
        """Test `load_excel` function by using a mock on `pandas.read_excel`"""
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_excel.return_value = mock_df

        filepath = 'dummy_path.csv'
        result = load_excel(filepath)

        mock_read_excel.assert_called_with(filepath, sheet_name=0)

        sheet_name = 1
        result = load_excel(filepath, sheet_name=sheet_name)

        mock_read_excel.assert_called_with(filepath, sheet_name=sheet_name)

        self.assertIs(result, mock_df)
        self.assertEqual(result.shape, (2, 2))


class TestFileOperations(unittest.TestCase):
    """
    Test `verify_file` and `load_data` functions.
    """

    @classmethod
    def setUpClass(cls):
        """Set up a temporary folder and dataset for testing."""
        cls.folder_name = 'test_data'
        setup_test_folder(cls.folder_name)
        data = {'col1': [1, 2], 'col2': [3, 4]}
        cls.valid_csv_path = setup_test_dataset(
            cls.folder_name, 'test.csv', data=data
        )
        cls.valid_excel_path = setup_test_dataset(
            cls.folder_name, 'test.xlsx', data=data
        )
        cls.invalid_file_path = os.path.join(cls.folder_name, 'test.txt')

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary folder and dataset after testing."""
        teardown_dataset(cls.valid_csv_path)
        teardown_dataset(cls.valid_excel_path)
        if os.path.exists(cls.invalid_file_path):
            os.remove(cls.invalid_file_path)
        teardown_folder(cls.folder_name)

    def test_verify_file_exists_and_format_valid_csv(self):
        """Test verify_file with a valid CSV file."""
        result = verify_file('test.csv', data_dir=self.folder_name)
        self.assertEqual(result, self.valid_csv_path)

    def test_verify_file_exists_and_format_valid_excel(self):
        """Test verify_file with a valid Excel file."""
        result = verify_file('test.xlsx', data_dir=self.folder_name)
        self.assertEqual(result, self.valid_excel_path)

    def test_verify_file_not_exists(self):
        """Test verify_file raises FileNotFoundError when file is missing."""
        with self.assertRaises(FileNotFoundError):
            verify_file('nonexistent.csv', data_dir=self.folder_name)

    def test_verify_file_invalid_format(self):
        """Test verify_file raises ValueError when format is invalid."""
        with open(self.invalid_file_path, 'w') as f:
            f.write('invalid content')
        with self.assertRaises(ValueError):
            verify_file('test.txt', data_dir=self.folder_name)

    def test_load_data_csv(self):
        """Test load_data function loads CSV files correctly."""
        df = load_data('test.csv', data_dir=self.folder_name)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))

    def test_load_data_excel(self):
        """Test load_data function loads Excel files correctly."""
        df = load_data('test.xlsx', data_dir=self.folder_name)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))

    @patch("pandas.read_csv")
    def test_load_data_csv_with_mock(self, mock_read_csv):
        """Test load_data for CSV file using a mock for read_csv."""
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_csv.return_value = mock_df

        df = load_data('test.csv', data_dir=self.folder_name)

        mock_read_csv.assert_called_once_with(self.valid_csv_path)
        self.assertIs(df, mock_df)

    @patch("pandas.read_excel")
    def test_load_data_excel_with_mock(self, mock_read_excel):
        """Test load_data for Excel file using a mock for read_excel."""
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_excel.return_value = mock_df

        df = load_data('test.xlsx', data_dir=self.folder_name)

        mock_read_excel.assert_called_once_with(
            self.valid_excel_path, sheet_name=0)
        self.assertIs(df, mock_df)


if __name__ == '__main__':
    unittest.main()
