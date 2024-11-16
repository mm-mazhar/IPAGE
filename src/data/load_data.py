"""
Module contains functions for loading the dataset into a Pandas DataFrame.

NOTE:
    Ensure the that the scripts are ran from the root directory '/IPAGE'

The absolute path should always point to the root folder '~/*/IPAGE' where the
'data' folder is found.
"""
import pandas as pd
from pathlib import Path

SUPPORTED_FORMATS = ['.csv', '.xls', '.xlsx']


def verify_file(filename: str, data_dir: str | None = None):
    """
    Check for existing file with supported format.

    Args:
        filename (str): File name with supported file format
        data_dir (str, None): Relative or absolute path to the data folder.
            defaults to None.

    Raises:
        FileNotFoundError: If file or parent directory is not found
        ValueError: For an unsupported file format
    """
    path = Path()
    # `absolution_path` should point to the root folder '~/*/IPAGE'
    absolute_path = path.absolute()
    if not data_dir:
        data_dir = absolute_path.joinpath('data')
    else:
        data_dir = absolute_path.joinpath(data_dir)
    filepath = data_dir.joinpath(filename)

    if not filepath.exists():
        raise FileNotFoundError(f'File not found: "{str(filepath)}"')
    
    if not filepath.is_file():
        raise FileNotFoundError(f'File not found: "{str(filepath)}"')

    if filepath.suffix not in SUPPORTED_FORMATS:
        error_msg = f'Unsupported file format. Supported\
 file formats are {", ".join(SUPPORTED_FORMATS)}'
        raise ValueError(error_msg)
    return str(filepath)


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.

    Args:
        filepath (str): Valid path to a CSV file

    Returns:
        pd.DataFrame
    """
    return pd.read_csv(filepath)


def load_excel(
    filepath: str, sheet_name: int | str | list | None = 0
) -> pd.DataFrame:
    """
    Load a dataset from an Excel file.

    Args:
        filepath (str): Valid path to an Excel file
        sheet_name (int, str, list, None, optional): Defaults to 0.

    Returns:
        pd.DataFrame
    """
    return pd.read_excel(filepath, sheet_name=sheet_name)


def load_data(filename: str, data_dir: str = None) -> pd.DataFrame:
    """Load a dataset from a CSV or Excel file.

    Args:
        filename (str): File name with supported file format
        data_dir (str, None): Relative or absolute path to the data folder.
            defaults to None.

    Returns:
        pd.DataFrame
    """
    filepath = verify_file(filename, data_dir)

    # Checks if file extension is either a CSV or Excel file
    if Path(filepath).suffix == SUPPORTED_FORMATS[0]:
        return load_csv(filepath)
    else:
        return load_excel(filepath)
