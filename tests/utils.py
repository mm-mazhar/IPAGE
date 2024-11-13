import os
import pandas as pd


def setup_test_folder(folder_name):
    """
    Set up a temporary test folder for testing data.
    """
    os.mkdir(folder_name)


def setup_test_dataset(folder_name, filename: str, data: list | dict = []):
    """
    Set up a temporary test dataset.

    Args:
        data (list | dict, optional): Valid data structure. Defaults to [].
    """
    df = pd.DataFrame(data)
    filepath = os.path.join(os.path.abspath('.'), folder_name, filename)
    
    if filename.endswith(".csv"):
        df.to_csv(filepath, index=False)
    else:
        df.to_excel(filepath, index=False)

    return filepath

   
def teardown_folder(folder_name):
    """Delete the temporary test folder created for testing."""
    os.rmdir(folder_name)

    
def teardown_dataset(filepath):
    """Delete file from temporary test folder."""
    os.remove(filepath)
    