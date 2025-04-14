import os
import pandas as pd

class DataHandler:
    def __init__(self, directory):
        """
        Initialize the DataHandler with the directory containing the CSV files.
        :param directory: Path to the directory containing the CSV files.
        """
        self.directory = directory
        self.dataframes = []
        self.load_csv_files()

    def load_csv_files(self):
        """
        Load all CSV files in the specified directory that start with 'prices_round_1',
        maintaining their order based on the numeric suffix.
        """
        files = [
            file_name for file_name in os.listdir(self.directory)
            if file_name.startswith("prices_round_1") and file_name.endswith(".csv")
        ]
        # Sort files based on the numeric suffix
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        for file_name in files:
            file_path = os.path.join(self.directory, file_name)
            df = pd.read_csv(file_path, delimiter=";")
            self.dataframes.append(df)

    def new_index(self, df):
        """
        Set a new index for all dataframes based on the 'timestamp' column.
        """
        df["index"] = (df["timestamp"] + 1e6 * (df["day"]+2)).astype(int)
        return df

    def get_combined_dataframes(self):
        """
        Combine all loaded dataframes into a single dataframe.
        :return: A combined pandas DataFrame.
        """
        if not self.dataframes:
            raise ValueError("No dataframes loaded. Please load CSV files first.")
        df = pd.concat(self.dataframes, ignore_index=True)
        return self.new_index(df)

    def save_combined_dataframe(self, output_path):
        """
        Save the combined dataframe to a CSV file.
        :param output_path: Path to save the combined CSV file.
        """
        combined_df = self.get_combined_dataframes()
        combined_df.to_csv(output_path, index=False)