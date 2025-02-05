import pandas as pd
import numpy as np
import os

class SignalAnalyzer:
    def __init__(self, csv_file_path):
        """
        Initialize the ReadingActions object.
        This class reads the data from a CSV file and analyzes the signals to detect noise on the signal direction.

        Parameters:
        csv_file_path (str): The file path of the CSV file containing the data.

        Attributes:
        df (pandas.DataFrame): The DataFrame containing the data from the CSV file.
        time_column (pandas.Series): The column representing the time values.
        signal_columns (pandas.DataFrame): The DataFrame containing the signal columns.
        change_points (list): A list to store the change points.
        """
        self.df = pd.read_csv(csv_file_path)
        self.time_column = self.df['time']
        self.signal_columns = self.df.drop(columns=['time'])
        self.change_points = []

    def analyze_signals(self):
        """
        Analyzes the signals and detects change points.

        This method analyzes the signals stored in the `signal_columns` attribute and detects change points
        where the slope of any signal changes. It updates the `change_points` attribute with the detected
        change points.

        Returns:
            None
        """
        # Initialize previous_slopes with the slopes for each signal
        previous_slopes = {signal_name: np.sign(self.signal_columns[signal_name].values[1] - self.signal_columns[signal_name].values[0]) for signal_name in self.signal_columns.columns}

        # for time, values in zip(self.time_column, self.signal_columns.values):
        for i in range(1, len(self.time_column)):
            current_slopes = {signal_name: np.sign(self.signal_columns[signal_name].values[i] - self.signal_columns[signal_name].values[i-1]) for signal_name in self.signal_columns.columns}
            # Check if any signal's slope changes at this time point
            if any(current_slopes[signal_name] != previous_slopes[signal_name] for signal_name in self.signal_columns.columns):
                change_point = {'time': self.time_column.iloc[i]}
                change_point.update({self.signal_columns.columns[k]: value for k, value in enumerate(self.signal_columns.iloc[i])})
                self.change_points.append(change_point)
            previous_slopes = current_slopes
        last_time = self.time_column.iloc[-1]
        last_values = self.signal_columns.iloc[-1]
        last_change_point = {'time': last_time, **{self.signal_columns.columns[i]: value for i, value in enumerate(last_values)}}
        self.change_points.append(last_change_point)

    def save_results(self, dir, name):
        """
        Save the change points to a CSV file.

        Args:
            dir (str): The directory path where the file will be saved.
            name (str): The name of the file.

        Returns:
            None
        """
        output_file = dir + 'for_execution/' + name
        change_points_df = pd.DataFrame(self.change_points)
        change_points_df.to_csv(output_file, index=False)

# Example usage:
# name = 'qt_arm_sides'
robotName = 'nao'
dir = 'data/test_' + robotName + '/GMM_learned_actions/'
# dir = './data/old_test_qt/GMM_learned_actions/'
csv_file_path = os.listdir(dir)
for name in csv_file_path:
    if name != "for_execution":
        analyzer = SignalAnalyzer(dir + name)
        analyzer.analyze_signals()
        analyzer.save_results(dir, name)
