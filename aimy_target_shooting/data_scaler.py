import json
import logging
import pathlib

import numpy as np


class DataScaler:
    """Scales data set according to preset scaling method."""

    def __init__(
        self,
        scaling_method: str = "standard",
        bias_value: float = None,
        scaling_value: float = None,
    ) -> None:
        """Initialises scaler.

        Args:
            scaling (str, optional): Scaling method. Supports standard and
            minmax scaling. Defaults to "standard".
            bias_value (float, optional): Predefined bias value.
            Defaults to None.
            scaling_value (float, optional): Predefined scaling value.
            Defaults to None.
        """
        self.scaling_method = scaling_method

        if bias_value is not None:
            self.bias_value = bias_value

        if scaling_value is not None:
            self.scaling_value = scaling_value

    def set_and_scale(self, data: np.ndarray) -> np.ndarray:
        """Determines the bias and scaling value of given data
        and scales data in minmax manner.

        Args:
            data (np.ndarray): Data to be scaled.

        Returns:
            np.ndarray: Scaled data
        """
        if self.scaling_method == "minmax":
            min_value = np.min(data, axis=0)
            max_value = np.max(data, axis=0)

            self.bias_value = min_value
            self.scaling_value = max_value - min_value

        if self.scaling_method == "standard":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            self.bias_value = mean
            self.scaling_value = std

        return (data - self.bias_value) / self.scaling_value

    def only_scale(self, data: np.ndarray) -> np.ndarray:
        """Scales data with preset bias and scaling value.

        Args:
            data (np.ndarray): Data to be scaled.

        Returns:
            np.ndarray: Scaled data.
        """
        return (data - self.bias_value) / self.scaling_value

    def unscale(self, scaled_data: np.ndarray) -> np.ndarray:
        """Reverses scaling of given data with preset
        bias and scaling value

        Args:
            scaled_data (np.ndarray): Scaled data to be unscaled.

        Returns:
            np.ndarray: Unscaled data.
        """
        return scaled_data * self.scaling_value + self.bias_value

    def export_values(self, export_path: pathlib.Path):
        """Exports scaler values to JSON file.

        Args:
            export_path (pathlib.Path): Export location and file name.
        """
        values = {"bias": list(self.bias_value), "scaling": list(self.scaling_value)}

        with open(export_path, "w") as outfile:
            json.dump(values, outfile)

    def import_values(self, import_path: pathlib.Path):
        """Imports scaler values from JSON file.

        Args:
            import_path (pathlib.Path): Import location and file name.
        """
        with open(import_path) as json_file:
            data = json.load(json_file)

            try:
                self.bias_value = np.array(data["bias"])
                self.scaling_value = np.array(data["scaling"])
            except KeyError as e:
                logging.error(f"File does not contain scaler values: {e}")
