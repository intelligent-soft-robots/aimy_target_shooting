import datetime
import json
import logging
import pathlib
import typing

import numpy as np
import tensorflow as tf

from aimy_target_shooting import export_tools
from aimy_target_shooting.configuration import get_config_path
from aimy_target_shooting.data_scaler import DataScaler
from aimy_target_shooting.learning_utils import generate_full_training_data


class TargetShootingNN:
    def __init__(self, verbose=True) -> None:
        self.model = None

        self.verbose = verbose

        path = get_config_path("learning")
        with open(path, "r") as file:
            self.conf = json.load(file)

    def load_model(self, import_path: pathlib.Path) -> None:
        """Loads a model from given import path.

        Args:
            import_path (pathlib.Path): Location of model file.
        """
        self.model = tf.keras.models.load_model(import_path)

        if self.verbose:
            logging.info("Model loaded.")

    def load_scaling(self, import_path: pathlib.Path = None) -> None:
        if import_path is None:
            import_path = "/tmp/nn_model/"

        self.control_scaler = DataScaler()
        self.target_scaler = DataScaler()

        control_file_name = "control_scaler.json"
        self.control_scaler.import_values(pathlib.Path(import_path + control_file_name))

        target_file_name = "target_scaler.json"
        self.target_scaler.import_values(pathlib.Path(import_path + target_file_name))

        if self.verbose:
            logging.info("Scaling loaded.")

    def export_model(self, export_path: typing.Optional[pathlib.Path] = None) -> None:
        """Exports current model to external file for deployment.

        Args:
            export_path (typing.Optional[pathlib.Path], optional): Path where the
            exported model will be stored. Defaults to temporary directory in Linux.
        """
        if export_path is None:
            export_path = pathlib.Path("/tmp/nn_model/model.hdf5")

        self.model.save(export_path)

        if self.verbose:
            logging.info("Model exported.")

    def export_scaling(self, export_path: typing.Optional[pathlib.Path] = None) -> None:
        if export_path is None:
            export_path = "/tmp/nn_model/"

        control_file_name = "control_scaler.json"
        self.control_scaler.export_values(pathlib.Path(export_path + control_file_name))

        target_file_name = "target_scaler.json"
        self.target_scaler.export_values(pathlib.Path(export_path + target_file_name))

        if self.verbose:
            logging.info("Scaling exported.")

    def generate_dataset(self, filepath: pathlib.Path) -> None:
        training_dataset = export_tools.import_all_from_hdf5(file_path=filepath)
        control_parameters, target_variables = generate_full_training_data(
            training_dataset
        )

        self.control_scaler = DataScaler()
        self.target_scaler = DataScaler()

        self.control_parameters_norm = self.control_scaler.set_and_scale(
            control_parameters
        )
        self.target_variables_norm = self.target_scaler.set_and_scale(target_variables)

        self.input_shape = self.target_variables_norm.shape[1]
        self.output_shape = self.control_parameters_norm.shape[1]

        if self.verbose:
            logging.info("Dataset generated.")

    def generate_MLP_model(
        self,
        n_layers: typing.Optional[int] = None,
        n_neurons: typing.Optional[typing.List[int]] = None,
        dropout: typing.Optional[float] = None,
    ) -> None:
        """Generates multilayer perceptron neural network model.

        Args:
            input_shape (tuple, optional): Input shape of network.
            Defaults to (5,).
            output_shape (int, optional): Output shape of network.
            Defaults to 5.
            n_layers (list): Number of layers. Defaults to 3.
            n_neurons (list): Number of neurons for each defined layer.
            Defaults to 108, 8, and 6.

        Returns:
            tf.keras.Model: Training model.
        """
        if n_layers is None:
            n_layers = self.conf["architecture"]["n_layers"]

        if n_neurons is None:
            n_neurons = self.conf["architecture"]["n_neurons"]

        if dropout is None:
            dropout = self.conf["architecture"]["dropout"]

        model = tf.keras.Sequential()

        if len(n_neurons) < n_layers:
            raise AttributeError(
                f"Not fitting number of neurons {n_neurons} to number "
                f"of layers {n_layers}"
            )

        model.add(tf.keras.Input(shape=self.input_shape))
        model.add(tf.keras.layers.Dense(n_neurons[0], activation="sigmoid"))

        if n_layers == 2:
            model.add(tf.keras.layers.Dropout(dropout))
            model.add(tf.keras.layers.Dense(n_neurons[1], activation="sigmoid"))
        if n_layers > 2:
            for i in range(1, n_layers):
                model.add(tf.keras.layers.Dense(n_neurons[i], activation="sigmoid"))

        model.add(tf.keras.layers.Dense(self.output_shape, activation="linear"))

        self.model = model

        if self.verbose:
            logging.info("MLP model generated.")
            self.model.summary()

    def train_model(
        self,
        epochs: typing.Optional[int] = None,
        optimizer: typing.Optional[str] = None,
        batch_size: typing.Optional[int] = None,
        verbose: typing.Optional[int] = None,
        validation_split: typing.Optional[float] = None,
        checkpoint_name: typing.Optional[str] = None,
        learning_rate: typing.Optional[float] = None,
    ) -> None:
        """Trains given keras model.

        Args:
            model (tf.keras.Model): Training model.
            training_set (np.ndarray): Target state data set.
            training_label (np.ndarray): Control values labels.
            epochs (typing.Optional[int], optional): Number of epochs. Defaults
            to None.
            optimizer (typing.Optional[str], optional): Optimizer for training.
            Defaults to None.
            batch_size (typing.Optional[int], optional): Batch size.
            Defaults to None.
            verbose (typing.Optional[int], optional): Print messages.
            Defaults to None.
            validation_split (typing.Optional[float], optional): Validation split.
            Defaults to None.
            checkpoint_name (typing.Optional[str], optional): Checkpoint name.
            Defaults to None.
            learning_rate (typing.Optional[float], optional): Learning rate.
            Defaults to None.

        Returns:
            tf.keras.History: Returns training history.
        """
        training_set = self.target_variables_norm
        training_label = self.control_parameters_norm

        tf.keras.backend.clear_session()

        path = get_config_path("learning")
        with open(path, "r") as file:
            conf = json.load(file)

        if epochs is None:
            epochs = conf["training"]["epochs"]

        if optimizer is None:
            optimizer = conf["training"]["optimizer"]

        if batch_size is None:
            batch_size = conf["training"]["batch_size"]

        if verbose is None:
            verbose = conf["training"]["verbose"]

        if validation_split is None:
            validation_split = conf["training"]["validation_split"]

        if learning_rate is None:
            learning_rate = conf["training"]["learning_rate"]

        if checkpoint_name is None:
            checkpoint_name = conf["training"]["checkpoint_name"]

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_name, save_best_only=True, monitor="val_loss"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=100, min_lr=learning_rate
            ),
            tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=50, verbose=1
            ),
        ]

        self.model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=[
                tf.keras.metrics.MeanSquaredError(),
            ],
        )

        self.history = self.model.fit(
            training_set,
            training_label,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
        )

        if self.verbose:
            logging.info("Model trained.")

    def launch_at_target(self, launcher, target: np.ndarray) -> None:
        """Launches to target with given launcher to predicted location.

        Args:
            launcher (_type_): Python API of physical launcher. Launcher is required
            to have a set_launcher_parameters() and launch() function.
            target (np.ndarray): Target location.
        """
        target_normalised = np.array([self.target_scaler.only_scale(target)])
        control_params = self.model.predict(target_normalised)
        control_params_unnormalised = self.control_scaler.unscale(control_params[0])

        launcher.set_launch_parameters(control_params_unnormalised)
        launcher.launch()

    def __exit__(self) -> None:
        self.export_model()
