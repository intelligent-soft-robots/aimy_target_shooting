import logging
import pathlib

import numpy as np

from aimy_target_shooting.ball_launcher_api import BallLauncherAPI
from aimy_target_shooting.target_shooting_nn import TargetShootingNN


def train_target_shooting():
    file_path = pathlib.Path(
        "/home/adittrich/nextcloud/82_Data_Processed/"
        "MN5008_training_data_with_outlier/"
        "MN5008_grid_data_all.hdf5"
    )

    nn = TargetShootingNN()
    nn.generate_dataset(filepath=file_path)
    nn.generate_MLP_model()
    nn.train_model()

    nn.export_model()
    nn.export_scaling()


def launch_ball_loaded_model():
    launcher = BallLauncherAPI(launcher_ip="10.42.26.171")

    nn = TargetShootingNN()
    nn.load_model(pathlib.Path("/tmp/nn_model/model.hdf5"))
    nn.load_scaling()

    target_position = np.array((3.0, 1.0, 0.76))
    control_parameters = nn.compute_control_parameters(target_position)

    launcher.set_rpm(*control_parameters)
    launcher.launch()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-6s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    np.set_printoptions(suppress=True)

    # train_target_shooting()
    launch_ball_loaded_model()
