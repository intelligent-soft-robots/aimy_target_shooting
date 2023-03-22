import pathlib
import logging

from aimy_target_shooting.ball_launcher_api import BallLauncherAPI
from aimy_target_shooting.target_shooting_nn import TargetShootingNN

def train_target_shooting():
    file_path = pathlib.Path(
        "/home/adittrich/nextcloud/82_Data_Processed/"
        "MN5008_training_data_with_outlier/"
        "MN5008_grid_data_equal_speeds.hdf5"
    )

    nn = TargetShootingNN()
    nn.generate_dataset(filepath=file_path)
    nn.generate_MLP_model()
    nn.train_model()

    nn.export_model()
    nn.export_scaling()

def launch_ball_loaded_model():
    #launcher = BallLauncherAPI(launcher_ip="10.42.26.171")

    nn = TargetShootingNN()
    nn.load_model(pathlib.Path("/tmp/nn_model/model.hdf5"))
    nn.load_scaling()

    print(nn.target_scaler.bias_value)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-6s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    target_position = (12.0, 12.0, 12.0)
    launch_ball_loaded_model()
