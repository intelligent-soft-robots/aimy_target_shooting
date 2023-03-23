import pathlib

from aimy_target_shooting.ball_launcher_api import BallLauncherAPI
from aimy_target_shooting.recording import Recording


def launch():
    launcher = BallLauncherAPI("10.42.26.171", 5555)
    launcher.set_rpm(0.5, 0.0, 0, 0, 0)
    launcher.launch()


def launch_and_record():
    launcher = "10.42.26.171"
    launch_parameters = (0.5, 0.5, 800, 800, 1600)

    env = Recording(launcher_ip=launcher)

    env.set_launch_parameters(launch_parameters, parameter_type="rpm")
    env.record_and_launch()

    dir_path = pathlib.Path("/tmp/recording")
    env.export_recordings(prefix="side_spin", path=dir_path, export_format="hdf5")


if __name__ == "__main__":
    launch()
