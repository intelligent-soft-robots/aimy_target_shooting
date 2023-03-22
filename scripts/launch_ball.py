import pathlib
import time

from aimy_target_shooting.ball_launcher_api import BallLauncherAPI
from aimy_target_shooting.recording import Recording


def motor_calibration():
    launcher_v2 = BallLauncherAPI("10.42.31.174", 5555)

    phi = 0.5
    theta = 0.0

    o_tl = 1430
    o_tr = 1460
    o_bc = 1260

    launcher_v2.set_rpm(
        phi=phi,
        theta=theta,
        rpm_top_left=o_tl,
        rpm_top_right=o_tr,
        rpm_bottom_center=o_bc,
    )

    time.sleep(30)

    launcher_v2.set_rpm(
        phi=phi,
        theta=theta,
        rpm_top_left=0.0,
        rpm_top_right=0.0,
        rpm_bottom_center=0.0,
    )


def launch():
    # launcher_v2 = BallLauncherAPI("10.42.31.174", 5555)
    # launcher_v2.set_rpm(0.5, 1.0, 1000, 1000, 1000)
    # launcher_v2.launch()

    launcher_v3 = BallLauncherAPI("10.42.26.171", 5555)
    launcher_v3.set_rpm(0.5, 0.0, 0, 0, 0)
    # launcher_v3.launch()


def launch_and_record():
    # launcher_v3_ip = "10.42.26.171"
    launcher_v2_ip = "10.42.26.171"

    launch_parameters = (0.5, 0.5, 800, 800, 1600)

    env = Recording(launcher_ip=launcher_v2_ip)

    env.set_launch_parameters(launch_parameters, parameter_type="rpm")
    env.record_and_launch()

    dir_path = pathlib.Path("/home/adittrich/Desktop/1221212_side_spin")
    env.export_recordings(prefix="side_spin", path=dir_path, export_format="hdf5")
    env.export_recordings(prefix="side_spin", path=dir_path, export_format="csv")


if __name__ == "__main__":
    # motor_calibration()
    launch()
