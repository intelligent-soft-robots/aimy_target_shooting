import warnings

import numpy as np
from ball_launcher_beepy import BallLauncherClient
from scipy.interpolate import interp1d

from aimy_target_shooting.utility import ip_to_launcher_name


class BallLauncherAPI:
    """Wrapper class for BallLauncherClient. Extends functionality of
    BallLauncherClient class with custom functions.
    """

    def __init__(self, config: dict) -> None:
        """Initialises BallLauncher class.
        Remark: Ball launcher and operating device have to be in same network.

        Args:
            config (dict): Configuration file.
        """
        self.config = config

        self._launcher_ip = config["launcher_ip"]
        self._launcher_port = config["launcher_port"]
        self._client = BallLauncherClient(self._launcher_ip, self._launcher_port)

    def set_actuation(
        self,
        phi: float = 0.5,
        theta: float = 0.5,
        top_left_actuation: float = 0.0,
        top_right_actuation: float = 0.0,
        bottom_center_actuation: float = 0.0,
    ):
        """Sets motor speeds according to actuation between 0 and 1.

        Args:
            phi (float, optional): Azimutal angle of the launch pad. Defaults
            to 0.5.
            theta (float, optional): Altitute angle of the launch pad.
            Defaults to 0.5.
            top_left_actuation (float, optional): Activation of the top left
            motor. Defaults to 0.0.
            top_right_actuation (float, optional): Activation of the top right
            motor. Defaults to 0.0.
            bottom_center_actuation (float, optional): Activation of the bottom
            center motor. Defaults to 0.0.
        """
        self._phi = phi
        self._theta = theta
        self._omega_top_left = top_left_actuation
        self._omega_top_right = top_right_actuation
        self._omega_bottom_center = bottom_center_actuation

        self.set_state()

    def set_rpm(
        self,
        phi: float = 0.5,
        theta: float = 0.5,
        rpm_top_left: float = 0.0,
        rpm_top_right: float = 0.0,
        rpm_bottom_center: float = 0.0,
    ):
        """Sets motor speeds according to desired rpm

        Args:
            phi (float, optional): Azimuthal angle of the launch pad. Defaults
            to 0.5.
            theta (float, optional): Altitude angle of the launch pad.
            Defaults to 0.5.
            top_left_actuation (float, optional): Activation of the top left
            motor. Defaults to 0.0.
            top_right_actuation (float, optional): Activation of the top right
            motor. Defaults to 0.0.
            bottom_center_actuation (float, optional): Activation of the bottom
            center motor. Defaults to 0.0.
        """
        launcher = ip_to_launcher_name(self._launcher_ip)

        fitting_method = self.config["fitting_method"]
        rpm_tl = self.config[launcher]["rpm_tl"]
        rpm_tr = self.config[launcher]["rpm_tr"]
        rpm_bc = self.config[launcher]["rpm_bc"]

        rpm_list = [rpm_top_left, rpm_top_right, rpm_bottom_center]

        actuation = self.config[launcher]["actuation"]

        set_flag = True
        minimum_rpm = float(max([min(rpm_tr), min(rpm_tl), min(rpm_bc)]))
        maximum_rpm = float(min([max(rpm_tr), max(rpm_tl), max(rpm_bc)]))

        for rpm in rpm_list:
            if minimum_rpm > rpm > maximum_rpm:
                warnings.warn("Given speed cannot be set. Set is ommitted.")
                set_flag = False

        if set_flag:
            f_tl = interp1d(rpm_tl, actuation, kind=fitting_method)
            f_tr = interp1d(rpm_tr, actuation, kind=fitting_method)
            f_bc = interp1d(rpm_bc, actuation, kind=fitting_method)

            self._phi = phi
            self._theta = theta
            self._omega_top_left = f_tl(rpm_top_left)
            self._omega_top_right = f_tr(rpm_top_right)
            self._omega_bottom_center = f_bc(rpm_bottom_center)

        self.set_state()

    def set_state(self):
        """Interfaces set state function of launcher client and launcher API."""
        self._client.set_state(
            self._phi,
            self._theta,
            self._omega_top_left,
            self._omega_top_right,
            self._omega_bottom_center,
        )

    def sample(self):
        """Generates random parameters within boundaries of the ball launcher and
        calls launch function with these parameters.

        """
        self._phi = np.random.uniform(0, 1)
        self._theta = np.random.uniform(0, 1)
        self._omega_top_left = np.random.uniform(0, 1)
        self._omega_top_right = np.random.uniform(0, 1)
        self._omega_bottom_center = np.random.uniform(0, 1)

        self.set_state()
        self.launch()

    def launch(self):
        """Calls launch command of ball launcher beepy."""
        self._client.launch_ball()
