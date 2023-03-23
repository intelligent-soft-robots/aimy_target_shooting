import json
import logging
import pathlib
import time
from datetime import datetime

import signal_handler
import tennicam_client

from aimy_target_shooting.ball_launcher_api import BallLauncherAPI
from aimy_target_shooting.configuration import get_config_path
from aimy_target_shooting.custom_types import (
    LaunchParameter,
    TrajectoryCollection,
    TrajectoryData,
)
from aimy_target_shooting.export_tools import export_data


class Recording:
    """BallRecording enables launching and recording of ball
    trajectories and unifies the interfaces to tennicam, the ball
    launcher and other custom developed lab specific packages.

    According to the notations usually used by the RL community,
    BallRecording is used for generating samples and represents
    the agent.
    """

    def __init__(self, launcher_ip: str = None) -> None:
        """Initialises recording environment.

        Args:
            launcher_ip (str, optional): IP address of launcher. Defaults to None.
        """
        path = get_config_path("recording")
        with open(path, "r") as file:
            config = json.load(file)

        self.record_duration_s = config["recording_param"]["recording_duration"]
        self.tennicam_segment_id = config["recording_param"]["tennicam_segment_id"]

        self.trajectory_collection = TrajectoryCollection()

        if launcher_ip is None:
            launcher_ip = config["recording_param"]["ip"]

        self.launcher = BallLauncherAPI(launcher_ip=launcher_ip)

    def set_launch_parameters(
        self, launch_parameters: LaunchParameter, parameter_type: str = "rpm"
    ) -> None:
        """Sets launch parameters for ball_launcher.

        Args:
            launch_parameters (tuple): launch parameters according to convention
            phi, theta, omega top left, omega top right, omega bottom.
        """
        self.launch_param = launch_parameters
        if parameter_type == "rpm":
            self.launcher.set_rpm(*launch_parameters)
        elif parameter_type == "actuation":
            self.launcher.set_actuation(*launch_parameters)
        else:
            raise AttributeError(f"Given parameter {parameter_type} is not valid.")

    def record_manual_launching(self, clipping_time: float) -> None:
        """Records all trajectories and clips separate trajectories.

        Args:
            clipping_time (float): Time between two separate
            trajectories.
        """
        ball_frontend = tennicam_client.FrontEnd(self.tennicam_segment_id)
        iteration = ball_frontend.latest().get_iteration()
        signal_handler.init()  # for detecting ctrl+c

        try:
            trajectory_data = TrajectoryData()

            clip_flag = True
            start_time = time.time()

            while not signal_handler.has_received_sigint():
                iteration += 1
                obs = ball_frontend.read(iteration)

                ball_id = obs.get_ball_id()
                time_stamp = obs.get_time_stamp()
                position = obs.get_position()
                velocity = obs.get_velocity()

                trajectory_data.append_sample(ball_id, time_stamp, position, velocity)

                if ball_id == -1 and clip_flag:
                    start_time = time.time()
                    clip_flag = False

                if ball_id != -1 and not clip_flag:
                    clip_flag = True

                if not clip_flag and time.time() - start_time > clipping_time:
                    if len(trajectory_data) > 100:
                        logging.info(
                            f"Data with length {len(trajectory_data)} recorded."
                        )
                        self.trajectory_collection.append(trajectory_data)
                        logging.info(
                            f"Collection contains {len(self.trajectory_collection)}"
                            "samples."
                        )

                    trajectory_data = TrajectoryData()
                    clip_flag = True

        except (KeyboardInterrupt, SystemExit):
            pass
        except Exception as e:
            print("Error while recording ball values:", e)

    def record(self) -> None:
        """Records one trajectory without launching."""
        frontend = tennicam_client.FrontEnd(self.tennicam_segment_id)
        iteration = frontend.latest().get_iteration()
        signal_handler.init()  # for detecting ctrl+c

        trajectory_data = TrajectoryData()
        trajectory_data.set_launch_param((0.0, 0.0, -1.0, -1.0, -1.0))

        start_time_s = time.time()
        try:
            while not signal_handler.has_received_sigint():
                iteration += 1
                obs = frontend.read(iteration)

                ball_id = obs.get_ball_id()
                time_stamp = obs.get_time_stamp()
                position = obs.get_position()
                velocity = obs.get_velocity()

                trajectory_data.append_sample(ball_id, time_stamp, position, velocity)

        except (KeyboardInterrupt, SystemExit):
            pass
        except Exception as e:
            logging.error(f"Error while recording ball values: {e}")

        logging.debug(f"Time: {time.time()- start_time_s}")
        self.trajectory_collection.append(trajectory_data)

    def record_and_launch(self) -> None:
        """Records ball ID, ball position, ball velocity and time stamps for
        specified recording duration after launching the ball. Ball data is
        stored in data manager.
        """
        trajectory_data = TrajectoryData()
        trajectory_data.set_launch_param(self.launch_param)

        frontend = tennicam_client.FrontEnd(self.tennicam_segment_id)
        iteration = frontend.latest().get_iteration()

        start_time_s = time.time()
        try:
            launch_flag = True
            while time.time() - start_time_s <= self.record_duration_s:
                if launch_flag:
                    self.launcher.launch()
                    launch_flag = False

                iteration += 1
                obs = frontend.read(iteration)

                ball_id = obs.get_ball_id()
                time_stamp = obs.get_time_stamp()
                position = obs.get_position()
                velocity = obs.get_velocity()

                trajectory_data.append_sample(ball_id, time_stamp, position, velocity)

        except (KeyboardInterrupt, SystemExit):
            pass
        except Exception as e:
            logging.error(f"Error while recording ball values: {e}")

        logging.debug(f"Time: {time.time()- start_time_s}")
        self.trajectory_collection.append(trajectory_data)

    def export_recordings(
        self,
        path: pathlib.Path = None,
        export_format: str = "hdf5",
        prefix: str = "ball_trajectories",
    ) -> None:
        """Calls the respective export function of data manager
        according to specified export data format.

        Args:
            export_format (str, optional): export data format.
            Defaults to "hdf5".
            prefix (str, optional): prefix name of exported file.
        """
        now = datetime.now()
        time_stamp = now.strftime("%Y%m%d%H%M%S")

        if export_format == "hdf5":
            prefix = time_stamp + "_" + prefix
        elif export_format == "csv":
            prefix = time_stamp + "_" + prefix + "_"

        export_data(
            trajectory_collection=self.trajectory_collection,
            export_path=path,
            export_format=export_format,
            prefix=prefix,
        )
