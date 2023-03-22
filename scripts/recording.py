import json
import logging
import pathlib

import numpy as np

from aimy_target_shooting.configuration import get_config_path
from aimy_target_shooting.recording import Recording


def grid_search() -> None:
    path = get_config_path("recording")
    with open(path, "r") as file:
        config = json.load(file)

    phi_range = config["gridsearch_param"]["phi_range"]
    theta_range = config["gridsearch_param"]["theta_range"]
    omega_tl_range = config["gridsearch_param"]["omega_tl_range"]
    omega_tr_range = config["gridsearch_param"]["omega_tr_range"]
    omega_bc_range = config["gridsearch_param"]["omega_bc_range"]

    n_phi_steps = config["gridsearch_param"]["n_phi_steps"]
    n_theta_steps = config["gridsearch_param"]["n_theta_steps"]
    n_omega_tl_steps = config["gridsearch_param"]["n_omega_tl_steps"]
    n_omega_tr_steps = config["gridsearch_param"]["n_omega_tr_steps"]
    n_omega_bc_steps = config["gridsearch_param"]["n_omega_bc_steps"]

    actuation_upper_threshold = config["gridsearch_param"]["actuation_upper_threshold"]
    actuation_lower_threshold = config["gridsearch_param"]["actuation_lower_threshold"]

    launcher_ip = config["recording_param"]["ip"]
    dir_path = pathlib.Path(config["recording_param"]["save_path"])
    prefix = config["recording_param"]["prefix"]

    env = Recording(launcher_ip=launcher_ip)

    phi_space = np.linspace(phi_range[0], phi_range[1], n_phi_steps)
    theta_space = np.linspace(theta_range[0], theta_range[1], n_theta_steps)
    omega_tl_space = np.linspace(omega_tl_range[0], omega_tl_range[1], n_omega_tl_steps)
    omega_tr_space = np.linspace(omega_tr_range[0], omega_tr_range[1], n_omega_tr_steps)
    omega_bc_space = np.linspace(omega_bc_range[0], omega_bc_range[1], n_omega_bc_steps)

    number_balls = (
        n_omega_bc_steps
        * n_omega_tl_steps
        * n_omega_tr_steps
        * n_phi_steps
        * n_theta_steps
    )
    logging.debug(f"{number_balls} to be shot")

    counter = 0

    for theta in np.nditer(theta_space):
        for phi in np.nditer(phi_space):
            for omega_tl in np.nditer(omega_tl_space):
                for omega_tr in np.nditer(omega_tr_space):
                    for omega_bc in np.nditer(omega_bc_space):

                        if (
                            actuation_lower_threshold > (omega_tl + omega_tr + omega_bc)
                            or (omega_tl + omega_tr + omega_bc)
                            > actuation_upper_threshold
                        ):
                            continue

                        # if omega_tl != omega_bc or omega_bc != omega_tr:
                        #    continue

                        # threshold = 800

                        # if (
                        #    abs(omega_bc - omega_tl) < threshold
                        #    or abs(omega_tr - omega_bc) < threshold
                        #    or abs(omega_tr - omega_tl) < threshold
                        # ):
                        #    continue

                        launch_parameters = (
                            phi,
                            theta,
                            omega_tl,
                            omega_tr,
                            omega_bc,
                        )
                        env.set_launch_parameters(launch_parameters)
                        env.record_and_launch()

                        counter += 1
                        logging.info(f"Shot No. {counter} fired.")

    env.export_recordings(prefix=prefix, path=dir_path, export_format="hdf5")
    env.export_recordings(prefix=prefix, path=dir_path, export_format="csv")

    print("Measurement completed.")


def run_single_measurement() -> None:
    path = get_config_path("recording")
    with open(path, "r") as file:
        config = json.load(file)

    dir_path = pathlib.Path(config["recording_param"]["save_path"])
    logging.info(f"Saving path: {dir_path}")

    prefix = config["recording_param"]["prefix"]

    launcher_ip = config["recording_param"]["ip"]

    launch_parameters = (
        config["single_measurement"]["phi"],
        config["single_measurement"]["theta"],
        config["single_measurement"]["omega_tl"],
        config["single_measurement"]["omega_tr"],
        config["single_measurement"]["omega_bc"],
    )

    env = Recording(launcher_ip=launcher_ip)

    env.set_launch_parameters(launch_parameters)
    env.record_and_launch()
    env.export_recordings(path=dir_path, prefix=prefix, export_format="hdf5")
    env.export_recordings(path=dir_path, export_format="csv")

    logging.info("Measurement completed.")


def run_multiple_measurements() -> None:
    path = get_config_path("recording")
    with open(path, "r") as file:
        config = json.load(file)

    dir_path = pathlib.Path(config["recording_param"]["save_path"])
    logging.info(f"Saving path: {dir_path}")

    prefix = config["recording_param"]["prefix"]
    launcher_ip = config["recording_param"]["ip"]
    number_shots = config["multiple_measurements"]["number_shots"]

    launch_parameters = (
        config["multiple_measurements"]["phi"],
        config["multiple_measurements"]["theta"],
        config["multiple_measurements"]["omega_tl"],
        config["multiple_measurements"]["omega_tr"],
        config["multiple_measurements"]["omega_bc"],
    )

    env = Recording(launcher_ip=launcher_ip)

    env.set_launch_parameters(launch_parameters)

    try:
        for i in range(number_shots):
            env.record_and_launch()
            logging.info(f"Measurement {i+1} completed")
    except KeyboardInterrupt:
        raise KeyboardInterrupt

    logging.info("All measurements completed")

    env.export_recordings(prefix=prefix, path=dir_path, export_format="hdf5")
    env.export_recordings(prefix=prefix, path=dir_path, export_format="csv")


def granny_launcher_recording():
    path = get_config_path("recording")
    with open(path, "r") as file:
        config = json.load(file)

    dir_path = pathlib.Path(config["recording_param"]["save_path"])
    prefix = config["recording_param"]["prefix"]
    clipping_time = config["granny_recording"]["clipping_time"]

    env = Recording()
    env.record_manual_launching(clipping_time=clipping_time)
    env.export_recordings(prefix=prefix, path=dir_path, export_format="csv")
    env.export_recordings(prefix=prefix, path=dir_path, export_format="hdf5")

    print("Measurement completed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-6s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_multiple_measurements()
    # run_single_measurement()
    # grid_search()
    # granny_launcher_recording()

    # room_recording()
