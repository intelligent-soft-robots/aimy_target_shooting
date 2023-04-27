import typing

import numpy as np

from aimy_target_shooting.custom_types import TrajectoryCollection, TrajectoryData
from aimy_target_shooting.hitpoint_detection import HitPointDetection
from aimy_target_shooting.utility import cartesian_to_polar


def generate_full_training_data(
    trajectory_collection: TrajectoryCollection, config: dict
) -> typing.List[np.ndarray]:
    """Generates training data with the mapping of control parameters
    to each sample of each ball trajectory in given collection.

    Args:
        trajectory_collection (TrajectoryCollection): Collection
        of trajectories.

    Returns:
        typing.List[np.ndarray, np.ndarray]: training data set with
        target states and training labels with control parameters.
    """
    fitting_window = config["dataset_generation"]["fitting_window"]

    control_parameters = []
    target_variables = []

    for trajectory_data in trajectory_collection:
        ctl_params = trajectory_data.launch_param

        for idx in range(fitting_window, len(trajectory_data)):
            positions = trajectory_data.positions[idx]
            velocities = extract_velocities(trajectory_data, idx, fitting_window)

            trg_params = create_targets(positions, velocities, config)

            control_parameters.append(ctl_params)
            target_variables.append(trg_params)

    control_parameters = np.array(control_parameters)
    target_variables = np.array(target_variables)

    return control_parameters, target_variables


def generate_hitpoint_training_data(
    trajectory_collection: TrajectoryCollection,
    config: dict,
) -> typing.List[np.ndarray]:
    """Generates training data with the mapping of control parameters
    to each detected hitpoint of each ball trajectory of given collection.

    Args:
        trajectory_collection (TrajectoryCollection): Collection
        of trajectories.

    Returns:
        typing.List[np.ndarray, np.ndarray]: training data set with
        target states and training labels with control parameters.
    """
    fitting_window = config["dataset_generation"]["fitting_window"]

    evaluator = HitPointDetection()
    trajectory_collection = evaluator.evaluate_hitpoints(trajectory_collection)

    control_parameters = []
    target_variables = []

    for trajectory_data in trajectory_collection:
        ctl_params = trajectory_data.launch_param

        if trajectory_data.hitpoints:
            positions = trajectory_data.hitpoints[0]
        else:
            continue

        index = trajectory_data.discontinuity_indices[0]

        if 0 > index - fitting_window or index > len(trajectory_data):
            continue

        velocities = extract_velocities(trajectory_data, index, fitting_window)

        trg_params = create_targets(positions, velocities, config)

        control_parameters.append(ctl_params)
        target_variables.append(trg_params)

    control_parameters = np.array(control_parameters)
    target_variables = np.array(target_variables)

    return control_parameters, target_variables


def create_targets(
    positions: np.ndarray, velocities: np.ndarray, config: dict
) -> np.ndarray:
    """Creates target states according to JSON config
    file.

    Args:
        positions (np.ndarray): Sample position.
        velocities (np.ndarray): Sample velocity.

    Returns:
        np.ndarray: Target parameters according to configuration
        file.
    """
    trg_params = []
    alpha, beta = compute_direction(velocities)

    if config["dataset_generation"]["polar_transform"]:
        positions = cartesian_to_polar(*positions)

    if config["dataset_generation"]["p_x"]:
        trg_params.append(positions[0])

    if config["dataset_generation"]["p_y"]:
        trg_params.append(positions[1])

    if config["dataset_generation"]["p_z"]:
        trg_params.append(positions[2])

    if config["dataset_generation"]["v_x"]:
        trg_params.append(velocities[0])

    if config["dataset_generation"]["v_y"]:
        trg_params.append(velocities[1])

    if config["dataset_generation"]["v_z"]:
        trg_params.append(velocities[2])

    if config["dataset_generation"]["alpha"]:
        trg_params.append(alpha)

    if config["dataset_generation"]["beta"]:
        trg_params.append(beta)

    if config["dataset_generation"]["v_mag"]:
        v_abs = np.linalg.norm(velocities)
        trg_params.append(v_abs)

    return trg_params


def compute_direction(state: typing.List[float]) -> typing.List[float]:
    """Computes spherical coordinates from position.

    Args:
        state (typing.List[float]): Position changes.

    Returns:
        typing.List[float, float]: Azimuthal and altitude angle.
    """
    spherical = cartesian_to_polar(*state)

    alpha = spherical[1]
    beta = spherical[2]

    return alpha, beta


def extract_velocities(
    trajectory_data: TrajectoryData,
    index: int,
    fitting_window: int,
) -> np.ndarray:
    """Extracts velocities at given index from ball trajectory via
    polynomial fitting.

    Args:
        trajectory_data (TrajectoryData): Ball trajectory.
        index (int): Velocity index.
        fitting_window (int): Neighborhood of samples evaluated
        for polynomial fitting.

    Raises:
        ValueError: Raised if desired index does not fit the
        requested fitting window.

    Returns:
        np.ndarray: Velocity in each spatial direction.
    """
    if 0 > index - fitting_window or index > len(trajectory_data):
        raise ValueError(
            f"Index {index} cannot be fitted. Out of scope."
            "Length trajectory data: {len(trajectory_data)}"
        )

    time_stamps = trajectory_data.time_stamps[index - fitting_window : index]
    positions = trajectory_data.positions[index - fitting_window : index]

    coefs = np.polynomial.polynomial.polyfit(time_stamps, positions, deg=1)
    velocities = list(coefs[1, :])

    return velocities
