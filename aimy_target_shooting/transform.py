import json
import math
from typing import List, Optional

import numpy as np

from aimy_target_shooting.configuration import get_config_path
from aimy_target_shooting.custom_types import TrajectoryCollection


def move_origin(
    trajectory_collection: TrajectoryCollection,
    reference_position: List[float] = None,
    first_point_reference: bool = False,
) -> TrajectoryCollection:
    """Sets origin of the coordinate system of the given trajectories according
    the given reference.

    Args:
        trajectory_collection (TrajectoryCollection): Collection of trajectories.
        reference_position (List[float], optional): Reference origin point.
        All samples are subtracted by reference point. Defaults to None.
        first_point_reference (bool, optional): The first samples can be also
        used as origin and define the reference point. Defaults to False.

    Returns:
        TrajectoryCollection: Collection of trajectories with transformed origin.
    """
    path = get_config_path("preprocessing")
    with open(path, "r") as file:
        config = json.load(file)

    if reference_position is None:
        reference_position = np.array(config["remove_bias"]["position_bias"])

    if first_point_reference is None:
        first_point_reference = bool(config["remove_bias"]["first_point_reference"])

    modified_trajectory_collection = trajectory_collection.deepcopy()

    for trajectory_data in modified_trajectory_collection:
        positions = np.array(trajectory_data["positions"])

        if first_point_reference:
            reference_position = np.average(trajectory_data["positions"][0:3], axis=0)

        positions -= reference_position

        trajectory_data["positions"] = [tuple(entry) for entry in positions]

    return modified_trajectory_collection


def rotate_coordinate_system(
    trajectory_collection: TrajectoryCollection,
    rotation_z_rad: Optional[float] = None,
) -> TrajectoryCollection:
    """Rotates coordinate system around z-axis. Default coordinate
    system defines z as height direction. x- and y-direction can be
    transformed with this function.

    Args:
        trajectory_collection (TrajectoryCollection): Collection of
        trajectories.
        rotation_z_rad (Optional[float], optional): Rotation around
        z-axis in radians.

    Returns:
        TrajectoryCollection: Collection of trajectories with
        transformed x- and y-values.
    """
    modified_trajectory_collection = trajectory_collection.deepcopy()

    if rotation_z_rad is None:
        path = get_config_path("preprocessing")
        with open(path, "r") as file:
            config = json.load(file)

        rotation_z_rad = math.radians(config["coordinate_rotation"]["rotation_z_deg"])

    for trajectory_data in modified_trajectory_collection:
        positions = []

        for point in trajectory_data["positions"]:
            rotated_point = [0.0, 0.0, 0.0]

            rotated_point[0] = (
                math.cos(rotation_z_rad) * point[0]
                - math.sin(rotation_z_rad) * point[1]
            )
            rotated_point[1] = (
                math.sin(rotation_z_rad) * point[0]
                + math.cos(rotation_z_rad) * point[1]
            )
            rotated_point[2] = 1 * point[2]

            positions.append(tuple(rotated_point))

        trajectory_data["positions"] = positions

        velocities = []

        for velocity in trajectory_data["velocities"]:
            rotated_velocity = [0.0, 0.0, 0.0]

            rotated_velocity[0] = (
                math.cos(rotation_z_rad) * velocity[0]
                - math.sin(rotation_z_rad) * velocity[1]
            )
            rotated_velocity[1] = (
                math.sin(rotation_z_rad) * velocity[0]
                + math.cos(rotation_z_rad) * velocity[1]
            )
            rotated_velocity[2] = 1 * velocity[2]

            velocities.append(tuple(rotated_velocity))

        trajectory_data["velocities"] = velocities

    return modified_trajectory_collection


def change_time_stamps(
    trajectory_collection: TrajectoryCollection, unit: str = "seconds"
) -> TrajectoryCollection:
    """Transforms time stamps to seconds from given unit.

    Args:
        trajectory_collection (TrajectoryCollection): Collection of
        trajectories.
        unit (str, optional): Unit of time stamps before transform.
        Defaults to "seconds".

    Returns:
        TrajectoryCollection: Collection of trajectories with transformed
        time stamps.
    """
    modified_trajectory_collection = trajectory_collection.deepcopy()

    unit_factor = {
        "seconds": 1e-9,
        "milliseconds": 1e-6,
        "microseconds": 1e-3,
        "nanoseconds": 1e-0,
    }

    for trajectory_data in modified_trajectory_collection:
        # transform from nano seconds to seconds
        trajectory_data["time_stamps"] = [
            i * unit_factor[unit] for i in trajectory_data["time_stamps"]
        ]

    return modified_trajectory_collection


def reset_time_stamps(
    trajectory_collection: TrajectoryCollection,
) -> TrajectoryCollection:
    """Sets time stamps in order to start with 0. Time differences
    between each time stamps are maintained.

    Args:
        trajectory_collection (TrajectoryCollection): Collection of
        trajectories.

    Returns:
        TrajectoryCollection: Collection with trajectories starting
        with time stamp 0.
    """
    modified_trajectory_collection = trajectory_collection.deepcopy()

    for trajectory_data in modified_trajectory_collection:
        trajectory_data.time_stamps = [
            i - trajectory_data.time_stamps[0] for i in trajectory_data.time_stamps
        ]

    return modified_trajectory_collection
