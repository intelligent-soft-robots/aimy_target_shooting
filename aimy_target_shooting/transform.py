import math
import typing

import numpy as np

from aimy_target_shooting.custom_types import TrajectoryCollection, TransformTimeUnits


def move_origin(
    trajectory_collection: TrajectoryCollection, config: dict
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

    try:
        reference_position = np.array(config["remove_bias"]["position_bias"])
        first_point_reference = bool(config["remove_bias"]["first_point_reference"])
    except Exception as e:
        raise AttributeError(f"Parameter not included in configuration file. {e}")

    modified_trajectory_collection = trajectory_collection.deepcopy()

    for trajectory_data in modified_trajectory_collection:
        positions = np.array(trajectory_data.positions)

        if first_point_reference:
            reference_position = np.average(trajectory_data.positions[0:3], axis=0)

        positions -= reference_position

        trajectory_data.positions = [tuple(entry) for entry in positions]

    return modified_trajectory_collection


def rotate_coordinate_system(
    trajectory_collection: TrajectoryCollection,
    config: dict,
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

    try:
        rotation_z_rad = math.radians(config["coordinate_rotation"]["rotation_z_deg"])
    except Exception as e:
        raise AttributeError(f"Configuration file does not include parameters. {e}")

    def calculate_rotation(point: typing.Sequence[float]):
        rotated_point = [0.0, 0.0, 0.0]

        rotated_point[0] = (
            math.cos(rotation_z_rad) * point[0] - math.sin(rotation_z_rad) * point[1]
        )
        rotated_point[1] = (
            math.sin(rotation_z_rad) * point[0] + math.cos(rotation_z_rad) * point[1]
        )
        rotated_point[2] = 1 * point[2]

        return rotated_point

    for trajectory_data in modified_trajectory_collection:
        positions = []
        velocities = []

        for position in trajectory_data.positions:
            rotated_position = calculate_rotation(position)

            positions.append(tuple(rotated_position))

        for velocity in trajectory_data.velocities:
            rotated_velocity = calculate_rotation(velocity)

            velocities.append(tuple(rotated_velocity))

        trajectory_data.positions = positions
        trajectory_data.velocities = velocities

    return modified_trajectory_collection


def change_time_stamps(
    trajectory_collection: TrajectoryCollection, unit: TransformTimeUnits
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

    for trajectory_data in modified_trajectory_collection:
        # transform from nano seconds to seconds
        trajectory_data.time_stamps = [
            i * unit.value for i in trajectory_data.time_stamps
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
