import logging
from typing import Sequence

import numpy as np

from aimy_target_shooting.custom_types import TrajectoryCollection


def remove_further_hitpoints(
    trajectory_collection: TrajectoryCollection, max_hitpoints: int = 1
) -> TrajectoryCollection:
    """Removes all hitpoints of each single trajectory exceeding specified
    maximum number of hitpoints.

    Args:
        trajectory_collection (TrajectoryCollection): Stored trajectories
        with hit points attached.
        max_hitpoints (int, optional): Threshold value of maximum permitted
        hit points. All hit points exceeding this number are removed.
        Defaults to 1.

    Returns:
        TrajectoryCollection: Trajectory collection with modified hitpoints.
    """
    trajectory_collection = trajectory_collection.deepcopy()

    for trajectory_data in trajectory_collection:
        if trajectory_data["hitpoints"]:
            if max_hitpoints < len(trajectory_data["hitpoints"]):
                trajectory_data["hitpoints"] = [
                    trajectory_data["hitpoints"][i] for i in range(max_hitpoints)
                ]
                trajectory_data["hitpoint_time_stamps"] = [
                    trajectory_data["hitpoint_time_stamps"][i]
                    for i in range(max_hitpoints)
                ]
            else:
                logging.debug(
                    "Number of hitpoints is equal or lower than specified "
                    "maximum hitpoints."
                )

    return trajectory_collection


def cherry_pick_hitpoint(
    trajectory_collection: TrajectoryCollection, index_hitpoint: int = 1
) -> TrajectoryCollection:
    """Utility function for removing all detected hitpoints not specified by
    index_hitpoint from TrajectoryCollection.

    Args:
        trajectory_collection (TrajectoryCollection): Trajectory collection
        with hit points attached
        index_hitpoint (int, optional): Index of desired hit point.
        For example, only keep the first hit point. Defaults to 1.

    Raises:
        IndexError: Raised if index exceeds number of hit points of one
        trajectory.

    Returns:
        TrajectoryCollection: Collection with cherry picked hit points.
    """
    trajectory_collection = trajectory_collection.deepcopy()

    for trajectory_data in trajectory_collection:
        try:
            trajectory_data["hitpoints"] = [
                trajectory_data["hitpoints"][index_hitpoint]
            ]
        except IndexError:
            raise IndexError(f"Index {index_hitpoint} not existing")

    return trajectory_collection


def average_close_hitpoints(
    trajectory_collection: TrajectoryCollection,
    tolerated_euclidian_distance: float = 0.1,
) -> TrajectoryCollection:
    """Averages hitpoints which are smaller than the tolerated Euclidian
    distance between each point. Close hitpoints are only averaged if they
    occur sequentially within one trajectory.

    Args:
        trajectory_collection (list): Ball data which is extended by detected
        hitpoints.
        tolerated_euclidian_distance (float, optional): Two adjacent points
        which are smaller than the tolerated distance are merged by averaging.
        Defaults to 0.1.

    Returns:
        TrajectoryCollection: Ball data with averaged hitpoints of each
        trajectory.
    """

    def euclidian_distance(point_1: Sequence[float], point_2: Sequence[float]):
        euc_distance: float = np.linalg.norm(np.array(point_2) - np.array(point_1))
        return euc_distance

    def average(point_1: Sequence[float], point_2: Sequence[float]):
        return np.average([point_1, point_2], axis=0)

    trajectory_collection = trajectory_collection.deepcopy()

    for sample in trajectory_collection:
        hitpoints = sample["hitpoints"]

        averaged_hitpoints = []

        x = 0
        while x < len(hitpoints) - 1:
            point_prev = hitpoints[x]
            point_succ = hitpoints[x + 1]

            distance = euclidian_distance(point_prev, point_succ)

            if distance < tolerated_euclidian_distance:
                averaged_point = average(point_prev, point_succ)
                averaged_hitpoints.append(averaged_point)
                x += 1
            else:
                averaged_hitpoints.append(point_prev)

                if x == (len(hitpoints) - 2):
                    averaged_hitpoints.append(point_succ)

            x += 1

        sample["hitpoints"] = averaged_hitpoints

    return trajectory_collection


def remove_trajectories_without_hitpoints(
    trajectory_collection: TrajectoryCollection,
) -> TrajectoryCollection:
    """Removes all trajectores from given trajectory collection
    with empty hit point storage. For example, trajectories which
    do not have a rebound point.

    Args:
        trajectory_collection (TrajectoryCollection): Collection of
        trajectories with hit point detection executed.

    Raises:
        ValueError: Raised if collection does not contain hit point
        storage.

    Returns:
        TrajectoryCollection: Collection only with trajectories
        having stored hit points.
    """
    trajectory_collection = trajectory_collection.deepcopy()

    for idx in reversed(range(len(trajectory_collection))):
        trajectory_data = trajectory_collection.get_item(idx)
        if not hasattr(trajectory_data, "hitpoints"):
            raise ValueError("Trajectory collection does not have hitpoints attached!")

        if not trajectory_data.hitpoints:
            trajectory_collection.delete_item(idx)

    return trajectory_collection


def sort_hitpoints_by_occurrance(
    trajectory_collection: TrajectoryCollection,
) -> list:
    """Generates hit point array of given trajectory collection
    by occurance of hit points. Since trajectories can have multiple
    hit points, this function sorts together hit points ordered by
    their occurance, e.g. all first hit points of each trajectories,
    all second hit points and so on.

    This function can be used for evaluation of similar hitpoints.

    Args:
        trajectory_collection (TrajectoryCollection): Collection
        of trajectories with hit points detected.

    Returns:
        list: List if sorted hitpoints extracted from collection.
    """
    trajectory_collection = trajectory_collection.deepcopy()
    ordered_hitpoints = []

    for trajectory_data in trajectory_collection:
        hitpoints = trajectory_data["hitpoints"]

        for index, hitpoint in enumerate(hitpoints):
            if len(ordered_hitpoints) < index + 1:
                ordered_hitpoints.append([])

            ordered_hitpoints[index].append(hitpoint)

    return ordered_hitpoints


def mean(trajectory_collection: TrajectoryCollection):
    """Calculates mean of appended hitpoints in extended data.

    Args:
        trajectory_collection (list): Ball data extended by hitpoints.

    Returns:
        tuple: Tuple of mean values.
    """
    ordered_hitpoints = sort_hitpoints_by_occurrance(trajectory_collection)

    mean_values = []
    for hitpoints in ordered_hitpoints:
        mean_values.append(tuple(np.mean(hitpoints, axis=0)))

    return mean_values


def std(trajectory_collection: TrajectoryCollection):
    """Calculates deviation of appended hitpoints in extended data.

    Args:
        trajectory_collection (list): Ball data extended by hitpoints.

    Returns:
        tuple: Tuple of deviation values.
    """
    ordered_hitpoints = sort_hitpoints_by_occurrance(trajectory_collection)

    std_values = []
    for hitpoints in ordered_hitpoints:
        std_values.append(tuple(np.std(hitpoints, axis=0)))

    return std_values


def count_trajectories_without_hitpoints(
    trajectory_collection: TrajectoryCollection,
) -> int:
    """Returns number of trajectories without hit points.
    Function can be used for assessing recording quality of
    trajectory collection.

    Args:
        trajectory_collection (TrajectoryCollection): Collection
        of trajectories with hit points detected.

    Returns:
        int: Number of trajectories without hit points in given
        collection.
    """
    count = 0
    for trajectory_data in trajectory_collection:
        hitpoints = trajectory_data["hitpoints"]

        if not hitpoints:
            count += 1

    return count
