import json
import logging
from typing import List, Optional

import numpy as np
from scipy.signal import butter, find_peaks, savgol_filter, sosfilt

from aimy_target_shooting.configuration import get_config_path
from aimy_target_shooting.custom_types import TrajectoryCollection
from aimy_target_shooting.utility import to_tuple_list


def filter_volatile_samples(
    trajectory_collection: TrajectoryCollection,
    threshold: float = None,
    window_size: int = None,
) -> TrajectoryCollection:
    """Filters volatile samples from trajectory collection.

    Args:
        trajectory_collection (TrajectoryCollection): Stored
        trajectories.
        threshold (float, optional): Threshold for sample jumps.
        Defaults to None.
        window_size (int, optional): Number of samples in neighborhood
        used as reference for jump evaluation. Defaults to None.

    Returns:
        TrajectoryCollection: Modified trajectories.
    """
    modified_trajectory_collection = trajectory_collection.deepcopy()

    path = get_config_path("preprocessing")
    with open(path, "r") as file:
        config = json.load(file)

    if threshold is None:
        threshold = config["volatile_samples"]["threshold"]

    if window_size is None:
        window_size = config["volatile_samples"]["window_size"]

    for trajectory_data in modified_trajectory_collection:
        positions = np.array(trajectory_data["positions"])

        removal_counter = 0
        removal_indices = []

        for i in range(len(positions) - window_size + 1):
            average = np.average(positions[i : i + window_size], axis=0)
            delta = np.abs(positions[i] - average)
            if np.any(delta > threshold):
                removal_counter += 1
                removal_indices.append(i)

        if len(positions) > window_size and False:
            # TODO: Filter from both sides.
            start_residual_window = len(positions) - window_size + 2
            end_residual_window = len(positions) - 1

            print(f"Residual window start: {start_residual_window}")
            print(f"Residual window end: {end_residual_window}")

            for i in range(start_residual_window, end_residual_window):
                average = np.average(
                    positions[start_residual_window:end_residual_window], axis=0
                )
                delta = np.abs(positions[i] - average)
                if np.any(delta > threshold):
                    removal_counter += 1
                    removal_indices.append(i)

        for idx in sorted(removal_indices, reverse=True):
            del trajectory_data["time_stamps"][idx]
            del trajectory_data["positions"][idx]
            del trajectory_data["velocities"][idx]

        logging.info(f"{removal_counter} samples removed.")

    return modified_trajectory_collection


def filter_samples_outside_region(
    trajectory_collection: TrajectoryCollection,
    xlimit: List[float] = None,
    ylimit: List[float] = None,
    zlimit: List[float] = None,
) -> TrajectoryCollection:
    """Filters samples outside of specified spatial limits.

    Args:
        trajectory_collection (TrajectoryCollection): Stored
        trajectories.
        xlimit (List[float], optional): Limits in x-direction.
        Defaults to None.
        ylimit (List[float], optional): Limits in y-direction.
        Defaults to None.
        zlimit (List[float], optional): Limits in z-direction.
        Defaults to None.

    Returns:
        TrajectoryCollection: Modified trajectories.
    """
    path = get_config_path("preprocessing")
    with open(path, "r") as file:
        config = json.load(file)

    if xlimit is None:
        xlimit = config["outlier_cut"]["xlimit"]

    if ylimit is None:
        ylimit = config["outlier_cut"]["ylimit"]

    if zlimit is None:
        zlimit = config["outlier_cut"]["zlimit"]

    limits = [xlimit, ylimit, zlimit]

    modified_trajectory_collection = trajectory_collection.deepcopy()

    for trajectory_data in modified_trajectory_collection:
        positions = trajectory_data["positions"]

        delete_indices = []

        for idx in range(len(positions)):
            for axis in range(3):
                if not (limits[axis][0] < positions[idx][axis] < limits[axis][1]):
                    delete_indices.append(idx)
                    break

        for idx in sorted(delete_indices, reverse=True):
            del trajectory_data["time_stamps"][idx]
            del trajectory_data["positions"][idx]
            del trajectory_data["velocities"][idx]

    return modified_trajectory_collection


def filter_samples_after_first_rebound(
    trajectory_collection: TrajectoryCollection,
    offset: Optional[float] = None,
    threshold: Optional[float] = None,
    distance: Optional[int] = None,
) -> TrajectoryCollection:
    """Removes all samples after first rebound of each trajectory stored
    in given trajectory collection. The rebound is defined by the sample
    with lowest z-value.

    Args:
        trajectory_collection (TrajectoryCollection): Stored trajectores.
        offset (Optional[float], optional): Offset to ground. Defaults to None.
        threshold (Optional[float], optional): Height threshold for rebound
        detection. Defaults to None.
        distance (Optional[int], optional): Number of omitted samples in
        neighborhood of peak. Used for rebound detection. Defaults to None.

    Returns:
        TrajectoryCollection: Modified trajectories.
    """
    path = get_config_path("preprocessing")
    with open(path, "r") as file:
        config = json.load(file)

    if offset is None:
        offset = config["rebound_filter"]["offset"]

    if threshold is None:
        threshold = config["rebound_filter"]["threshold"]

    if distance is None:
        distance = config["rebound_filter"]["distance"]

    modified_trajectory_collection = trajectory_collection.deepcopy()

    for trajectory_data in modified_trajectory_collection:
        time_stamps = np.array(trajectory_data.time_stamps)
        positions = np.array(trajectory_data.positions)
        velocities = np.array(trajectory_data.velocities)

        positions_axis = positions[:, 2].copy()

        positions_axis -= offset
        positions_axis *= -1

        discontinuity_indices = find_peaks(
            positions_axis,
            height=threshold,
            distance=distance,
        )[0]

        if discontinuity_indices.any():
            time_stamps = time_stamps[0 : discontinuity_indices[0] + 1]
            positions = positions[0 : discontinuity_indices[0] + 1]

        trajectory_data.time_stamps = time_stamps
        trajectory_data.positions = positions
        trajectory_data.velocities = velocities

    return modified_trajectory_collection


def filter_samples_after_time_stamp(
    trajectory_collection: TrajectoryCollection, max_time_stamp: Optional[float] = None
) -> TrajectoryCollection:
    if max_time_stamp is None:
        path = get_config_path("preprocessing")
        with open(path, "r") as file:
            config = json.load(file)

        max_time_stamp = config["time_stamp_filter"]["max_time_stamp"]

    modified_trajectory_collection = trajectory_collection.deepcopy()

    for trajectory_data in modified_trajectory_collection:
        time_stamps = np.array(trajectory_data.time_stamps)
        positions = np.array(trajectory_data.positions)
        velocities = np.array(trajectory_data.velocities)

        remove_indices = np.nonzero(time_stamps > max_time_stamp)[0]

        logging.info(f"{len(remove_indices)} samples removed.")

        time_stamps = np.delete(time_stamps, remove_indices, axis=0)
        positions = np.delete(positions, remove_indices, axis=0)
        velocities = np.delete(velocities, remove_indices, axis=0)

        trajectory_data.time_stamps = time_stamps
        trajectory_data.positions = positions
        trajectory_data.velocities = velocities

    return modified_trajectory_collection


def smooth_samples(
    trajectory_collection: TrajectoryCollection,
) -> TrajectoryCollection:
    """Smoothes position and velocities samples of given trajectory
    collection with smoothing method specified in JSON configuration
    file.

    Args:
        trajectory_collection (TrajectoryCollection): Stored
        trajectories.

    Returns:
        TrajectoryCollection: Trajectory collection with smoothed
        positions and velocities.
    """
    path = get_config_path("preprocessing")
    with open(path, "r") as file:
        config = json.load(file)

    smoothing_method = config["smoothing_method"]

    smoothed_trajectory_collection = trajectory_collection.deepcopy()

    for trajectory_data in smoothed_trajectory_collection:
        positions = np.array(trajectory_data["positions"])
        velocities = np.array(trajectory_data["velocities"])

        for i in range(3):
            if smoothing_method == "savitzky-golay":
                window_size = config["savitzky-golay"]["window_size"]
                polyorder = config["savitzky-golay"]["polyorder"]

                positions[:, i] = savgol_filter(
                    positions[:, i],
                    window_length=window_size,
                    polyorder=polyorder,
                )

                velocities[:, i] = savgol_filter(
                    velocities[:, i],
                    window_length=window_size,
                    polyorder=polyorder,
                )

            if smoothing_method == "zero-phase-sos":
                order = config["zero-phase-sos"]["order"]
                cutoff_frequency = config["zero-phase-sos"]["cutoff_frequency"]

                sos = butter(order, cutoff_frequency, "lowpass", output="sos")
                velocities[:, i] = sosfilt(sos, velocities[:, i])

        trajectory_data["positions"] = to_tuple_list(positions)
        trajectory_data["velocities"] = to_tuple_list(velocities)

    return smoothed_trajectory_collection


def remove_patchy_trajectories(
    trajectory_collection: TrajectoryCollection, max_patch_size: float = None
) -> TrajectoryCollection:
    """Removes trajectories from given trajectory collection if spacing
    between samples exceed a limit value.

    Parameter fetched from JSON:
        max_patch_size (float): maximum allowed gap in seconds.

    Args:
        trajectory_collection (TrajectoryCollection): Trajectory collection
        to be filtered.

    Returns:
        TrajectoryCollection: Filtered trajectory collection.
    """
    modified_trajectory_collection = trajectory_collection.deepcopy()

    if max_patch_size is None:
        path = get_config_path("preprocessing")
        with open(path, "r") as file:
            config = json.load(file)

        max_patch_size = config["patchy_trajectories"]["max_patch_size"]

    for idx in reversed(range(len(modified_trajectory_collection))):
        trajectory_data = modified_trajectory_collection.get_item(idx)
        time_stamps = np.array(trajectory_data.time_stamps)

        gaps = np.diff(time_stamps)

        if np.max(gaps) > max_patch_size:
            modified_trajectory_collection.delete_item(idx)

    return modified_trajectory_collection


def remove_short_trajectories(
    trajectory_collection: TrajectoryCollection, min_length: Optional[int] = None
) -> TrajectoryCollection:
    """Removes trajectories from trajectory collection with length
    smaller than specified minimum length. For example, empty trajectories.

    Args:
        trajectory_collection (TrajectoryCollection): Trajectories to be
        filtered.
        min_length (Optional[int], optional): Minimum length for trajectory
        to prevent filtering. Defaults to None.

    Returns:
        TrajectoryCollection: Modified collection.
    """
    modified_trajectory_collection = trajectory_collection.deepcopy()

    if min_length is None:
        path = get_config_path("preprocessing")
        with open(path, "r") as file:
            config = json.load(file)

        min_length = config["empty_trajectories"]["min_length"]

    for idx in reversed(range(len(modified_trajectory_collection))):
        trajectory_data = modified_trajectory_collection.get_item(idx)

        if len(trajectory_data["positions"]) < min_length:
            modified_trajectory_collection.delete_item(idx)

    return modified_trajectory_collection
