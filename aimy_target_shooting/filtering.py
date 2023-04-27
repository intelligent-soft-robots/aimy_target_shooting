import logging

import numpy as np
from scipy.signal import butter, find_peaks, savgol_filter, sosfilt

from aimy_target_shooting.custom_types import TrajectoryCollection
from aimy_target_shooting.utility import to_tuple_list


def filter_noisy_samples(
    trajectory_collection: TrajectoryCollection,
    config: dict,
) -> TrajectoryCollection:
    """Filters noisy samples from trajectory collection.

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

    try:
        threshold = config["noisy_samples"]["threshold"]
        window_size = config["noisy_samples"]["window_size"]
    except Exception as e:
        raise AttributeError(f"Configuration does not include parameters. {e}")

    for trajectory_data in modified_trajectory_collection:
        positions = np.array(trajectory_data.positions)

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
            del trajectory_data.time_stamps[idx]
            del trajectory_data.positions[idx]
            del trajectory_data.velocities[idx]

        logging.info(f"{removal_counter} samples removed.")

    return modified_trajectory_collection


def filter_samples_outside_region(
    trajectory_collection: TrajectoryCollection,
    config: dict,
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
    try:
        xlimit = config["outlier_cut"]["xlimit"]
        ylimit = config["outlier_cut"]["ylimit"]
        zlimit = config["outlier_cut"]["zlimit"]
    except Exception as e:
        raise AttributeError(f"Configuration does not include parameters. {e}")

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
    config: dict,
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
    try:
        offset = config["rebound_filter"]["offset"]
        threshold = config["rebound_filter"]["threshold"]
        distance = config["rebound_filter"]["distance"]
    except Exception as e:
        raise AttributeError(f"Configuration does not include parameters. {e}")

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
    trajectory_collection: TrajectoryCollection,
    config: dict,
) -> TrajectoryCollection:

    try:
        max_time_stamp = config["time_stamp_filter"]["max_time_stamp"]
    except Exception as e:
        raise AttributeError(f"Configuration does not include parameters. {e}")

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
    config: dict,
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

    try:
        smoothing_method = config["smoothing_method"]
    except Exception as e:
        raise AttributeError(f"Configuration does not include parameters. {e}")

    smoothed_trajectory_collection = trajectory_collection.deepcopy()

    for trajectory_data in smoothed_trajectory_collection:
        positions = np.array(trajectory_data.positions)
        velocities = np.array(trajectory_data.velocities)

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

        trajectory_data.positions = to_tuple_list(positions)
        trajectory_data.velocities = to_tuple_list(velocities)

    return smoothed_trajectory_collection


def remove_patchy_trajectories(
    trajectory_collection: TrajectoryCollection,
    config: dict,
) -> TrajectoryCollection:
    """Removes trajectories from given trajectory collection if temporal
    difference between subsequent samples exceed a limit value.

    Parameter fetched from JSON:
        max_patch_size (float): maximum allowed gap in seconds.

    Args:
        trajectory_collection (TrajectoryCollection): Trajectory collection
        to be filtered.

    Returns:
        TrajectoryCollection: Filtered trajectory collection.
    """
    modified_trajectory_collection = trajectory_collection.deepcopy()

    try:
        max_patch_size = config["patchy_trajectories"]["max_patch_size"]
    except Exception as e:
        raise AttributeError(f"Configuration does not include parameters. {e}")

    for idx in reversed(range(len(modified_trajectory_collection))):
        trajectory_data = modified_trajectory_collection.get_item(idx)
        time_stamps = np.array(trajectory_data.time_stamps)

        gaps = np.diff(time_stamps)

        if np.max(gaps) > max_patch_size:
            modified_trajectory_collection.delete_item(idx)

    return modified_trajectory_collection


def remove_short_trajectories(
    trajectory_collection: TrajectoryCollection,
    config: dict,
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

    try:
        min_length = config["empty_trajectories"]["min_length"]
    except Exception as e:
        raise AttributeError(f"Configuration does not include parameters. {e}")

    for idx in reversed(range(len(modified_trajectory_collection))):
        trajectory_data = modified_trajectory_collection.get_item(idx)

        if len(trajectory_data.positions) < min_length:
            modified_trajectory_collection.delete_item(idx)

    return modified_trajectory_collection
