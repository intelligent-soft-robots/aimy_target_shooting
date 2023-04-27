import logging
import pathlib

import h5py
import pandas as pd

from aimy_target_shooting.configuration import get_default_path
from aimy_target_shooting.custom_types import TrajectoryCollection, TrajectoryData
from aimy_target_shooting.utility import to_tuple_list


def import_from_hdf5(
    index: int,
    group: str = "originals",
    file_path: pathlib.Path = None,
    import_launch_param: bool = True,
    import_velocities: bool = True,
) -> TrajectoryData:
    """Imports ball data from HDF5 file specified in path argument from specified
    group and index. Imported data is stored in TrajectoryData.

    Args:
        index (int): Index of trajectory to be imported.
        group (str): Nested group within HDF5 file. Defaults to "originals".
        file_path (pathlib.Path, optional): Path object specifying file location.
        Defaults to None.
        import_launch_param (bool, optional): Specifier if stored launch parameter
        should be imported. Defaults to True.
        import_velocities (bool, optional): Specifier if stored velocities should
        be imported. Defaults to True.
    """
    trajectory_data = TrajectoryData()

    index_str = str(index)

    if file_path is None:
        file_path = get_default_path() / "target_shooting" / "ball_trajectories.hdf5"

    # "r" specifies only read permissions
    file = h5py.File(file_path, "r")

    if import_launch_param:
        trajectory_data["launch_param"] = tuple(file[group][index_str]["launch_param"])

    trajectory_data["time_stamps"] = list(file[group][index_str]["time_stamps"])
    trajectory_data["positions"] = to_tuple_list(
        list(file[group][index_str]["positions"])
    )

    if import_velocities:
        trajectory_data["velocities"] = to_tuple_list(
            list(file[group][index_str]["velocities"])
        )

    file.close()

    return trajectory_data


def import_all_from_hdf5(
    group: str = "originals",
    file_path: pathlib.Path = None,
    import_launch_param: bool = True,
    import_velocities: bool = True,
) -> TrajectoryCollection:
    """Imports all stored ball data from HDF5 file specified in given path argument
    from desired group. Imported data is stored in TrajectoryCollection.

    Args:
        group (str, optional): Nested group within HDF5 file. Defaults to
        "originals".
        file_path (pathlib.Path, optional): Path object specifying file location.
        Defaults to None.
        import_launch_param (bool, optional): Specifier if stored launch
        parameter should be imported. Defaults to True.
        import_velocities (bool, optional): Specifier if stored velocities
        should be imported. Defaults to True.
    """
    trajectory_collection = TrajectoryCollection()

    if file_path is None:
        file_path = get_default_path() / "target_shooting" / "ball_trajectories.hdf5"

    # "r" specifies only read permissions
    file = h5py.File(file_path, "r")

    for index in list(file[group].keys()):
        trajectory_data = TrajectoryData()

        if import_launch_param:
            trajectory_data["launch_param"] = tuple(file[group][index]["launch_param"])

        trajectory_data["time_stamps"] = list(file[group][index]["time_stamps"])
        trajectory_data["positions"] = to_tuple_list(
            list(file[group][index]["positions"])
        )

        if import_velocities:
            trajectory_data["velocities"] = to_tuple_list(
                list(file[group][index]["velocities"])
            )

        trajectory_collection.append(trajectory_data)

    file.close()

    return trajectory_collection


def export_to_hdf5(
    trajectory_collection: TrajectoryCollection,
    directory_path: pathlib.Path = None,
    prefix: str = "ball_trajectories",
    clear_storage: bool = False,
) -> None:
    """Exports stored ball data in TrajectoryCollection to HDF5 file specified in path
    argument. Name of file can be specified via prefix argument.

    Args:
        trajectory_collection (TrajectoryCollection): List of trajectory data.
        directory_path (pathlib.Path, optional): File location including filename.
        Defaults to None.
        prefix (str, optional): Filename if no path is given. Defaults to
        "ball_trajectories".
        clear_storage (bool, optional): Specifier if ball data should be cleared after
        export. Defaults to False.

    Raises:
        IndexError: Raised if no data is stored in TrajectoryCollection.
    """
    if not trajectory_collection:
        raise IndexError("No trajectory is stored in trajectory collection.")

    if directory_path is None:
        directory_path = get_default_path() / "target_shooting"

    pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
    file_path = directory_path / (prefix + ".hdf5")

    file = h5py.File(file_path, "a")

    if "originals" not in file.keys():
        dataset = file.create_group("originals")
    else:
        dataset = file["originals"]

    if list(dataset.keys()):
        current_iter = max([int(i) for i in dataset.keys()]) + 1
    else:
        current_iter = 0

    for sample in trajectory_collection:
        iteration = dataset.create_group(str(current_iter))

        if "launch_param" in sample:
            iteration.create_dataset("launch_param", data=sample["launch_param"])

        iteration.create_dataset("time_stamps", data=sample["time_stamps"])
        iteration.create_dataset("positions", data=sample["positions"])

        if "velocities" in sample:
            iteration.create_dataset("velocities", data=sample["velocities"])

        current_iter += 1

    if clear_storage:
        trajectory_collection.clear_collection()


def import_from_csv(
    file_path: pathlib.Path,
    import_launch_param: bool = True,
    import_velocities: bool = True,
) -> TrajectoryData:
    """Imports ball data from CSV file specified in path argument.
    Imported data is stored in TrajectoryCollection.

    Args:
        file_path (pathlib.Path): File location
        import_launch_param (bool, optional): Specifier if stored launch
        parameter should be imported. Defaults to True.
        import_velocities (bool, optional): Specifier if stored velocity
        should be imported. Defaults to True.
    """
    df = pd.read_csv(file_path)

    trajectory_data = TrajectoryData()

    # Import launch parameter
    if import_launch_param:
        launch_param = df["launch_param"].tolist()
        launch_param = tuple([v for v in launch_param if str(v) != "nan"])
        trajectory_data["launch_param"] = launch_param

    # Import time stamps
    time_stamps = df["time_stamps"].tolist()
    trajectory_data["time_stamps"] = list(time_stamps)

    # Import trajectory
    x = df["x"].tolist()
    y = df["y"].tolist()
    z = df["z"].tolist()

    trajectory_data["positions"] = list([(i, j, k) for i, j, k in zip(x, y, z)])

    # Import velocities
    if import_velocities:
        v_x = df["vx"].tolist()
        v_y = df["vy"].tolist()
        v_z = df["vz"].tolist()

        trajectory_data["velocities"] = list(
            [(i, j, k) for i, j, k in zip(v_x, v_y, v_z)]
        )

    return trajectory_data


def import_all_from_csv(
    directory_path: pathlib.Path,
    import_launch_param: bool = True,
    import_velocities: bool = True,
) -> TrajectoryCollection:
    """Imports ball data from all CSV files in specified directory path argument.
    Imported data is stored in TrajectoryCollection.

    Args:
        directory_path (str, optional): Directory location. Defaults to None.
        import_launch_param (bool, optional): Specifier, if launch parameters should
        be imported. Defaults to True.
        import_velocities (bool, optional): Specifier, if velocities should be imported.
        Defaults to True.
    """
    trajectory_collection = TrajectoryCollection()

    if directory_path is None:
        directory_path = get_default_path() / "target_shooting"

    file_paths = sorted(pathlib.Path(directory_path).glob("*.csv"))

    for path in file_paths:
        trajectory_data = TrajectoryData()
        trajectory_data = import_from_csv(
            file_path=path,
            import_launch_param=import_launch_param,
            import_velocities=import_velocities,
        )
        trajectory_collection.append(trajectory_data)

    return trajectory_collection


def export_to_csv(
    trajectory_collection: TrajectoryCollection,
    directory_path: pathlib.Path = None,
    prefix: str = "ball_trajectory_",
    clear_storage: bool = False,
) -> None:
    """Exports stored trajectories into a CSV file in specified directory path.
    Each trajectory has its own CSV file. Name of file can be specified via
    prefix argument.

    Args:
        trajectory_collection (TrajectoryCollection): List of trajectory data.
        directory_path (pathlib.Path, optional): Directory location. Defaults to
        None.
        prefix (str, optional): Prefix for each CSV file. Defaults to
        "ball_trajectory_".
        clear_storage (bool, optional): Specifier if ball data should be cleared
        after export. Defaults to False.

    Raises:
        IndexError: Raised if no data is stored in trajectory collection.
    """
    if not trajectory_collection:
        raise IndexError("No trajectory is stored in trajectory collection.")

    if not isinstance(directory_path, pathlib.Path):
        raise TypeError(f"Given path is {type(directory_path)} is not pathlib.Path!")

    if directory_path is None:
        directory_path = get_default_path() / "target_shooting"

    pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(trajectory_collection):
        file_name = prefix + str(index) + ".csv"
        file_path = directory_path / file_name

        df = pd.DataFrame()

        if "launch_param" in sample:
            if len(sample["positions"]) >= 5:
                trajectory_length = len(sample["positions"])
                params_length = len(sample["launch_param"])

                launch_param = list(sample["launch_param"])
                launch_param.extend([""] * (trajectory_length - params_length))

                df["launch_param"] = launch_param
            else:
                logging.warning(f"Exported data {file_name} is empty.")

        df["time_stamps"] = sample["time_stamps"]

        df["x"] = [i[0] for i in sample["positions"]]
        df["y"] = [i[1] for i in sample["positions"]]
        df["z"] = [i[2] for i in sample["positions"]]

        if "velocities" in sample:
            df["vx"] = [i[0] for i in sample["velocities"]]
            df["vy"] = [i[1] for i in sample["velocities"]]
            df["vz"] = [i[2] for i in sample["velocities"]]

        df.to_csv(file_path)

    if clear_storage:
        trajectory_collection.clear_collection()


def export_data(
    trajectory_collection: TrajectoryCollection,
    export_format: str = "hdf5",
    export_path: pathlib.Path = None,
    prefix: str = "ball_trajectories",
    clear_storage: bool = False,
) -> None:
    """Wrapper function for exporting data by specifying export format and prefix.

    Args:
        trajectory_collection (TrajectoryCollection): List of trajectory data.
        export_format (str, optional): Export data format. Supports "hdf5" and "csv".
        Defaults to "hdf5".
        export_path (str, optional): Export path. Defaults to None.
        prefix (str, optional): File name for exported files. Defaults to
        "ball_trajectories".
        clear_storage (bool, optional): Specifier if ball data should be cleared
        after export. Defaults to False.

    Raises:
        IndexError: Raised if no ball data is stored in TrajectoryCollection.
        ValueError: Raised if export data format is not supported.
    """
    if not trajectory_collection:
        raise IndexError("No trajectory is stored in TrajectoryCollection")

    if export_format == "hdf5":
        export_to_hdf5(
            trajectory_collection,
            directory_path=export_path,
            prefix=prefix,
            clear_storage=clear_storage,
        )
    elif export_format == "csv":
        export_to_csv(
            trajectory_collection,
            directory_path=export_path,
            prefix=prefix,
            clear_storage=clear_storage,
        )
    else:
        raise ValueError(f"Given export format {export_format} is not supported.")
