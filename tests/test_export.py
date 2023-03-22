import datetime
import pathlib

import numpy as np
from target_shooting.utils import export_tools
from target_shooting.utils.custom_types import TrajectoryCollection, TrajectoryData


def generate_test_collection() -> TrajectoryCollection:
    n_datasets: int = 5
    sample_size: int = 200

    collection = TrajectoryCollection()

    for i in range(n_datasets):
        launch_parameters = [1.0, 0.99, 0.4, 0.4, 0.4]

        trajectory_data = TrajectoryData()
        trajectory_data.set_launch_param(launch_parameters)

        for i in range(sample_size):
            ball_id = 1
            time_stamp = (i + 5) * 0.001
            trajectory = tuple([np.random.random_sample() for _ in range(3)])
            velocity = tuple([np.random.random_sample() for _ in range(3)])

            trajectory_data.append_sample(ball_id, time_stamp, trajectory, velocity)

        collection.append(trajectory_data)

    return collection


def test_export_single_dataset_to_csv():
    # Generate test data
    collection = generate_test_collection()

    # Export collection
    now = datetime.datetime.now()
    time_stamp = now.strftime("%Y%m%d%H%M%S")

    dir_name = "/tmp/testing/" + str(time_stamp) + "_single_csv"
    tmp_path = pathlib.Path(dir_name)
    prefix = "test"

    export_tools.export_to_csv(collection, tmp_path, prefix)

    # Import collection
    file_name = prefix + "1.csv"
    file_path = tmp_path / file_name

    data_imported = export_tools.import_from_csv(
        file_path=file_path,
        import_launch_param=True,
        import_velocities=True,
    )

    # Test
    data_reference = collection.get_item(1)

    np.testing.assert_almost_equal(data_reference.positions, data_imported.positions)


def test_export_dataset_to_csv():
    # Generate test data
    collection = generate_test_collection()

    # Export collection
    now = datetime.datetime.now()
    time_stamp = now.strftime("%Y%m%d%H%M%S")

    dir_name = "/tmp/testing/" + str(time_stamp) + "_csv"
    tmp_path = pathlib.Path(dir_name)
    prefix = "test"

    export_tools.export_to_csv(collection, tmp_path, prefix)

    # Import collection
    collection_imported = export_tools.import_all_from_csv(
        directory_path=tmp_path,
        import_launch_param=True,
        import_velocities=True,
    )

    # Test
    data_reference = collection.get_item(1)
    data_imported = collection_imported.get_item(1)

    np.testing.assert_almost_equal(data_reference.positions, data_imported.positions)


def test_export_single_dataset_to_hdf5():
    # Generate test data
    collection = generate_test_collection()

    # Export collection
    now = datetime.datetime.now()
    time_stamp = now.strftime("%Y%m%d%H%M%S")
    dir_name = "/tmp/testing/" + str(time_stamp) + "_single_hdf5"
    tmp_path = pathlib.Path(dir_name)
    hdf5_file_name = "ball_trajectories_" + str(np.random.randint(0, 1000))
    hdf5_file_path = tmp_path / (hdf5_file_name + ".hdf5")

    # Test export and import to hdf5 format
    export_tools.export_to_hdf5(
        trajectory_collection=collection, directory_path=tmp_path, prefix=hdf5_file_name
    )

    data_imported = export_tools.import_from_hdf5(
        group="originals",
        index=0,
        file_path=hdf5_file_path,
        import_launch_param=True,
        import_velocities=True,
    )

    data_reference = collection.get_item(0)

    # Test
    np.testing.assert_almost_equal(data_reference.positions, data_imported.positions)


def test_export_dataset_to_hdf5():
    # Generate test data
    collection = generate_test_collection()

    # Export collection
    now = datetime.datetime.now()
    time_stamp = now.strftime("%Y%m%d%H%M%S")
    dir_name = "/tmp/testing/" + str(time_stamp) + "_hdf5"
    tmp_path = pathlib.Path(dir_name)
    hdf5_file_name = "ball_trajectories_" + str(np.random.randint(0, 1000))
    hdf5_file_path = tmp_path / (hdf5_file_name + ".hdf5")

    # Test export and import to hdf5 format
    export_tools.export_to_hdf5(
        trajectory_collection=collection, directory_path=tmp_path, prefix=hdf5_file_name
    )

    collection_imported = export_tools.import_all_from_hdf5(
        group="originals",
        file_path=hdf5_file_path,
        import_launch_param=True,
        import_velocities=True,
    )

    data_reference = collection.get_item(1)
    data_imported = collection_imported.get_item(1)

    # Test
    np.testing.assert_almost_equal(data_reference.positions, data_imported.positions)
