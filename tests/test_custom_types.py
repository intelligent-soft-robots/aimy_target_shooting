import random

import numpy as np
import target_shooting.utils.custom_types


def test_get_item():
    item_0 = target_shooting.utils.custom_types.TrajectoryData()
    item_1 = target_shooting.utils.custom_types.TrajectoryData()
    item_2 = target_shooting.utils.custom_types.TrajectoryData()
    item_3 = target_shooting.utils.custom_types.TrajectoryData()

    rand_list = []
    for _ in range(20):
        rand_list.append(random.randint(0, 100))

    item_0.time_stamps = rand_list

    collection = target_shooting.utils.custom_types.TrajectoryCollection()

    collection.append(item_0)
    collection.append(item_1)
    collection.append(item_2)
    collection.append(item_3)

    # Check whether trajectory collection with items is returned
    assert type(collection.get_item(1)) != type(collection)
    assert type(collection[0:2]) == type(collection)

    # Check whether trajectory data is returned
    item_0_fetched = collection.get_item(0)
    assert type(item_0) == type(item_0_fetched)
    assert item_0 == item_0_fetched

    # Check whether trajectory can be copied
    collection_copy = collection.deepcopy()
    assert collection_copy.get_item(0).time_stamps == collection.get_item(0).time_stamps


def test_slicing_collection():
    collection = target_shooting.utils.custom_types.TrajectoryCollection()

    # Append items
    n_elements = 10
    for _ in range(n_elements):
        new_item = target_shooting.utils.custom_types.TrajectoryData()
        collection.append(new_item)

    assert len(collection) == n_elements
    assert len(collection[0:5]) == 5


def test_appending_samples():
    n_datasets = 5
    n_samples = 200

    collection = target_shooting.utils.custom_types.TrajectoryCollection()

    for _ in range(n_datasets):
        launch_parameter = [0.5, 0.5, 0.3, 0.2, 0.1]
        trajectory_data = target_shooting.utils.custom_types.TrajectoryData()

        trajectory_data.set_launch_param(launch_parameter)

        # append iteratively samples
        for i in range(n_samples):
            ball_id = i
            time_stamp = i * 0.001
            position = [np.random.random_sample() for _ in range(3)]
            velocity = [np.random.random_sample() for _ in range(3)]

            trajectory_data.append_sample(ball_id, time_stamp, position, velocity)

        collection.append(trajectory_data)
