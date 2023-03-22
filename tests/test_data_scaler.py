import pathlib
import random

import numpy as np

from aimy_target_shooting.data_scaler import DataScaler


def test_scaler_export():
    bias = []
    scaling = []

    for _ in range(5):
        bias.append(random.uniform(0.0, 1.0))
        scaling.append(random.uniform(-3.0, 4.0))

    scaler = DataScaler(
        scaling_method="standard", bias_value=bias, scaling_value=scaling
    )
    tmp_path = pathlib.Path("/tmp/test_scaling.json")
    scaler.export_values(tmp_path)

    # Generate new scaler with old scaler parameters from JSON
    new_scaler = DataScaler("standard")
    new_scaler.import_values(tmp_path)

    assert (new_scaler.bias_value == np.array(bias)).all()
    assert (new_scaler.scaling_value == np.array(scaling)).all()


def test_scaler():
    test_data = np.random.randn(230, 4)

    scaler = DataScaler(scaling_method="minmax")
    scaler.set_and_scale(test_data)

    print(scaler.bias_value)
    print(np.min(test_data, axis=0))

    assert (scaler.bias_value == np.min(test_data, axis=0)).all()

    tmp_path = pathlib.Path("/tmp/test_scaling.json")
    scaler.export_values(tmp_path)
