import json
import pathlib

from aimy_target_shooting import export_tools, filtering, transform
from aimy_target_shooting.configuration import get_config_path
from aimy_target_shooting.custom_types import TransformTimeUnits


def preprocessing_dialog():
    path = get_config_path("preprocessing")
    with open(path, "r") as file:
        config = json.load(file)

    # Import-export options
    directory_path = pathlib.Path(
        "/home/adittrich/nextcloud/82_Data_Processed/MN5008_spin_validation"
    )

    file_name = "mn5008_no_spin.hdf5"

    export_dir_path = pathlib.Path("/tmp/")
    export_file_name = "mn5008_mixed_spin"

    # Load raw data
    data_raw = export_tools.import_all_from_hdf5(file_path=directory_path / file_name)
    print(f"Length raw: {len(data_raw)}")

    # Trajectory Preprocessing
    data_real_processed = filtering.remove_short_trajectories(data_raw, config)
    print(f"Length without empty: {len(data_real_processed)}")

    # Transformation
    data_real_processed = transform.change_time_stamps(
        data_real_processed, TransformTimeUnits.Seconds
    )
    data_real_processed = transform.move_origin(data_real_processed, config)
    # data_real_processed = transform.rotate_coordinate_system(data_real_processed)
    data_real_processed = transform.reset_time_stamps(data_real_processed)

    # Filtering
    data_region = filtering.filter_samples_outside_region(data_real_processed, config)
    data_region = data_real_processed
    print(f"Length outside region: {len(data_region)}")

    data_jumps = filtering.filter_noisy_samples(data_region, config)
    data_jumps = filtering.remove_short_trajectories(data_jumps, config)
    data_jumps = transform.reset_time_stamps(data_jumps)
    print(f"Length without jumps: {len(data_jumps)}")

    data_patchy = filtering.remove_patchy_trajectories(data_jumps, config)
    data_patchy = filtering.remove_short_trajectories(data_patchy, config)
    print(f"Length without patchy: {len(data_patchy)}")

    data_patchy = filtering.filter_samples_after_time_stamp(data_patchy, config)
    print(f"Length after time filter: {len(data_patchy)}")

    # Hitpoint computation
    # detector = hitpoint_detection.HitPointDetection()
    # data_hitpoints = detector.evaluate_hitpoints(data_patchy)
    # data_hitpoints = hitpoint_utils.remove_further_hitpoints(
    #    data_hitpoints, max_hitpoints=1
    # )

    # data_hitpoints = hitpoint_utils.remove_trajectories_without_hitpoints(
    #    data_hitpoints
    # )
    # print(f"Length with hitpoints: {len(data_hitpoints)}")

    # for idx in sorted([4,18,21], reverse=True):
    #    data_patchy.delete_item(idx)

    for i in range(len(data_patchy)):
        x = data_patchy.get_item(i)
        print(f"{i}: {x.launch_param}")

    print(f"Length after delete: {len(data_patchy)}")
    print("Export? [y/n]")
    export_answer = input()

    if export_answer == "y":
        export_tools.export_to_hdf5(
            data_patchy, directory_path=export_dir_path, prefix=export_file_name
        )


if __name__ == "__main__":
    preprocessing_dialog()
