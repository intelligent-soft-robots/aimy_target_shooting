import logging
import os
import pathlib


def get_config_path(config_name: str) -> pathlib.Path:
    """Returns configuration path. Configurations are stored along
    source code in the directory of the repository.

    Args:
        config (str): Config specific specifier.

    Returns:
        pathlib.Path: Location of configuration file.
    """
    parent_dir = pathlib.Path(__file__).parents[1]

    filename = config_name + ".json"
    path = os.path.join(parent_dir, "config", filename)

    return pathlib.Path(path)


def get_default_path() -> pathlib.Path:
    """Returns default path for storing data. Usually data is
    stored on the Desktop.

    Returns:
        pathlib.Path: Storing location.
    """
    path = os.path.join(
        os.path.join(os.path.expanduser("~")), "Desktop", "target_shooting"
    )
    logging.info(f"Default path: {path}")

    return pathlib.Path(path)


def get_temp_path() -> pathlib.Path:
    """Returns temporary path. The temporary directory is
    cleaned after each reboot.

    Returns:
        pathlib.Path: Path to temporary directory.
    """
    return pathlib.Path(os.path.join("tmp", "target_shooting"))
