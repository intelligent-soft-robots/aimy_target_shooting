import json

from aimy_target_shooting.configuration import get_config_path
from aimy_target_shooting.control_panel import run_control_panel


def run():
    path = get_config_path("recording")
    with open(path, "r") as file:
        config = json.load(file)

    demo_mode = False
    verbose = True

    run_control_panel(config, demo_mode=demo_mode, verbose=verbose)


if __name__ == "__main__":
    run()
