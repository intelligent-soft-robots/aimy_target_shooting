from aimy_target_shooting.control_panel import run_control_panel


def run():
    ip = "10.42.26.171"
    demo_mode = False
    verbose = True

    run_control_panel(launcher_ip=ip, demo_mode=demo_mode, verbose=verbose)


if __name__ == "__main__":
    run()
