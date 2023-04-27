import logging
import tkinter as tk
import tkinter.font as tkFont
import typing

from aimy_target_shooting.recording import Recording


class ControlPanel:
    """App provides a simple graphical user interface for easier recording of ball
    trajectories.
    """

    def __init__(
        self,
        root: tk.Tk,
        config: dict,
        demo_mode: bool = False,
        verbose: bool = True,
        width: int = 403,
        height: int = 470,
    ) -> None:
        """Initialises graphical user interace with tkinter interface objects
        as well as an instance of the recording environment.

        Args:
            root (tk.Tk): Tk toplevel widget.
            width (int, optional): Fixed width of GUI window. Defaults to 403.
            height (int, optional): Fixed height of GUI window. Defaults to 470.
            demo_mode (bool, optional): Demo mode omits connection to physical system.
            Demo mode can be used for development purposes. Defaults to False.
            verbose (bool, optional): Specifies if logging messages should be prompted
            in terminal. Defaults to True.
        """
        self.demo_mode = demo_mode

        if verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s %(levelname)-6s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # setting title
        root.title("Ball Launcher Control Panel")
        # setting window size
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = "%dx%d+%d+%d" % (
            width,
            height,
            (screenwidth - width) / 2,
            (screenheight - height) / 2,
        )

        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        ft_btn_large = tkFont.Font(family="Times", size=16)
        ft_btn_normal = tkFont.Font(family="Times", size=10)
        ft_btn_radio = tkFont.Font(family="Times", size=10)
        ft_entry = tkFont.Font(family="Times", size=10)
        ft_label = tkFont.Font(family="Times", size=10)

        def _get_button(
            self,
            root: tk.Tk,
            command: str,
            bg: str = "#990000",
            cursor: str = "arrow",
            font: tkFont.Font = ft_btn_normal,
            fg: str = "#FFFFFF",
            justify: str = "center",
        ):
            btn = tk.Button(root)
            for config in ("bg", "cursor", "font", "fg", "justify"):
                btn[config] = locals()[config]
            btn["command"] = getattr(self, f"_btn_{command}")
            return btn

        commands = ("launch", "test_launch", "save_trajectory", "save_params")
        buttons = {command: _get_button(self, root, command) for command in commands}

        self.label_phi = tk.StringVar()
        self.label_theta = tk.StringVar()
        self.label_omega_1 = tk.StringVar()
        self.label_omega_2 = tk.StringVar()
        self.label_omega_3 = tk.StringVar()

        self.radioValue = tk.StringVar()
        self.radioValue.set("hdf5")

        buttons["launch"]["font"] = ft_btn_large
        buttons["launch"]["text"] = "Record and \nLaunch"
        buttons["launch"].place(x=20, y=360, width=200, height=90)
        buttons["launch"]["command"] = self._btn_launch

        buttons["test_launch"]["bg"] = "#c71585"
        buttons["test_launch"]["font"] = ft_btn_large
        buttons["test_launch"]["text"] = "Test Launch"
        buttons["test_launch"].place(x=230, y=360, width=150, height=90)
        buttons["test_launch"]["command"] = self._btn_test_launch

        buttons["save_trajectory"]["bg"] = "#01aaed"
        buttons["save_trajectory"]["text"] = "Save Trajectories"
        buttons["save_trajectory"].place(x=20, y=270, width=200, height=80)
        buttons["save_trajectory"]["command"] = self._btn_save_trajectory

        buttons["save_params"]["fg"] = "#000000"
        buttons["save_params"]["bg"] = "#ffb800"
        buttons["save_params"]["text"] = "Set Parameters"
        buttons["save_params"].place(x=240, y=90, width=140, height=160)
        buttons["save_params"]["command"] = self._btn_save_params

        self.btn_radio_csv = tk.Radiobutton(root)
        self.btn_radio_csv["font"] = ft_btn_normal
        self.btn_radio_csv["fg"] = "#333333"
        self.btn_radio_csv["justify"] = "left"
        self.btn_radio_csv["text"] = " CSV"
        self.btn_radio_csv.place(x=240, y=270, width=140, height=40)
        self.btn_radio_csv["value"] = "csv"
        self.btn_radio_csv["variable"] = self.radioValue
        self.btn_radio_csv["command"] = self._btn_radio_csv

        self.btn_radio_hdf5 = tk.Radiobutton(root)
        self.btn_radio_hdf5["font"] = ft_btn_radio
        self.btn_radio_hdf5["fg"] = "#333333"
        self.btn_radio_hdf5["justify"] = "left"
        self.btn_radio_hdf5["text"] = "HDF5"
        self.btn_radio_hdf5.place(x=240, y=310, width=140, height=40)
        self.btn_radio_hdf5["value"] = "hdf5"
        self.btn_radio_hdf5["variable"] = self.radioValue
        self.btn_radio_hdf5["command"] = self._btn_radio_hdf5

        entries_keys = ("omega_tl", "omega_tr", "omega_b", "phi", "theta")
        self.entries = {entry: tk.Entry(root) for entry in entries_keys}
        for entry_key, entry_value in zip(self.entries.keys(), self.entries.values()):
            entry_value["font"] = ft_entry
            entry_value["borderwidth"] = "1px"
            entry_value["fg"] = "#333333"
            entry_value["justify"] = "center"
            entry_value["text"] = entry_key

        self.entries["omega_tr"].place(x=20, y=120, width=70, height=25)
        self.entries["omega_b"].place(x=20, y=150, width=70, height=25)
        self.entries["omega_tl"].place(x=20, y=90, width=70, height=25)
        self.entries["phi"].place(x=20, y=200, width=70, height=25)
        self.entries["theta"].place(x=20, y=230, width=70, height=25)

        glabel_keys: typing.Tuple[int, ...] = tuple([i for i in range(1, 16)])
        glabels = {key: tk.Label(root) for key in glabel_keys}
        for glabel in glabels.values():
            glabel["font"] = ft_label
            glabel["fg"] = "#333333"
            glabel["justify"] = "left"

        glabels[1]["text"] = "omega top left"
        glabels[1].place(x=100, y=90, width=110, height=25)

        glabels[2]["text"] = "omega top right"
        glabels[2].place(x=100, y=120, width=110, height=25)

        glabels[3]["text"] = "omega bottom"
        glabels[3].place(x=100, y=150, width=110, height=25)

        glabels[4]["text"] = "phi"
        glabels[4].place(x=100, y=200, width=100, height=25)

        glabels[5]["text"] = "theta"
        glabels[5].place(x=100, y=230, width=100, height=25)

        glabels[6]["text"] = "Omega B"
        glabels[6].place(x=150, y=20, width=70, height=25)

        glabels[7]["text"] = "Omega TR"
        glabels[7].place(x=80, y=20, width=70, height=25)

        glabels[8]["text"] = "Omega TL"
        glabels[8].place(x=10, y=20, width=70, height=25)

        glabels[9]["text"] = "Phi"
        glabels[9].place(x=240, y=20, width=70, height=25)

        glabels[10]["text"] = "Theta"
        glabels[10].place(x=310, y=20, width=70, height=25)

        glabels[11]["bg"] = "#ffffff"
        glabels[11]["textvariable"] = self.label_omega_1
        glabels[11].place(x=10, y=40, width=70, height=25)

        glabels[12]["bg"] = "#ffffff"
        glabels[12]["textvariable"] = self.label_omega_2
        glabels[12].place(x=80, y=40, width=70, height=25)

        glabels[13]["bg"] = "#ffffff"
        glabels[13]["textvariable"] = self.label_omega_3
        glabels[13].place(x=150, y=40, width=70, height=25)

        glabels[14]["bg"] = "#ffffff"
        glabels[14]["textvariable"] = self.label_phi
        glabels[14].place(x=240, y=40, width=70, height=25)

        glabels[15]["bg"] = "#ffffff"
        glabels[15]["textvariable"] = self.label_theta
        glabels[15].place(x=310, y=40, width=70, height=25)

        # Ball launcher and record environment
        if not self.demo_mode:
            self.record_env = Recording(config)

    def _btn_launch(self) -> None:
        """Actuation launches ball and stores trajectory in instance
        of data manger within the recording environment.
        """
        if not self.demo_mode:
            self.record_env.record_and_launch()

        logging.info("Launched.")

    def _btn_save_trajectory(self) -> None:
        """Actuation exports stored trajectories to export format
        specified via radio buttons in GUI.
        """
        export_format = self.radioValue.get()

        if not self.demo_mode:
            self.record_env.export_recordings(export_format=export_format)

        logging.info("Trajectories saved and memory cleared.")

    def _btn_radio_csv(self) -> None:
        """Sets export format of saving button to CSV."""
        logging.info(f"Export format set to: {self.radioValue.get()}")

    def _btn_radio_hdf5(self) -> None:
        """Sets export format of saving button to HDF5."""
        logging.info(f"Export format set to: {self.radioValue.get()}")

    def _btn_save_params(self) -> None:
        """Sets parameters of ball launcher specified in the entry fields
        of GUI.
        """
        omega_1 = float(self.entries["omega_tl"].get())
        omega_2 = float(self.entries["omega_tr"].get())
        omega_3 = float(self.entries["omega_b"].get())

        phi = float(self.entries["phi"].get())
        theta = float(self.entries["theta"].get())

        self.label_omega_1.set(omega_1)
        self.label_omega_2.set(omega_2)
        self.label_omega_3.set(omega_3)

        self.label_phi.set(phi)
        self.label_theta.set(theta)

        launch_params = (phi, theta, omega_1, omega_2, omega_3)

        if not self.demo_mode:
            self.record_env.set_launch_parameters(launch_params)

        logging.info("Launch parameter set.")

    def _btn_test_launch(self) -> None:
        """Actuation launches a test ball which will not be
        recorded.
        """
        if not self.demo_mode:
            self.record_env.launcher.launch()

        logging.info("Test launch fired.")

    def __exit__(self):
        pass


def run_control_panel(
    config: dict,
    demo_mode: bool = True,
    verbose: bool = True,
):
    """Runs control panel.

    Args:
        launcher_ip (typing.Optional[str], optional): IP address of launcher.
        Defaults to None.
        demo_mode (bool, optional): Demo mode if launcher is not available.
        Defaults to True.
        verbose (bool, optional): Print all messages in console.
        Defaults to True.
    """
    root = tk.Tk()
    ControlPanel(root, demo_mode=demo_mode, config=config, verbose=verbose)
    root.mainloop()
