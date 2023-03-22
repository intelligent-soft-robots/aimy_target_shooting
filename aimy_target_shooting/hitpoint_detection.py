import json
from typing import Optional

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, make_interp_spline
from scipy.signal import find_peaks

from aimy_target_shooting.configuration import get_config_path
from aimy_target_shooting.custom_types import TrajectoryCollection, TrajectoryData


class HitPointDetection:
    """Hit point detection finds the rebound points of a given trajectory
    by approximating polynomials around the lowest peaks of the trajectory.
    """

    def __init__(
        self,
        axis: Optional[int] = None,
        offset: Optional[float] = None,
        distance: Optional[int] = None,
        position_threshold: Optional[float] = None,
        acceleration_threshold: Optional[float] = None,
        polynom_order: Optional[float] = None,
        use_windows: Optional[bool] = None,
        window_size: Optional[int] = None,
        root_eval_radius: Optional[int] = None,
        position_based_discontinuities: Optional[bool] = None,
        rebound_regression: Optional[bool] = None,
    ) -> None:
        """Initialises hit point detector with given attributes.

        Args:
            axis (Optional[int], optional): Hit point axis encoded as int (e.g. 1 for
            x-axis). Defaults to None.
            offset (Optional[float], optional): Offset of hit point axis to origin.
            Defaults to None.
            distance (Optional[int], optional): Number of neighboring samples to peak
            not to be considered as peak.
            Defaults to None.
            position_threshold (Optional[float], optional): Threshold for detecting
            peaks. Defaults to None.
            acceleration_threshold (Optional[float], optional): Threshold for
            detecting peaks via acceleration jumps. Defaults to None.
            polynom_order (Optional[float], optional): Approximation order for
            intersecting polynomials. Defaults to None.
            use_windows (Optional[bool], optional): Only consider intersections of
            polynomials within certain window. Defaults to None.
            window_size (Optional[int], optional): Window size for detecting peaks.
            Defaults to None.
            root_eval_radius (Optional[int], optional): Size of evaluation region of
            intersecting polynomials. Smaller region prevents accidentially found roots.
            Defaults to None.
            position_based_discontinuities (Optional[bool], optional): Specifier if
            discontiunity points should be found by position based peak or
            acceleration. Defaults to None.
            rebound_regression (Optional[bool], optional): Specifier, if hit point
            should be approximated with polynomials or discontinuity point should
            be used. Defaults to None.

        Raises:
            ValueError: Raised if axis is not 0, 1 or 2.
            ValueError: Raised if polynomial order is larger than window size.
        """
        self.TABLE_HEIGHT = 0.76

        path = get_config_path("hitpoint")
        with open(path, "r") as file:
            config = json.load(file)

        if axis is None:
            self.axis = config["axis"]

            if self.axis not in [0, 1, 2]:
                raise ValueError(f"Axis {self.axis} must be 0, 1, 2!")
        else:
            self.axis = axis

        if offset is None:
            self.offset = config["offset"]
        else:
            self.offset = offset

        if position_threshold is None:
            self.position_threshold = config["position_threshold"]
        else:
            self.position_threshold = position_threshold

        if acceleration_threshold is None:
            self.acceleration_threshold = config["acceleration_threshold"]
        else:
            self.acceleration_threshold = acceleration_threshold

        if distance is None:
            self.distance = config["distance"]
        else:
            self.distance = distance

        if polynom_order is None:
            self.polynom_order = config["polynom_order"]
        else:
            self.polynom_order = polynom_order

        if use_windows is None:
            self.use_windows = config["use_windows"]
        else:
            self.use_windows = use_windows

        if window_size is None:
            self.window_size = config["window_size"]
        else:
            self.window_size = window_size

        if self.window_size < self.polynom_order:
            raise ValueError(
                "Window size has to be equally or larger then polynom order."
            )

        if root_eval_radius is None:
            self.root_eval_radius = config["root_eval_radius"]
        else:
            self.root_eval_radius = root_eval_radius

        if position_based_discontinuities is None:
            self.position_based_indices = config["position_based_indices"]
        else:
            self.position_based_indices = position_based_discontinuities

        if rebound_regression is None:
            self.regression = config["regression"]
        else:
            self.regression = rebound_regression

    def evaluate_hitpoints(
        self, trajectory_collection: TrajectoryCollection
    ) -> TrajectoryCollection:
        """Evaluates the hitpoints of given trajectory collection with
        settings specified in hit point detection class.

        Args:
            trajectory_collection (TrajectoryCollection): Collection
            with stored trajectories.

        Returns:
            TrajectoryCollection: Collection with trajectories extended
            by hit points according to specification.
        """
        trajectory_collection = TrajectoryCollection(trajectory_collection)

        for trajectory_data in trajectory_collection:
            self.trajectory_data = trajectory_data
            self.trajectory_data["axis"] = self.axis

            if self.axis == 0 or self.axis == 1:
                self._find_crossings()
            elif self.axis == 2:
                if self.position_based_indices:
                    self._extract_discontinuities_by_position()
                else:
                    self._extract_discontinuities_by_acceleration()

                self._check_approximation_windows()

                if self.regression:
                    self._find_rebounds_by_regression()
                else:
                    self._find_rebounds_by_position()

        return trajectory_collection

    def evaluate_hitpoints_from_trajectory(
        self, trajectory_data: TrajectoryData
    ) -> TrajectoryData:
        self.trajectory_data = trajectory_data
        self.trajectory_data["axis"] = self.axis

        if self.axis == 0 or self.axis == 1:
            self._find_crossings()
        elif self.axis == 2:
            if self.position_based_indices:
                self._extract_discontinuities_by_position()
            else:
                self._extract_discontinuities_by_acceleration()

            self._check_approximation_windows()

            if self.regression:
                self._find_rebounds_by_regression()
            else:
                self._find_rebounds_by_position()

        return trajectory_data

    def _find_crossings(self) -> None:
        """Find crossing of trajectory with hyperplane specified
        by offset and axis. This function is used, if not a rebound
        point is searched but crossings with any virtual plane.
        """
        time_stamps = np.array(self.trajectory_data["time_stamps"])
        positions = np.array(self.trajectory_data["positions"])

        positions_offset = positions[:, self.axis] - self.offset
        univariate_spline = InterpolatedUnivariateSpline(time_stamps, positions_offset)

        crossing_times = list(univariate_spline.roots())

        spline = make_interp_spline(time_stamps, positions, axis=0)

        hitpoint_time_stamps = []
        hitpoints = []

        for t in crossing_times:
            hitpoints.append(spline(t))
            hitpoint_time_stamps.append(t)

        self.trajectory_data["offset"] = self.offset
        self.trajectory_data["hitpoint_time_stamps"] = hitpoint_time_stamps
        self.trajectory_data["hitpoints"] = hitpoints

    def _extract_discontinuities_by_position(self) -> None:
        """Extracts discontinuities on basis of position peaks. This is
        used to find first candidates for rebound points.
        """
        positions = np.array(self.trajectory_data["positions"])
        positions_axis = positions[:, 2]

        positions_axis -= self.TABLE_HEIGHT
        positions_axis *= -1

        discontinuity_indices = find_peaks(
            positions_axis,
            height=self.position_threshold,
            distance=self.distance,
        )[0]

        self.trajectory_data["distance"] = self.distance
        self.trajectory_data["position_threshold"] = self.position_threshold
        self.trajectory_data["discontinuity_indices"] = discontinuity_indices

    def _extract_discontinuities_by_acceleration(self) -> None:
        """Extracts discontuinities on basis of acceleration jumps.
        This is used to find first candidates for rebound points.
        """
        time_stamps = np.array(self.trajectory_data["time_stamps"])
        positions = np.array(self.trajectory_data["positions"])

        velocities = np.gradient(positions[:, self.axis], time_stamps)
        accelerations = np.gradient(velocities, time_stamps)

        discontinuity_indices = find_peaks(
            accelerations,
            height=self.acceleration_threshold,
            distance=self.distance,
        )[0]

        self.trajectory_data["distance"] = self.distance
        self.trajectory_data["acceleration_threshold"] = self.acceleration_threshold
        self.trajectory_data["discontinuity_indices"] = list(discontinuity_indices)

    def _check_approximation_windows(self):
        """Checks if rebound points have been detected, otherwise does not
        generate approximation indicies.
        """
        discontinuity_indices = self.trajectory_data["discontinuity_indices"]

        if len(discontinuity_indices) == 0:
            self.trajectory_data["approximation_indices"] = []
        else:
            self._get_approximation_windows()

    def _get_approximation_windows(self):
        """Generates approximation windows and checks if enough samples
        are available to generate polynomial. There can be not enough
        samples, e.g. when the rebound happens at the very end of recorded
        trajectory, or rebounds happen high frequently directly after each other.
        """
        time_stamps = self.trajectory_data["time_stamps"]
        discontinuity_indices = self.trajectory_data["discontinuity_indices"]

        if not self.use_windows:
            window_size = self.polynom_order
        else:
            window_size = self.window_size

        # Check if enough samples between each rebound
        for index in reversed(range(1, len(discontinuity_indices) - 1)):
            if (
                discontinuity_indices[index] - discontinuity_indices[index - 1]
                < window_size
            ):
                discontinuity_indices = np.delete(discontinuity_indices, index)

        # Check if enough samples before first rebound
        if discontinuity_indices[0] < window_size:
            discontinuity_indices = discontinuity_indices[1:]

        # Check if enough samples after last rebound
        if discontinuity_indices[-1] > len(time_stamps) - window_size:
            discontinuity_indices = discontinuity_indices[:-1]

        # Generate approximation indices
        approximation_indices = []

        if not self.use_windows:
            if len(discontinuity_indices) == 1:
                lower_index = 0
                upper_index = len(time_stamps) - 1

                approximation_indices.append(
                    (lower_index, discontinuity_indices[0], upper_index)
                )

            for index in range(len(discontinuity_indices) - 1):
                if index == 0:
                    lower_index = 0
                else:
                    lower_index = discontinuity_indices[index - 1]

                if index == discontinuity_indices[-1]:
                    upper_index = len(time_stamps) - 1
                else:
                    upper_index = discontinuity_indices[index + 1]

                approximation_indices.append(
                    (lower_index, discontinuity_indices[index], upper_index)
                )

        if self.use_windows:
            for index in discontinuity_indices:
                lower_index = index - self.window_size
                upper_index = index + self.window_size

                approximation_indices.append((lower_index, index, upper_index))

        self.trajectory_data["approximation_indices"] = approximation_indices

    def _find_rebounds_by_position(self) -> None:
        """Finds rebounds on basis of position peaks."""
        discontinuity_indices = self.trajectory_data["discontinuity_indices"]
        time_stamps = self.trajectory_data["time_stamps"]
        positions = self.trajectory_data["positions"]

        hitpoint_time_stamps = []
        hitpoints = []

        for idx in discontinuity_indices:
            hitpoint_time_stamps.append(time_stamps[idx])
            hitpoints.append(positions[idx])

        self.trajectory_data["hitpoint_time_stamps"] = hitpoint_time_stamps
        self.trajectory_data["hitpoints"] = hitpoints

    def _find_rebounds_by_regression(self) -> None:
        """Finds rebounds on basis of acceleration jumps."""
        time_stamps = np.array(self.trajectory_data["time_stamps"])
        positions = np.array(self.trajectory_data["positions"])
        approximation_indices = self.trajectory_data["approximation_indices"]

        hitpoint_time_stamps = []
        hitpoints = []
        poly_params = []

        for indices in approximation_indices:
            lower_limit, rebound_index, upper_limit = indices

            time_stamps_before_rebound = time_stamps[lower_limit : rebound_index + 1]
            time_stamps_after_rebound = time_stamps[rebound_index : upper_limit + 1]
            positions_before_rebound = positions[lower_limit : rebound_index + 1, :]
            positions_after_rebound = positions[rebound_index : upper_limit + 1, :]

            coef_before_rebound = np.polyfit(
                time_stamps_before_rebound,
                positions_before_rebound[:, 2],
                deg=self.polynom_order,
            )

            coef_after_rebound = np.polyfit(
                time_stamps_after_rebound,
                positions_after_rebound[:, 2],
                deg=self.polynom_order,
            )

            poly_params.append([coef_before_rebound, coef_after_rebound])

            coef_intersections = coef_after_rebound - coef_before_rebound
            t_intersections = list(np.real(np.roots(coef_intersections)))

            if t_intersections:
                for t in t_intersections:
                    hitpoint = [0.0, 0.0, 0.0]

                    if time_stamps[rebound_index] + self.root_eval_radius < t:
                        continue

                    if t < time_stamps[rebound_index] - self.root_eval_radius:
                        continue

                    for i in range(3):
                        p = np.polyfit(
                            time_stamps_before_rebound,
                            positions_before_rebound[:, i],
                            deg=self.polynom_order,
                        )

                        hitpoint[i] = np.polyval(p, t)

                    hitpoint_time_stamps.append(t)
                    hitpoints.append(tuple(hitpoint))

        self.trajectory_data["poly_params"] = poly_params
        self.trajectory_data["hitpoint_time_stamps"] = hitpoint_time_stamps
        self.trajectory_data["hitpoints"] = hitpoints
