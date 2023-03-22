from __future__ import annotations

import copy
import typing
from typing import Iterator, List, Optional

Position3D = typing.Tuple[float, float, float]
Velocity3D = typing.Tuple[float, float, float]

TimeStamps = typing.List[float]
LaunchParameter = typing.Tuple[float, ...]

PositionTrajectory = typing.List[Position3D]
VelocityTrajectory = typing.List[Velocity3D]


class TrajectoryData:
    """Container for trajectory data default storing launch
    parameter, time stamps, positions and velocities.
    TrajectoryData information can be fetched via keys.
    """

    def __init__(self, trajectory_data: TrajectoryData = None) -> None:
        """Initiates trajectory data. Can be initialised on basis of
        other trajectory data, where the given trajectory data is copied.

        Args:
            trajectory_data (TrajectoryData, optional): Data to be copied
            to new trajectory data. Defaults to None.
        """
        self.start_time: Optional[float] = None

        if trajectory_data:
            self.time_stamps: TimeStamps = trajectory_data.time_stamps.copy()
            self.positions: PositionTrajectory = trajectory_data.positions.copy()
            self.velocities: VelocityTrajectory = trajectory_data.velocities.copy()
            self.launch_param: LaunchParameter = trajectory_data.launch_param
        else:
            self.time_stamps = []
            self.positions = []
            self.velocities = []
            self.launch_param = ()

    def __getitem__(self, key: typing.Any) -> typing.Any:
        """Magic function for dict functionality.

        Args:
            key (typing.Any): Key name.

        Returns:
            _type_: Stored attribute.
        """
        return getattr(self, key)

    def __len__(self) -> int:
        """Magic function for usage of len-function. Returns number
        of stored samples.

        Returns:
            int: Number of stored samples.
        """
        return len(self.positions)

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        """Magic function for setting new key.

        Args:
            key (typing.Any): Key name.
            value (typing.Any): Key value.
        """
        setattr(self, key, value)

    def __contains__(self, key: typing.Any) -> bool:
        """Magic function for dict-functionality.

        Args:
            key (typing.Any): Stored parameter keys.

        Returns:
            bool: True, if parameter key is available.
        """
        if hasattr(self, key):
            return True
        else:
            return False

    def set_launch_param(self, value: LaunchParameter) -> None:
        """Sets launch parameter of trajectory data.

        Args:
            value (LaunchParameter): Launch parameter of trajectory.
        """
        self.launch_param = value

    def reset(self) -> None:
        """Clears all values stored in trajectory data."""
        self.start_time = None

        self.time_stamps = []
        self.positions = []
        self.velocities = []

    def append_sample(
        self,
        ball_id: int,
        time_stamp: float,
        position: Position3D,
        velocity: Velocity3D = None,
    ) -> None:
        """Append single measurement sample of trajectory.

        Args:
            ball_id (int): ID of measurement system.
            time_stamp (float): Time stamp of measurment.
            position (Position3D): Position tuple.
            velocity (Velocity3D, optional): Velocity tuple.
            Defaults to None.
        """
        if ball_id >= 0:
            time_stamp = time_stamp
            if self.start_time is None:
                self.start_time = time_stamp
            time_stamp -= self.start_time

            self.time_stamps.append(time_stamp)
            self.positions.append(position)

            if velocity:
                self.velocities.append(velocity)

    def set_full_trajectory(
        self,
        time_stamps: TimeStamps,
        positions: PositionTrajectory,
        velocities: VelocityTrajectory = None,
        launch_parameters: LaunchParameter = None,
    ) -> None:
        """Add full trajectory data with single function.

        Args:
            time_stamps (TimeStamps): Time stamps of trajectory.
            positions (PositionTrajectory): Positions as tuple trajectory.
            velocities (VelocityTrajectory, optional): Velocities. Defaults to None.
            launch_parameters (LaunchParameter, optional): Launch parameter of ball
            trajectory. Defaults to None.
        """
        self.time_stamps = time_stamps
        self.positions = positions

        if velocities is not None:
            self.velocities = velocities

        if launch_parameters is not None:
            self.set_launch_param(launch_parameters)


class TrajectoryCollection:
    """Collection of trajectory data objects which acts like
    a list.
    """

    def __init__(
        self, trajectory_collection: Optional[TrajectoryCollection] = None
    ) -> None:
        """Generates trajectory collection. Can be initialised with another
        trajectory collection, copying the items to new collection without
        dependency.

        Args:
            trajectory_collection (Optional[TrajectoryCollection], optional):
            Template collection. Defaults to None.

        Raises:
            TypeError: Is thrown, if given is not a trajectory collection.
        """
        if trajectory_collection is None:
            self._collection: List[TrajectoryData] = []
        else:
            if isinstance(trajectory_collection, TrajectoryCollection):
                self._collection = copy.deepcopy(trajectory_collection.get_collection())
            elif isinstance(trajectory_collection, list):
                self._collection: List[TrajectoryData] = copy.deepcopy(
                    trajectory_collection
                )
            else:
                raise TypeError(
                    "List or trajectory_collection expected,"
                    f"{type(trajectory_collection)} received."
                )

    def __len__(self) -> int:
        """Magic function enabling len-functionality.

        Returns:
            int: Number of collection items stored.
        """
        return len(self._collection)

    def __getitem__(self, index: int) -> TrajectoryCollection:
        """Magic function enabling squared brackets functionality.

        Args:
            index (int): Index range.

        Raises:
            TypeError: Raised if input is neither slice nor int.

        Returns:
            TrajectoryCollection: Slice or item of collection.
        """
        if isinstance(index, slice):
            return TrajectoryCollection(self._collection[index.start : index.stop])
        elif isinstance(index, int):
            return TrajectoryCollection([self._collection[index]])
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self) -> Iterator[TrajectoryData]:
        """Magic function for for-list functionality.

        Returns:
            _type_: Iterator.

        Yields:
            Iterator[TrajectoryData]: Iterates to next item in collection list.
        """
        return iter(self._collection)

    def __contains__(self, value: TrajectoryData) -> bool:
        """Magic function used for dict functionality.

        Args:
            value (TrajectoryData): Key name.

        Returns:
            bool: True if key is available.
        """
        if value in self._collection:
            return True
        else:
            return False

    def __setitem__(self, index: int, value: TrajectoryData) -> None:
        self._collection[index] = value

    def append(self, value: TrajectoryData) -> None:
        """Append trajectory data object to trajectory collection.

        Args:
            value (TrajectoryData): trajectory data object.
        """
        assert isinstance(value, TrajectoryData)
        self._collection.append(value)

    def clear_collection(self) -> None:
        """Clears trajectory collection."""
        self._collection = []

    def get_collection(self) -> List[TrajectoryData]:
        """Returns collection items as list.

        Returns:
            List[TrajectoryData]: Trajectory data objects.
        """
        return self._collection

    def get_item(self, index: int) -> TrajectoryData:
        """Gets trajectory data object from specified index.

        Args:
            index (int): List index of collection element.

        Returns:
            TrajectoryData: Collection element.
        """
        return self._collection[index]

    def deepcopy(self) -> TrajectoryCollection:
        """Returns copy of trajectory collection without dependencies.

        Returns:
            TrajectoryCollection: Copied trajectory collection.
        """
        return copy.deepcopy(TrajectoryCollection(self._collection))

    def delete_item(self, index: int) -> None:
        """Deletes collection element from specified index.

        Args:
            index (int): Deletion index.
        """
        del self._collection[index]
