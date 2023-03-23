import typing

import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse
from scipy import interpolate
from scipy.signal import find_peaks

from aimy_target_shooting.custom_types import Position3D, Velocity3D


def to_tuple_list(numpy_list: np.ndarray) -> typing.Union[Position3D, Velocity3D]:
    """Transforms list with position tuple from numpy
    array.

    Args:
        numpy_list (np.ndarray): Numpy array (nx3) with
        position (or velocity) trajectory.

    Returns:
        list: List of position (or velocity) tuples.
    """
    return [tuple(entry) for entry in numpy_list]


def to_numpy_list(
    tuple_list: typing.List[typing.Union[Position3D, Velocity3D]]
) -> typing.List[np.ndarray]:
    """List of numpy arrays for more efficient storing.
    Otherwise tuple lists can be fully transformed to np.array
    by standard numpy array initialisation of sequences.

    Args:
        tuple_list (typing.List[typing.Union[Position3D, Velocity3D]]):
        List of position (or velocity) tuples.

    Returns:
        typing.List[np.ndarray]: List of numpy nd.arrays.
    """
    return [np.array(entry) for entry in tuple_list]


def ip_to_launcher_name(ip: str) -> str:
    """Convenience function for fetching string from name.

    Args:
        ip (str): Launcher name.

    Returns:
        str: Affiliated IP address.
    """
    ip_str = {"10.42.31.174": "v2_offset", "10.42.26.171": "v3"}
    return ip_str[ip]


def confidence_ellipse(x, y, ax, n_std=4.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def cartesian_to_polar(
    x: float, y: float, z: float
) -> typing.Tuple[float, float, float]:
    """Transforms given Cartesian coordinates into
    polar / spherical coordinates.

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        z (float): z-coordinate.

    Returns:
        typing.List[float, float, float]: Spherical coordinate
        radius, phi and theta.
    """
    radius = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arcsin(y / np.sqrt(x**2 + y**2))
    theta = np.arcsin(z / radius)

    return radius, phi, theta


def polar_to_cartesian(
    radius: float, phi: float, theta: float
) -> typing.Tuple[float, float, float]:
    """Transforms polar / spherical coordinates
    into Cartesian coordinates.

    Args:
        radius (float): Radius of sphere.
        phi (float): Azimuthal angle of sphere.
        theta (float): Altitude angle of sphere.

    Returns:
        typing.List[float, float, float]: Cartesian coordinates
        x, y, z.
    """
    z = np.sin(theta) * radius
    y = np.sin(phi) * np.cos(theta) * radius
    x = np.cos(phi) * np.cos(theta) * radius

    return x, y, z


def find_rebound(
    positions: typing.Union[np.ndarray, typing.List[float]],
    detection_height: float = 0.76,
    detection_threshold: float = -0.03,
    detection_distance: int = 10,
) -> typing.List[int]:
    """Finds rebound indices.

    Args:
        positions (typing.Union[np.ndarray, typing.List[float]]): Height positions
        of a ball trajectory.
        detection_height (float, optional): Height for detecting rebounds.
        Defaults to 0.76.
        detection_threshold (float, optional): Detection threshold. Defaults to -0.03.
        detection_distance (int, optional): Distance parameter for checking the
        neighborhood for other rebounds. Defaults to 10.

    Returns:
        typing.List[int]: List of rebound indices.
    """

    positions = np.array(positions)
    positions_unbaised = positions[:, 2] - detection_height
    positions_transformed = -1 * positions_unbaised

    discontinuity_indices = find_peaks(
        positions_transformed,
        height=detection_threshold,
        distance=detection_distance,
    )[0]

    return discontinuity_indices


def f_phi(
    actuation: typing.Optional[float] = None,
    phi: typing.Optional[float] = None,
    radians: bool = True,
) -> float:
    """Transformation of actuation with range 0 to 1 to Azimuth angle and
    vice versa.

    Args:
        actuation (float, optional): Given actuation. Defaults to None.
        phi (float, optional): Given angle value. Defaults to None.
        radians (bool, optional): Specifier if given or returned angle
        should be radian. Defaults to True.

    Raises:
        AttributeError: Raised if neither actuation or angle are given.
        AttributeError: Raised if both actuation and angle are given.

    Returns:
        float: Returns either angle or actuation, depending on given
        attributes.
    """
    if actuation is None and phi is None:
        raise AttributeError("No argument given!")

    if actuation is not None and phi is not None:
        raise AttributeError("Only specify either actuation or angle!")

    phi_pos = [
        16.81766552,
        12.41133775,
        4.05559336,
        0.0,
        -4.26665414,
        -11.6580105,
        -16.54297629,
    ]
    phi_pos_actuation = [i * (1 / 6) for i in range(7)]

    output = None

    if phi is not None:
        f = interpolate.interp1d(phi_pos, phi_pos_actuation)

        if radians:
            phi = np.rad2deg(phi)

        actuation = f(phi)

        output = actuation

    if actuation is not None:
        f = interpolate.interp1d(phi_pos_actuation, phi_pos)
        phi_calc = f(actuation)

        if radians:
            phi_calc = np.radians(phi_calc)

        output = phi_calc

    return output


def f_theta(
    actuation: typing.Optional[float] = None,
    theta: typing.Optional[float] = None,
    radians: bool = True,
):
    """Transformation of actuation with range 0 to 1 to altitude angle and
    vice versa.

    Args:
        actuation (float, optional): Given actuation. Defaults to None.
        theta (float, optional): Given angle value. Defaults to None.
        radians (bool, optional): Specifier if given or returned angle
        should be radian. Defaults to True.

    Raises:
        AttributeError: Raised if neither actuation or angle are given.
        AttributeError: Raised if both actuation and angle are given.

    Returns:
        float: Returns either angle or actuation, depending on given
        attributes.
    """
    if actuation is None and theta is None:
        raise AttributeError("No argument given!")

    if actuation is not None and theta is not None:
        raise AttributeError("Only specify either actuation or angle!")

    theta_pos = [6.40400874, 12.17836522, 19.90385964, 28.11300675, 37.14585762]
    theta_pos_actuation = [i * 0.25 for i in range(5)]

    output = None

    if theta is not None:
        f = interpolate.interp1d(theta_pos, theta_pos_actuation)

        if radians:
            theta = np.rad2deg(theta)

        actuation = f(theta)

        output = actuation

    if actuation is not None:
        f = interpolate.interp1d(theta_pos_actuation, theta_pos)
        theta_calc = f(actuation)

        if radians:
            theta_calc = np.radians(theta_calc)

        output = theta_calc

    return output
