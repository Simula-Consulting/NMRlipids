import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from matplotlib.axes import Axes
from typing import Any, Optional
from pathlib import Path

def getFormFactorAndTotalDensityPair(
    system: dict[str, Any], databankPath: str
) -> tuple[Optional[list[Any]], Optional[list[Any]]]:
    """
    Returns form factor and total density profiles of the simulation

    :param system: NMRlipids databank dictionary describing the simulation
    :param databankPath: Path to the databank

    :return: form factor (FFsim) and total density (TDsim) of the simulation
    """
    databankPath = Path(databankPath)
    FFpathSIM = databankPath / "Data" / "Simulations" / system['path'] / "FormFactor.json"
    TDpathSIM = databankPath / "Data" / "Simulations" / system['path'] / "TotalDensity.json"

    # Load form factor and total density
    try:
        with open(FFpathSIM, "r") as json_file:
            FFsim = json.load(json_file)
        with open(TDpathSIM, "r") as json_file:
            TDsim = json.load(json_file)
    except Exception:
        FFsim = None
        TDsim = None

    return FFsim, TDsim


def plot_total_densities_to_ax(
    ax: Axes, all_td_x: list[float], all_td_y: list[float], lines: list[float] = []
) -> Axes:
    """
    Plot all total density profiles to ax

    :param ax: Axes object to plot on.
    :param all_td_x: List of lists: x coordinates for each total density profile
    :param all_td_y: List of lists: y coordinates for each total density profile
    :param lines: List of x values to plot as vertical lines

    :return: ax object with total densities (and lines) plotted
    """
    if isinstance(all_td_y, list):
        for x_vector, y_vector in zip(all_td_x, all_td_y):
            ax.plot(x_vector, y_vector)
    elif isinstance(all_td_y, pd.DataFrame):
        for _, row in all_td_y.iterrows():
            ax.plot(all_td_x, row.to_list())
    for value in lines:
        ax.axvline(value, color="k", linestyle="solid")
    return ax


def plot_form_factors_to_ax(ax: Axes, sim_FF_df: pd.DataFrame) -> Axes:
    """
    Plot all form factor profiles to ax

    :param ax: Axes object to plot on
    :param sim_FF_df: pd.DataFrame with form factors as rows

    :return: ax object with form factors plotted
    """
    for index, row in sim_FF_df.iterrows():
        ax.plot(row.to_list())
    return ax


def extrapolate_X(
    x_vector: np.ndarray,
    length_of_padded_data: int,
    x_interval_start: float,
    x_interval_end: float,
) -> np.ndarray:
    """
    Extrapolates total density x values to match desired x range and dimensionality

    :param x_vector: Original total density x values
    :param length_of_padded_data: Desired length of padded data
    :param x_interval_start: Lower end of range for the homogenized data
    :param x_interval_end: Lower end of range for the homogenized data

    :return: padded x vector
    """
    padding_length = max(0, length_of_padded_data - len(x_vector))
    first_padding_length = padding_length // 2
    last_padding_length = padding_length - first_padding_length

    x_min = min(x_vector)
    x_max = max(x_vector)

    # Check if the range of the x values is smaller than the required range:
    if x_min > x_interval_start and x_max < x_interval_end:
        # If narrower, extrapolate in the x direction by replicating the y values at the ends
        padding_start = np.linspace(
            x_interval_start, x_min, num=max(0, first_padding_length), endpoint=False
        )
        padding_end = np.linspace(
            x_max, x_interval_end, num=max(0, last_padding_length), endpoint=False
        )
    elif x_min < x_interval_start and x_max > x_interval_end:
        # If wider, pad at the ends without extrapolating to make dimensions equal
        padding_start = np.repeat(x_min, first_padding_length)
        padding_end = np.repeat(x_max, last_padding_length)
    else:
        raise NotImplementedError
    return np.concatenate([padding_start, x_vector, padding_end])


def extrapolate_Y(y_vector: np.ndarray, length_of_padded_data: int) -> np.ndarray:
    """
    Extrapolates total density y values by repeating the y values at the ends of the observation window

    :param y_vector: Original total density y values
    :param length_of_padded_data: Desired length of padded data

    :return: padded y vector
    """
    padding_length = max(0, length_of_padded_data - len(y_vector))
    first_padding_length = padding_length // 2
    last_padding_length = padding_length - first_padding_length

    first_pad_value = y_vector[0]
    last_pad_value = y_vector[-1]

    padding_start = [first_pad_value] * first_padding_length
    padding_end = [last_pad_value] * last_padding_length
    return np.concatenate([padding_start, y_vector, padding_end])


def interpolate_with_GPR(
    rescaled_all_td_x: list[np.ndarray],
    rescaled_all_td_y: list[np.ndarray],
    uniform_x_range: np.ndarray,
) -> list[np.ndarray]:
    """
    Fits a Gaussian process regression to the observed points, then predicts values on the  uniform grid between the observations

    :param rescaled_all_x: List of total density x values (np.ndarray) for all cases
    :param rescaled_all_y: List of total density y values (np.ndarray) for all cases
    :param uniform_x_range: X coordinates on which the y values will be predicted for all patients

    :return: List og np.ndarrays, where each np.ndarray contains the total density y values predicted by on uniform x range for one patient
    """
    # Create and fit Gaussian process regressor
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
    return [
        gp.fit(x_vector.reshape(-1, 1), y_vector).predict(uniform_x_range)
        for x_vector, y_vector in zip(rescaled_all_td_x, rescaled_all_td_y)
    ]


def split_train_and_test(
    sim_FF_df: pd.DataFrame, sim_TD_y_df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split FF (input) and TD (output) pairs into train and test input and output pairs

    :param sim_FF_df: Data frame containing form factors
    :param sim_TD_y_df: Data frame containing total density profiles in the same order as sim_FF_df
    :param rng: random number generator created and seeded at the beginning of the main script

    :return: train_input (FF), train_output (TD), test_input (FF), test_output (TD)
    """
    N_total = sim_FF_df.shape[0]
    N_train = int(round(0.8 * N_total, 0))
    shuffle_indices = rng.permutation(N_total)
    train_indices = shuffle_indices[0:N_train]
    test_indices = shuffle_indices[N_train:]

    # Select, transpose or not, and convert to numpy float
    train_input = sim_FF_df.loc[train_indices, :].astype(np.float32)
    train_output = sim_TD_y_df.loc[train_indices, :].astype(np.float32)
    test_input = sim_FF_df.loc[test_indices, :].astype(np.float32)
    test_output = sim_TD_y_df.loc[test_indices, :].astype(np.float32)

    return train_input, train_output, test_input, test_output


def plot_training_trajectory(ax: Axes, history: object) -> Axes:
    """
    Plot the training trajectory

    :param ax: Axes object to plot on
    :param history: Training history object output from the model fit procedure

    :return: ax object with training history plotted
    """
    ax.plot(history.history["loss"], color="green", label="Training loss")
    ax.plot(history.history["val_loss"], color="orange", label="Validation loss")
    ax.legend()
    return ax
