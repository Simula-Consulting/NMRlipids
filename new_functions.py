import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

def getFormFactorAndTotalDensityPair(system):
    """
    
    Returns form factor and total density profiles of the simulation
    
    :param system: NMRlipids databank dictionary describing the simulation.
    
    :return: form factor (FFsim) and total density (TDsim) of the simulation
    """
    FFpathSIM = "./Databank/Data/Simulations/" + system['path'] + "FormFactor.json"
    TDpathSIM = "./Databank/Data/Simulations/" + system['path'] + "TotalDensity.json"

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


def plot_densities(all_td_x, all_td_y, title='All total density profiles', lines=[]):
    """

    Plot all total density profiles
    """
    plt.figure(figsize=(6, 6))
    if isinstance(all_td_y, list):
        for x_vector, y_vector in zip(all_td_x, all_td_y):
            plt.plot(x_vector, y_vector)
    elif isinstance(all_td_y, pd.DataFrame):
        for index, row in all_td_y.iterrows():
            plt.plot(all_td_x, row.to_list())
    for line_value in lines:
        plt.axvline(line_value, color='k', linestyle='solid')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_form_factors(sim_FF_df):
    # Plot all form factors
    plt.figure(figsize=(12, 6))
    for index, row in sim_FF_df.iterrows():
        plt.plot(row.to_list())
    plt.title('All form factor profiles')
    plt.tight_layout()
    plt.show()


def extrapolate_X(x_vector, length_of_padded_data, x_interval_start, x_interval_end):
    """
    
    Returns form factor and total density profiles of the simulation
    
    :param system: NMRlipids databank dictionary describing the simulation.
    
    :return: form factor (FFsim) and total density (TDsim) of the simulation
    """
    """
    Extrapolates 

    """
    padding_length = max(0, length_of_padded_data - len(x_vector))
    first_padding_length = padding_length // 2
    last_padding_length = padding_length - first_padding_length

    x_min = min(x_vector)
    x_max = max(x_vector)

    # Check if the range of the x values is smaller than the required range: 
    if x_min > x_interval_start and x_max < x_interval_end:
        # If narrower, extrapolate in the x direction by replicating the y values at the ends
        padding_start = np.linspace(x_interval_start, x_min, num=max(0, first_padding_length), endpoint=False)
        padding_end = np.linspace(x_max, x_interval_end, num=max(0, last_padding_length), endpoint=False)
    else: 
        # If wider, pad at the ends without extrapolating to make dimensions equal
        padding_start = np.repeat(x_min, first_padding_length)
        padding_end = np.repeat(x_max, last_padding_length)
    return np.concatenate([padding_start, x_vector, padding_end])


def extrapolate_Y(y_vector, length_of_padded_data):
    """
    
    Returns form factor and total density profiles of the simulation
    
    :param system: NMRlipids databank dictionary describing the simulation.
    
    :return: form factor (FFsim) and total density (TDsim) of the simulation
    """
    """
    The y values at the beginning and end are copied and used to extrapolate

    """
    padding_length = max(0, length_of_padded_data - len(y_vector))
    first_padding_length = padding_length // 2
    last_padding_length = padding_length - first_padding_length

    first_pad_value = y_vector[0]
    last_pad_value = y_vector[-1]

    padding_start = [first_pad_value] * first_padding_length
    padding_end = [last_pad_value] * last_padding_length
    return np.concatenate([padding_start, y_vector, padding_end])


def interpolate_with_GPR(rescaled_all_td_x, rescaled_all_td_y, uniform_x_range):
    # Create and fit Gaussian process regressor
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
    return [gp.fit(x_vector.reshape(-1, 1), y_vector).predict(uniform_x_range) for x_vector, y_vector in zip(rescaled_all_td_x, rescaled_all_td_y)]


"""
def plot_gpr_fits():
    # Plot the original data and the predictions
    plt.figure(figsize=(12, 6))

    # Plot the function, the prediction and the 95% confidence interval
    plt.plot(uniform_x_range, y_pred, label='GPR Prediction')
    plt.fill_between(uniform_x_range.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.5)
    plt.scatter(x_vector, y_vector, label='Training Points', c='red')
    plt.show()

    # Set labels and legend
    plt.xlabel('X values')
    plt.ylabel('Predicted Y values')
    plt.legend()
    plt.show()
"""


def split_train_and_test(sim_FF_df, sim_TD_y_df, rng):
    """
    
    :return: 
    """
    N_total = sim_FF_df.shape[0]
    N_train = int(round(0.8*N_total,0))
    shuffle_indices = rng.permutation(N_total)
    train_indices = shuffle_indices[0:N_train]
    test_indices = shuffle_indices[N_train:]

    # Select, transpose or not, and convert to numpy float
    train_input = sim_FF_df.loc[train_indices,:].astype(np.float32)
    train_output = sim_TD_y_df.loc[train_indices,:].astype(np.float32)
    test_input = sim_FF_df.loc[test_indices,:].astype(np.float32)
    test_output = sim_TD_y_df.loc[test_indices,:].astype(np.float32)
    
    return train_input, train_output, test_input, test_output 


def plot_training_trajectory(history):
    plt.plot(history.history['loss'], color = 'green', label = 'Training loss')
    plt.plot(history.history['val_loss'], color = 'orange', label = 'Validation loss')
    plt.legend()
    plt.show()


# The following functions already exist: 
#
# def plotSimulation(ID, lipid):
#    """
#    Creates plots of form factor and C-H bond order parameters for the selected ``lipid`` from a simulation with the given ``ID`` number. 
#
#    :param ID: NMRlipids databank ID number of the simulation
#    :param lipid: universal molecul name of the lipid
#
#    """
