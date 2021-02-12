""" Executable script to train a custom Gaussian Process on recorded flight data.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""


import os
import time
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.utils import safe_mkdir_recursive, load_pickled_models
from src.utils.utils import distance_maximizing_points, get_model_dir_and_file
from src.utils.utils import sample_random_points
from src.model_fitting.gp import CustomKernelFunctions as npKernelFunctions
from src.model_fitting.gp import CustomGPRegression as npGPRegression
from src.model_fitting.gp import GPEnsemble
from src.model_fitting.gp_common import GPDataset, restore_gp_regressors, read_dataset
from src.model_fitting.gp_visualization import gp_visualization_experiment
from config.configuration_parameters import ModelFitConfig as Conf


def plot_gp_regression(x_test, y_test, x_train, y_train, gp_mean, gp_std, gp_regressor, labels, title='', n_samples=3):

    # Assert the number of provided labels is coherent with the feature dimension [1] of the x vectors
    if len(x_test.shape) == 1:
        x_test = np.expand_dims(x_test, 1)
        n_subplots = 1
    else:
        n_subplots = x_test.shape[1]

    assert len(labels) == x_test.shape[1]

    # Generate samples if a gp_regressor is provided
    if gp_regressor is not None:
        # Sample from GP & plot samples
        y_samples = gp_regressor.sample_y(x_test, n_samples)
        y_samples = np.squeeze(y_samples)
    else:
        y_samples = None

    for i in range(n_subplots):
        plt.subplot(n_subplots, 1, i + 1)

        # Sort x axis values
        x_sort_ind_test = np.argsort(x_test[:, i])

        # Plot gp mean line
        plt.plot(x_test[x_sort_ind_test, i], gp_mean[x_sort_ind_test], 'k', lw=3, zorder=9)

        # Plot gp std area
        plt.fill_between(x_test[x_sort_ind_test, i],
                         gp_mean[x_sort_ind_test] - 3 * gp_std[x_sort_ind_test],
                         gp_mean[x_sort_ind_test] + 3 * gp_std[x_sort_ind_test],
                         alpha=0.2, color='k')

        if y_samples is not None:
            plt.plot(x_test[x_sort_ind_test, i], y_samples[x_sort_ind_test], '-o', lw=1)
            plt.xlim(min(x_test[:, i]), max(x_test[:, i]))
            plt.ylim(min(np.min(y_samples), np.min(gp_mean - 3 * gp_std)),
                     max(np.max(y_samples), np.max(gp_mean + 3 * gp_std)))

        if x_train is not None and y_train is not None:
            plt.scatter(x_train[:, i], y_train, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))

        if y_test is not None:
            plt.plot(x_test[x_sort_ind_test, i], y_test[x_sort_ind_test], lw=1, marker='o')

        if i == 0 and title != '':
            plt.title(title, fontsize=12)

        plt.ylabel(labels[i])

    plt.tight_layout()


def gp_train_and_save(x, y, gp_regressors, save_model, save_file, save_path, y_dims, cluster_n, progress_bar=True):
    """
    Trains and saves the 'm' GP's in the gp_regressors list. Each of these regressors will predict on one of the output
    variables only.

    :param x: Feature variables for the regression training. A list of length m where each entry is a Nxn dataset, N is
    the number of training samples and n is the feature space dimensionality. Each entry of this list will be used to
    train the respective GP.
    :param y: Output variables for the regression training. A list of length m where each entry is a N array, N is the
    number of training samples. Each entry of this list will be used as ground truth output for the respective GP.
    :param gp_regressors: List of m GPRegression objects (npGPRegression or skGPRegression)
    :param save_model: Bool. Whether to save the trained models or not.
    :param save_file: Name of file where to save the model. The 'pkl' extension will be added automatically.
    :param save_path: Path where to store the trained model pickle file.
    :param y_dims: List of length m, where each entry represents the state index that corresponds to each GP.
    :param cluster_n: Number (int) of the cluster currently being trained.
    :param progress_bar: Bool. Whether to visualize a progress bar or not.
    :return: the same list as te input gp_regressors variable, but each GP has been trained and saved if requested.
    """

    # If save model, generate saving directory
    if save_model:
        safe_mkdir_recursive(save_path)

    if progress_bar:
        print("Training {} gp regression models".format(len(y_dims)))
    prog_range = tqdm(y_dims) if progress_bar else y_dims

    for y_dim_reg, dim in enumerate(prog_range):

        # Fit one regressor for each output dimension
        gp_regressors[y_dim_reg].fit(x[y_dim_reg], y[y_dim_reg])
        if save_model:
            full_path = os.path.join(save_path, save_file + '_' + str(dim) + '_' + str(cluster_n) + '.pkl')
            gp_regressors[y_dim_reg].save(full_path)

    return gp_regressors


def main(x_features, u_features, reg_y_dim, quad_sim_options, dataset_name,
         x_cap, hist_bins, hist_thresh,
         n_train_points=50, n_restarts=10, n_clusters=1, load_clusters=False, model_name="model",
         dense_gp_name="model", dense_gp_points=100, dense_gp_version="", use_dense=False,
         visualize_data=False, visualize_model=False):

    """
    Reads the dataset specified and trains a GP model or ensemble on it. The regressed variables is the time-derivative
    of one of the states.
    The feature and regressed variables are identified through the index number in the state vector. It is defined as:
    [0: p_x, 1: p_y, 2:, p_z, 3: q_w, 4: q_x, 5: q_y, 6: q_z, 7: v_x, 8: v_y, 9: v_z, 10: w_x, 11: w_y, 12: w_z]
    The input vector is also indexed:
    [0: u_0, 1: u_1, 2: u_2, 3: u_3].

    :param x_features: List of n regression feature indices from the quadrotor state indexing.
    :type x_features: list
    :param u_features: List of n' regression feature indices from the input state.
    :type u_features: list
    :param reg_y_dim: Index of output dimension being regressed as the time-derivative.
    :type reg_y_dim: float
    :param dataset_name: Name of the dataset
    :param quad_sim_options: Dictionary of metadata of the dataset to be read.
    :param x_cap: cap value (in absolute value) for dataset pruning. Any input feature that exceeds this number in any
    dimension will be removed.
    :param hist_bins: Number of bins used for histogram pruning
    :param hist_thresh: Any bin with less data percentage than this number will be removed.
    :param n_train_points: Number of training points used for the current GP training.
    :param dense_gp_points: Number of training points used for the dense GP training. The dense GP will be loaded if
    possible.
    :param n_restarts: Number of restarts of nonlinear optimizer.
    :param n_clusters: Number of clusters used in GP ensemble. If 1, a normal GP is trained.
    :param load_clusters: True if attempt to load clusters from last GP training.
    :param model_name: Given name to the currently trained GP.
    :param dense_gp_name: Given name to the dense GP in case it needs to be used.
    :param dense_gp_version: Git hash of the folder where to find the dense GP.
    :param use_dense: Whether to sample the training set from a dense GP or sample from the training data.
    :param visualize_data: True if display some plots about the data loading, pruning and training process.
    :param visualize_model: True if display some performance plots about the trained model.
    """

    # #### Prepare saving directory for GP's #### #
    # Get git commit hash for saving the model
    git_version = ''
    try:
        git_version = subprocess.check_output(['git', 'describe', '--always']).strip().decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(e.returncode, e.output)
    print("The model will be saved using hash: %s" % git_version)

    gp_name_dict = {"git": git_version, "model_name": model_name, "params": quad_sim_options}
    save_file_path, save_file_name = get_model_dir_and_file(gp_name_dict)

    # #### DATASET LOADING #### #
    if isinstance(dataset_name, str):
        df_train = read_dataset(dataset_name, True, quad_sim_options)
        gp_dataset = GPDataset(df_train, x_features, u_features, reg_y_dim,
                               cap=x_cap, n_bins=hist_bins, thresh=hist_thresh, visualize_data=visualize_data)
    elif isinstance(dataset_name, GPDataset):
        gp_dataset = dataset_name
    else:
        raise TypeError("dataset_name must be a string or a GPDataset instance.")

    # Make clusters for multi-gp prediction
    gp_dataset.cluster(n_clusters, load_clusters=load_clusters, save_dir=save_file_path, visualize_data=visualize_data)

    # #### LOAD DENSE GP IF USING GP ENSEMBLE #### #
    if use_dense:
        load_options = {"git": dense_gp_version, "model_name": dense_gp_name, "params": quad_sim_options}
        loaded_models = load_pickled_models(model_options=load_options)

        if loaded_models is None:
            print("Model not found. Training a new dense GP with ")
            # Train model as accurate as possible. If dense_gp_points is specified, train a gp with that amount of
            # straining samples and use it to generate a dataset for the GP ensemble.
            dense_gp = main(x_features, u_features, reg_y_dim, quad_sim_options, dataset_name, x_cap, hist_bins,
                            hist_thresh, n_train_points=dense_gp_points, n_restarts=n_restarts,
                            model_name=dense_gp_name)

        else:
            dense_gp = restore_gp_regressors(loaded_models)
            print("Loaded dense GP model from: %s/%s" % (dense_gp_version, dense_gp_name))

    else:
        dense_gp = None

    # #### DECLARE SOME PARAMETERS AND VARIABLES #### #
    # List of trained GP regressors. One for each cluster
    gp_regressors = []

    # Prior parameters
    sigma_f = 0.5
    length_scale = .1
    sigma_n = 0.01

    gp_params = {"x_features": x_features, "u_features": u_features, "reg_dim": reg_y_dim,
                 "sigma_n": sigma_n, "n_restarts": n_restarts}

    # Get all cluster centroids for the current output dimension
    centroids = gp_dataset.centroids
    print("Training {} cluster model(s)".format(n_clusters))
    range_vec = tqdm(range(n_clusters))
    for cluster in range_vec:

        # #### TRAINING POINT SELECTION #### #

        cluster_mean = centroids[cluster]
        cluster_x_points = gp_dataset.get_x(cluster=cluster)
        cluster_y_points = gp_dataset.get_y(cluster=cluster)
        cluster_u_points = gp_dataset.get_u(cluster=cluster)

        # Select a base set of training points for the current cluster using PCA that are as separate from each
        # other as possible
        selected_points = distance_maximizing_points(
            cluster_x_points, cluster_mean, n_train_points=n_train_points, dense_gp=dense_gp, plot=False)

        cluster_y_mean = np.mean(cluster_y_points, 0)

        # If no dense_gp was provided to the previous function, training_points will be the indices of the training
        # points to choose from the training set
        if dense_gp is None:
            x_train = cluster_x_points[selected_points]
            y_train = np.squeeze(cluster_y_points[selected_points])
            training_points = selected_points
        else:
            # Generate a new dataset of synthetic data composed of x and y values
            x_mock = np.zeros((13, selected_points.shape[1]))
            if x_features:
                x_mock[np.array(x_features), :] = selected_points[:len(x_features)]
            u_mock = np.zeros((4, selected_points.shape[1]))
            if u_features:
                u_mock[np.array(u_features), :] = selected_points[len(x_features):]
            out = dense_gp.predict(x_mock, u_mock)
            out["pred"] = np.atleast_2d(out["pred"])
            y_train = np.squeeze(out["pred"][np.where(dense_gp.dim_idx == reg_y_dim)])
            x_train = selected_points.T
            training_points = []

        # Check if we still haven't used the entirety of the available points
        n_used_points = x_train.shape[0]
        if n_used_points < n_train_points and n_used_points < cluster_x_points.shape[0]:

            missing_pts = n_train_points - n_used_points

            training_points = sample_random_points(cluster_x_points, training_points, missing_pts, dense_gp)
            if dense_gp is None:
                # Transform from cluster data index to full dataset index
                x_train = cluster_x_points[training_points]
                y_train = np.squeeze(cluster_y_points[training_points])

            else:
                # Generate a new dataset of synthetic data composed of x and y values
                training_points = training_points.astype(int)
                x_mock = np.zeros((13, len(training_points)))
                if x_features:
                    x_mock[np.array(x_features), :] = cluster_x_points[training_points, :len(x_features)].T
                u_mock = np.zeros((4, len(training_points)))
                if u_features:
                    u_mock[np.array(u_features), :] = cluster_u_points[len(x_features):]
                out = dense_gp.predict(x_mock, u_mock)
                y_additional = np.squeeze(out["pred"][np.where(dense_gp.dim_idx == reg_y_dim)])
                y_train = np.append(y_train, y_additional)
                x_train = np.concatenate((x_train, cluster_x_points[training_points, :len(x_features)]), axis=0)

        # #### GP TRAINING #### #
        # Multidimensional input GP regressors
        l_scale = length_scale * np.ones((x_train.shape[1], 1))

        cluster_mean = centroids[cluster]
        gp_params["mean"] = cluster_mean
        gp_params["y_mean"] = cluster_y_mean

        # Train one independent GP for each output dimension
        exponential_kernel = npKernelFunctions('squared_exponential', params={'l': l_scale, 'sigma_f': sigma_f})
        gp_regressors.append(npGPRegression(kernel=exponential_kernel, **gp_params))
        gp_regressors[cluster] = gp_train_and_save([x_train], [y_train], [gp_regressors[cluster]], True, save_file_name,
                                                   save_file_path, [reg_y_dim], cluster, progress_bar=False)[0]

    if visualize_model:
        gp_ensemble = GPEnsemble()
        gp_ensemble.add_model(gp_regressors)
        x_features = x_features
        gp_visualization_experiment(quad_sim_options, gp_dataset,
                                    x_cap, hist_bins, hist_thresh,
                                    x_features, u_features, reg_y_dim,
                                    grid_sampling_viz=True, pre_set_gp=gp_ensemble)


def gp_evaluate_test_set(gp_regressors, gp_dataset, pruned=False, timed=False, progress_bar=False):
    """
    Runs GP prediction on a specified dataset.

    :param gp_regressors: GPEnsemble object
    :type gp_regressors: GPEnsemble
    :param gp_dataset: Dataset for evaluation
    :type gp_dataset: GPDataset
    :param pruned: Whether to use the pruned data or the raw data from the GPDataset.
    :param timed: whether to return the elapsed time of the data evaluation or not
    :param progress_bar: If True, a progress bar will be shown when evaluating the test data.
    :return: Given n number of samples of dimension d, return:
        - n x d vector of mean predictions,
        - n x d vector of std predictions,
        - n x d vector of ground truth values to compare with the predictions
    """

    x_test = gp_dataset.get_x(pruned=pruned, raw=True)
    u_test = gp_dataset.get_u(pruned=pruned, raw=True)
    dt_test = gp_dataset.get_dt(pruned=pruned)

    tic = time.time()
    out = gp_regressors.predict(x_test.T, u_test.T, return_std=True, progress_bar=progress_bar)
    mean_post = out["pred"]
    std_post = out["cov_or_std"]
    elapsed = time.time() - tic
    mean_post *= dt_test
    std_post *= dt_test
    mean_post = mean_post.T
    std_post = std_post.T

    if not timed:
        return mean_post, std_post
    else:
        return mean_post, std_post, elapsed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_points", type=int, default="20",
                        help="Number of training points used to fit the current GP model.")

    parser.add_argument("--model_name", type=str, default="",
                        help="Name assigned to the trained model.")

    parser.add_argument('--x', nargs='+', type=int, default=[7],
                        help='Regression X variables. Must be a list of integers between 0 and 12. Velocities xyz '
                             'correspond to indices 7, 8, 9.')

    parser.add_argument("--y", type=int, default=7,
                        help="Regression Y variable. Must be an integer between 0 and 12. Velocities xyz correspond to"
                             "indices 7, 8, 9.")

    input_arguments = parser.parse_args()

    # Use vx, vy, vz as input features
    x_feats = input_arguments.x
    u_feats = []

    # Regression dimension
    y_regressed_dim = input_arguments.y
    n_train_pts = input_arguments.n_points
    gp_name = input_arguments.model_name

    ds_name = Conf.ds_name
    simulation_options = Conf.ds_metadata

    histogram_pruning_bins = Conf.histogram_bins
    histogram_pruning_threshold = Conf.histogram_threshold
    x_value_cap = Conf.velocity_cap

    gp_dense_name = Conf.dense_model_name
    gp_id_custom = Conf.dense_model_version
    dense_n_points = Conf.dense_training_points
    with_dense = Conf.use_dense_model

    main(x_feats, u_feats, y_regressed_dim, simulation_options, ds_name,
         x_value_cap, histogram_pruning_bins, histogram_pruning_threshold,
         model_name=gp_name, n_train_points=n_train_pts,
         n_clusters=Conf.clusters, load_clusters=Conf.load_clusters,
         use_dense=with_dense,
         dense_gp_points=dense_n_points, dense_gp_name=gp_dense_name, dense_gp_version=gp_id_custom,
         visualize_data=Conf.visualize_data, visualize_model=Conf.visualize_training_result)
