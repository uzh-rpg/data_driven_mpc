""" Executable script for visual evaluation of the trained GP quality.

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

import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.model_fitting.gp_common import GPDataset, restore_gp_regressors, read_dataset
from config.configuration_parameters import ModelFitConfig as Conf
from src.utils.utils import load_pickled_models
from src.utils.visualization import visualize_gp_inference


def gp_visualization_experiment(quad_sim_options, dataset_name,
                                x_cap, hist_bins, hist_thresh,
                                x_vis_feats, u_vis_feats, y_vis_feats,
                                grid_sampling_viz=False,
                                load_model_version="", load_model_name="", pre_set_gp=None):

    # #### GP ENSEMBLE LOADING #### #
    if pre_set_gp is None:
        load_options = {"git": load_model_version, "model_name": load_model_name, "params": quad_sim_options}
        loaded_models = load_pickled_models(model_options=load_options)
        if loaded_models is None:
            raise FileNotFoundError("Model not found")
        gp_ensemble = restore_gp_regressors(loaded_models)
    else:
        gp_ensemble = pre_set_gp

    # #### DATASET LOADING #### #
    # Pre-set labels of the data:
    labels_x = [
        r'$p_x\:\left[m\right]$', r'$p_y\:\left[m\right]$', r'$p_z\:\left[m\right]$',
        r'$q_w\:\left[rad\right]$', r'$q_x\:\left[rad\right]$', r'$q_y\:\left[rad\right]$', r'$q_z\:\left[rad\right]$',
        r'$v_x\:\left[\frac{m}{s}\right]$', r'$v_y\:\left[\frac{m}{s}\right]$', r'$v_z\:\left[\frac{m}{s}\right]$',
        r'$w_x\:\left[\frac{rad}{s}\right]$', r'$w_y\:\left[\frac{rad}{s}\right]$', r'$w_z\:\left[\frac{rad}{s}\right]$'
    ]
    labels_u = [r'$u_1$', r'$u_2$', r'$u_3$', r'$u_4$']
    labels = [labels_x[feat] for feat in x_vis_feats]
    labels_ = [labels_u[feat] for feat in u_vis_feats]
    labels = labels + labels_

    if isinstance(dataset_name, str):
        test_ds = read_dataset(dataset_name, True, quad_sim_options)
        test_gp_ds = GPDataset(test_ds, x_features=x_vis_feats, u_features=u_vis_feats, y_dim=y_vis_feats,
                               cap=x_cap, n_bins=hist_bins, thresh=hist_thresh, visualize_data=False)
    else:
        test_gp_ds = dataset_name

    x_test = test_gp_ds.get_x(pruned=True, raw=True)
    u_test = test_gp_ds.get_u(pruned=True, raw=True)
    y_test = test_gp_ds.get_y(pruned=True, raw=False)
    dt_test = test_gp_ds.get_dt(pruned=True)
    x_pred = test_gp_ds.get_x_pred(pruned=True, raw=False)

    if isinstance(y_vis_feats, list):
        y_dims = [np.where(gp_ensemble.dim_idx == y_dim)[0][0] for y_dim in y_vis_feats]
    else:
        y_dims = np.where(gp_ensemble.dim_idx == y_vis_feats)[0]

    # #### VISUALIZE GRID SAMPLING RESULTS IN TRAINING SET RANGE #### #
    if grid_sampling_viz:
        visualize_gp_inference(x_test, u_test, y_test, gp_ensemble, x_vis_feats, y_dims, labels)

    # #### EVALUATE GP ON TEST SET #### #
    print("Test set prediction...")
    outs = gp_ensemble.predict(x_test.T, u_test.T, return_std=True, progress_bar=True)
    mean_estimate = np.atleast_2d(np.atleast_2d(outs["pred"])[y_dims])
    std_estimate = np.atleast_2d(np.atleast_2d(outs["cov_or_std"])[y_dims])
    mean_estimate = mean_estimate.T * dt_test[:, np.newaxis]
    std_estimate = std_estimate.T * dt_test[:, np.newaxis]

    # Undo dt normalization
    y_test *= dt_test[:, np.newaxis]

    # Error of nominal model
    nominal_diff = y_test

    # GP regresses model error, correct the predictions of the nominal model
    augmented_diff = nominal_diff - mean_estimate
    mean_estimate += x_pred

    nominal_rmse = np.mean(np.abs(nominal_diff), 0)
    augmented_rmse = np.mean(np.abs(augmented_diff), 0)

    labels = [r'$v_x$ error', r'$v_y$ error', r'$v_z$ error']
    t_vec = np.cumsum(dt_test)
    plt.figure()
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    for i in range(std_estimate.shape[1]):
        plt.subplot(std_estimate.shape[1], 1, i+1)
        plt.plot(t_vec, np.zeros(augmented_diff[:, i].shape), 'k')
        plt.plot(t_vec, augmented_diff[:, i], 'b', label='augmented_err')
        plt.plot(t_vec, augmented_diff[:, i] - 3 * std_estimate[:, i], ':c')
        plt.plot(t_vec, augmented_diff[:, i] + 3 * std_estimate[:, i], ':c', label='3 std')
        if nominal_diff is not None:
            plt.plot(t_vec, nominal_diff[:, i], 'r', label='nominal_err')
            plt.title('Mean dt: %.2f s. Nominal RMSE: %.5f [m/s].  Augmented rmse: %.5f [m/s]' % (
                float(np.mean(dt_test)), nominal_rmse[i], augmented_rmse[i]))
        else:
            plt.title('Mean dt: %.2f s. Augmented RMSE: %.5f [m/s]' % (
                float(np.mean(dt_test)), float(augmented_rmse[i])))

        plt.ylabel(labels[i])
        plt.legend()

        if i == std_estimate.shape[1] - 1:
            plt.xlabel('time (s)')

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_version", type=str, default="",
                        help="Version to load for the regression models. By default it is an 8 digit git hash.")

    parser.add_argument("--model_name", type=str, default="",
                        help="Name of the regression model within the specified <model_version> folder.")

    input_arguments = parser.parse_args()

    ds_name = Conf.ds_name
    simulation_options = Conf.ds_metadata

    histogram_pruning_bins = Conf.histogram_bins
    histogram_pruning_threshold = Conf.histogram_threshold
    x_value_cap = Conf.velocity_cap

    x_viz = Conf.x_viz
    u_viz = Conf.u_viz
    y_viz = Conf.y_viz
    gp_load_model_name = input_arguments.model_name
    gp_load_model_version = input_arguments.model_version
    gp_visualization_experiment(simulation_options, ds_name,
                                x_value_cap, histogram_pruning_bins, histogram_pruning_threshold,
                                x_vis_feats=x_viz, u_vis_feats=u_viz, y_vis_feats=y_viz,
                                grid_sampling_viz=True,
                                load_model_version=gp_load_model_version,
                                load_model_name=gp_load_model_name)