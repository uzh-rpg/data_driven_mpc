""" Implementation and fitting of the RDRv linear regression model on flight data.

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
import joblib
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from src.model_fitting.gp_common import GPDataset, read_dataset
from src.utils.utils import v_dot_q, get_model_dir_and_file, safe_mknode_recursive
from config.configuration_parameters import ModelFitConfig as Conf


def linear_rdrv_fitting(x, y, feat_idx):
    drag_coeffs = np.zeros((3, 3))
    for i in range(x.shape[1]):
        reg = LinearRegression(fit_intercept=False).fit(x[:, i, np.newaxis], y[:, i])
        drag_coeffs[feat_idx[i], feat_idx[i]] = reg.coef_

    return drag_coeffs


def load_rdrv(model_options):
    directory, file_name = get_model_dir_and_file(model_options)
    rdrv_d = joblib.load(os.path.join(directory, file_name + '.pkl'))
    return rdrv_d


def main(model_name, features, quad_sim_options, dataset_name,
         x_cap, hist_bins, hist_thresh, plot=False):

    df_train = read_dataset(dataset_name, True, quad_sim_options)
    gp_dataset = GPDataset(df_train, features, [], features,
                           cap=x_cap, n_bins=hist_bins, thresh=hist_thresh, visualize_data=False)

    # Get X,Y datasets for the specified regression dimensions (subset of [7, 8, 9])
    a_err_b = gp_dataset.y
    v_b = gp_dataset.x

    drag_coeffs = linear_rdrv_fitting(v_b, a_err_b, np.array(features) - 7)

    # Get git commit hash for saving the model
    git_version = ''
    try:
        git_version = subprocess.check_output(['git', 'describe', '--always']).strip().decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(e.returncode, e.output)
    gp_name_dict = {"git": git_version, "model_name": model_name, "params": quad_sim_options}
    save_file_path, save_file_name = get_model_dir_and_file(gp_name_dict)
    safe_mknode_recursive(save_file_path, save_file_name + '.pkl', overwrite=True)
    file_name = os.path.join(save_file_path, save_file_name + '.pkl')
    joblib.dump(drag_coeffs, file_name)

    if not plot:
        return drag_coeffs

    # Get X,Y datasets for velocity dimensions [7, 8, 9]
    a_err_b = gp_dataset.get_y(pruned=True, raw=True)[:, 7:10]
    v_b = gp_dataset.get_x(pruned=True, raw=True)[:, 7:10]

    # Compute predictions with RDRv model
    preds = []
    for i in range(len(a_err_b)):
        preds.append(np.matmul(drag_coeffs, v_b[i]))
    preds = np.stack(preds)

    ax_names = [r'$v_B^x$', r'$v_B^y$', r'$v_B^z$']
    y_dim_name = [r'drag $a_B^x$', 'drag $a_B^y$', 'drag $a_B^z$']
    fig, ax = plt.subplots(len(features), 1, sharex='all')
    for i in range(len(features)):
        ax[i].scatter(v_b[:, i], a_err_b[:, i], label='data')
        ax[i].scatter(v_b[:, i], preds[:, i], label='RDRv')
        ax[i].legend()
        ax[i].set_ylabel(y_dim_name[features[i] - 7])
        ax[i].set_xlabel(ax_names[features[i] - 7])
    ax[0].set_title('Body reference frame')

    # Remap error to world coordinates and predictions too
    q = gp_dataset.get_x(pruned=True, raw=True)[:, 3:7]
    for i in range(len(a_err_b)):
        a_err_b[i] = v_dot_q(a_err_b[i], q[i])
        preds[i] = v_dot_q(preds[i], q[i])

    ax_names = [r'$v_W^x$', r'$v_W^y$', r'$v_W^z$']
    y_dim_name = [r'drag $a_W^x$', 'drag $a_W^y$', 'drag $a_W^z$']
    fig, ax = plt.subplots(len(features), 1, sharex='all')
    for i in range(len(features)):
        ax[i].scatter(v_b[:, i], a_err_b[:, i], label='data')
        ax[i].scatter(v_b[:, i], preds[:, i], label='RDRv')
        ax[i].legend()
        ax[i].set_ylabel(y_dim_name[features[i] - 7])
        ax[i].set_xlabel(ax_names[features[i] - 7])
    ax[0].set_title('World reference frame')
    plt.show()

    return drag_coeffs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="",
                        help="Name assigned to the trained model.")

    parser.add_argument('--x', nargs='+', type=int, default=[7, 8, 9],
                        help='Regression X and Y variables. Must be a list of integers between 0 and 12.'
                             'Velocities xyz correspond to indices 7, 8, 9.')

    input_arguments = parser.parse_args()

    reg_dimensions = input_arguments.x
    rdrv_name = input_arguments.model_name

    histogram_bins = Conf.histogram_bins
    histogram_threshold = Conf.histogram_threshold
    histogram_cap = Conf.velocity_cap

    ds_name = Conf.ds_name
    simulation_options = Conf.ds_metadata

    main(model_name=rdrv_name, features=reg_dimensions, quad_sim_options=simulation_options, dataset_name=ds_name,
         x_cap=histogram_cap, hist_bins=histogram_bins, hist_thresh=histogram_threshold,
         plot=Conf.visualize_training_result)
