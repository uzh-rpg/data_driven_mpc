""" Contains a set of utility functions and classes for the GP training and inference.

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


import copy
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.model_fitting.gp import GPEnsemble, CustomGPRegression as npGPRegression
from src.utils.utils import undo_jsonify, prune_dataset, safe_mknode_recursive, get_data_dir_and_file, \
    separate_variables, v_dot_q, quaternion_inverse
from src.utils.visualization import visualize_data_distribution


class GPDataset:
    def __init__(self, raw_ds=None,
                 x_features=None, u_features=None, y_dim=None,
                 cap=None, n_bins=None, thresh=None,
                 visualize_data=False):

        self.data_labels = [r'$p_x$', r'$p_y$', r'$p_z$', r'$q_w$', r'$q_x$', r'$q_y$', r'$q_z$',
                            r'$v_x$', r'$v_y$', r'$v_z$', r'$w_x$', r'$w_y$', r'$w_z$']

        # Raw dataset data
        self.x_raw = None
        self.x_out_raw = None
        self.u_raw = None
        self.y_raw = None
        self.x_pred_raw = None
        self.dt_raw = None

        self.x_features = x_features
        self.u_features = u_features
        self.y_dim = y_dim

        # Data pruning parameters
        self.cap = cap
        self.n_bins = n_bins
        self.thresh = thresh
        self.plot = visualize_data

        # GMM clustering
        self.pruned_idx = []
        self.cluster_agency = None
        self.centroids = None

        # Number of data clusters
        self.n_clusters = 1

        if raw_ds is not None:
            self.load_data(raw_ds)
            self.prune()

    def load_data(self, ds):
        x_raw = undo_jsonify(ds['state_in'].to_numpy())
        x_out = undo_jsonify(ds['state_out'].to_numpy())
        x_pred = undo_jsonify(ds['state_pred'].to_numpy())
        u_raw = undo_jsonify(ds['input_in'].to_numpy())

        dt = ds["dt"].to_numpy()
        invalid = np.where(dt == 0)

        # Remove invalid entries (dt = 0)
        x_raw = np.delete(x_raw, invalid, axis=0)
        x_out = np.delete(x_out, invalid, axis=0)
        x_pred = np.delete(x_pred, invalid, axis=0)
        u_raw = np.delete(u_raw, invalid, axis=0)
        dt = np.delete(dt, invalid, axis=0)

        # Rotate velocities to body frame and recompute errors
        x_raw = world_to_body_velocity_mapping(x_raw)
        x_pred = world_to_body_velocity_mapping(x_pred)
        x_out = world_to_body_velocity_mapping(x_out)
        y_err = x_out - x_pred

        # Normalize error by window time (i.e. predict error dynamics instead of error itself)
        y_err /= np.expand_dims(dt, 1)

        # Select features
        self.x_raw = x_raw
        self.x_out_raw = x_out
        self.u_raw = u_raw
        self.y_raw = y_err
        self.x_pred_raw = x_pred
        self.dt_raw = dt

    def prune(self):
        # Prune noisy data
        if self.cap is not None and self.n_bins is not None and self.thresh is not None:
            x_interest = np.array([7, 8, 9])
            y_interest = np.array([7, 8, 9])

            labels = [self.data_labels[dim] for dim in np.atleast_1d(y_interest)]

            prune_x_data = self.get_x(pruned=False, raw=True)[:, x_interest]
            prune_y_data = self.get_y(pruned=False, raw=True)[:, y_interest]
            self.pruned_idx.append(prune_dataset(prune_x_data, prune_y_data, self.cap, self.n_bins, self.thresh,
                                                 plot=self.plot, labels=labels))

    def get_x(self, cluster=None, pruned=True, raw=False):

        if cluster is not None:
            assert pruned

        if raw:
            return self.x_raw[tuple(self.pruned_idx)] if pruned else self.x_raw

        if self.x_features is not None and self.u_features is not None:
            x_f = self.x_features
            u_f = self.u_features
            data = np.concatenate((self.x_raw[:, x_f], self.u_raw[:, u_f]), axis=1) if u_f else self.x_raw[:, x_f]
        else:
            data = np.concatenate((self.x_raw, self.u_raw), axis=1)
        data = data[:, np.newaxis] if len(data.shape) == 1 else data

        if pruned or cluster is not None:
            data = data[tuple(self.pruned_idx)]
            data = data[self.cluster_agency[cluster]] if cluster is not None else data

        return data

    @property
    def x(self):
        return self.get_x()

    def get_x_out(self, cluster=None, pruned=True):

        if cluster is not None:
            assert pruned

        if pruned or cluster is not None:
            data = self.x_out_raw[tuple(self.pruned_idx)]
            data = data[self.cluster_agency[cluster]] if cluster is not None else data

            return data

        return self.x_out_raw[tuple(self.pruned_idx)] if pruned else self.x_out_raw

    @property
    def x_out(self):
        return self.get_x_out()

    def get_u(self, cluster=None, pruned=True, raw=False):

        if cluster is not None:
            assert pruned

        if raw:
            return self.u_raw[tuple(self.pruned_idx)] if pruned else self.u_raw

        data = self.u_raw[:, self.u_features] if self.u_features is not None else self.u_raw
        data = data[:, np.newaxis] if len(data.shape) == 1 else data

        if pruned or cluster is not None:
            data = data[tuple(self.pruned_idx)]
            data = data[self.cluster_agency[cluster]] if cluster is not None else data

        return data

    @property
    def u(self):
        return self.get_u()

    def get_y(self, cluster=None, pruned=True, raw=False):

        if cluster is not None:
            assert pruned

        if raw:
            return self.y_raw[self.pruned_idx] if pruned else self.y_raw

        data = self.y_raw[:, self.y_dim] if self.y_dim is not None else self.y_raw
        data = data[:, np.newaxis] if len(data.shape) == 1 else data

        if pruned or cluster is not None:
            data = data[tuple(self.pruned_idx)]
            data = data[self.cluster_agency[cluster]] if cluster is not None else data

        return data

    @property
    def y(self):
        return self.get_y()

    def get_x_pred(self, pruned=True, raw=False):

        if raw:
            return self.x_pred_raw[tuple(self.pruned_idx)] if pruned else self.x_pred_raw

        data = self.x_pred_raw[:, self.y_dim] if self.y_dim is not None else self.x_pred_raw
        data = data[:, np.newaxis] if len(data.shape) == 1 else data

        if pruned:
            data = data[tuple(self.pruned_idx)]

        return data

    @property
    def x_pred(self):
        return self.get_x_pred()

    def get_dt(self, pruned=True):

        return self.dt_raw[tuple(self.pruned_idx)] if pruned else self.dt_raw

    @property
    def dt(self):
        return self.get_dt()

    def cluster(self, n_clusters, load_clusters=False, save_dir="", visualize_data=False):
        self.n_clusters = n_clusters

        x_pruned = self.x
        y_pruned = self.y

        file_name = os.path.join(save_dir, 'gmm.pkl')
        loaded = False
        gmm = None
        if os.path.exists(file_name) and load_clusters:
            print("Loaded GP clusters from last GP training session. File: {}".format(file_name))
            gmm = joblib.load(file_name)
            if gmm.n_components != n_clusters:
                print("The loaded GP does not coincide with the number of set clusters: Found {} but requested"
                      "is {}".format(gmm.n_components, n_clusters))
            else:
                loaded = True
        if not loaded:
            gmm = GaussianMixture(n_clusters).fit(x_pruned)
        clusters = gmm.predict_proba(x_pruned)
        centroids = gmm.means_

        if not load_clusters and n_clusters > 1:
            safe_mknode_recursive(save_dir, 'gmm.pkl', overwrite=True)
            joblib.dump(gmm, file_name)

        index_aux = np.arange(0, clusters.shape[0])
        cluster_agency = {}

        # Add to the points corresponding to each cluster the points that correspond to it with top 2 probability,
        # if this probability is high
        for cluster in range(n_clusters):
            top_1 = np.argmax(clusters, 1)
            clusters_ = copy.deepcopy(clusters)
            clusters_[index_aux, top_1] *= 0
            top_2 = np.argmax(clusters_, 1)
            idx = np.where(top_1 == cluster)[0]
            idx = np.append(idx, np.where((top_2 == cluster) * (clusters_[index_aux, top_2] > 0.2))[0])
            cluster_agency[cluster] = idx

        # Only works in len(x_features) >= 3
        if visualize_data:
            x_aux = self.get_x(pruned=False)
            y_aux = self.get_y(pruned=False)
            visualize_data_distribution(x_aux, y_aux, cluster_agency, x_pruned, y_pruned)

        self.cluster_agency = cluster_agency
        self.centroids = centroids


def restore_gp_regressors(pre_trained_models):
    """
    :param pre_trained_models: A dictionary with all the GP models to be restored in 'models'.
    :return: The GP ensemble restored from the models.
    :rtype: GPEnsemble
    """

    gp_reg_ensemble = GPEnsemble()
    # TODO: Deprecate compatibility mode with old models.
    if all(item in list(pre_trained_models.keys()) for item in ["x_features", "u_features"]):
        x_features = pre_trained_models["x_features"]
        u_features = pre_trained_models["u_features"]
    else:
        x_features = u_features = None

    if isinstance(pre_trained_models['models'][0], dict):
        pre_trained_gp_reg = {}
        for _, model_dict in enumerate(pre_trained_models['models']):
            if x_features is not None:
                gp_reg = npGPRegression(x_features, u_features, model_dict["reg_dim"])
            else:
                gp_reg = npGPRegression(model_dict["x_features"], model_dict["u_features"], model_dict["reg_dim"])
            gp_reg.load(model_dict)
            if model_dict["reg_dim"] not in pre_trained_gp_reg.keys():
                pre_trained_gp_reg[model_dict["reg_dim"]] = [gp_reg]
            else:
                pre_trained_gp_reg[model_dict["reg_dim"]] += [gp_reg]

        # Add the GP's in a per-output-dim basis into the Ensemble
        for key in np.sort(list(pre_trained_gp_reg.keys())):
            gp_reg_ensemble.add_model(pre_trained_gp_reg[key])
    else:
        raise NotImplementedError("Cannot load this format of GP model.")

    return gp_reg_ensemble


def read_dataset(name, train_split, sim_options):
    """
    Attempts to read a dataset given its name and its metadata.
    :param name: Name of the dataset
    :param train_split: Whether to load the training split (True) or the test split (False)
    :param sim_options: Dictionary of metadata of the dataset to be read.
    :return:
    """
    data_file = get_data_dir_and_file(name, training_split=train_split, params=sim_options, read_only=True)
    if data_file is None:
        raise FileNotFoundError
    rec_file_dir, rec_file_name = data_file
    rec_file = os.path.join(rec_file_dir, rec_file_name)
    ds = pd.read_csv(rec_file)

    return ds


def world_to_body_velocity_mapping(state_sequence):
    """

    :param state_sequence: N x 13 state array, where N is the number of states in the sequence.
    :return: An N x 13 sequence of states, but where the velocities (assumed to be in positions 7, 8, 9) have been
    rotated from world to body frame. The rotation is made using the quaternion in positions 3, 4, 5, 6.
    """

    p, q, v_w, w = separate_variables(state_sequence)
    v_b = []
    for i in range(len(q)):
        v_b.append(v_dot_q(v_w[i], quaternion_inverse(q[i])))
    v_b = np.stack(v_b)
    return np.concatenate((p, q, v_b, w), 1)