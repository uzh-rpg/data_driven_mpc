""" Miscellaneous utility functions.

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
import math
import json
import errno
import shutil
import joblib
import random
import pyquaternion
import numpy as np
import casadi as cs
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.interpolate.interpolate import interp1d
from config.configuration_parameters import DirectoryConfig as GPConfig
import matplotlib.pyplot as plt
import xml.etree.ElementTree as XMLtree


def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print('Error while removing directory: {0}'.format(directory))


def safe_mknode_recursive(destiny_dir, node_name, overwrite):
    safe_mkdir_recursive(destiny_dir)
    if overwrite and os.path.exists(os.path.join(destiny_dir, node_name)):
        os.remove(os.path.join(destiny_dir, node_name))
    if not os.path.exists(os.path.join(destiny_dir, node_name)):
        os.mknod(os.path.join(destiny_dir, node_name))
        return False
    return True


def jsonify(array):
    if isinstance(array, np.ndarray):
        return array.tolist()
    if isinstance(array, list):
        return array
    return array


def undo_jsonify(array):
    x = []
    for elem in array:
        a = elem.split('[')[1].split(']')[0].split(',')
        a = [float(num) for num in a]
        x = x + [a]
    return np.array(x)


def get_data_dir_and_file(ds_name, training_split, params, read_only=False):
    """
    Returns the directory and file name where to store the next simulation-based dataset.
    """

    # Data folder directory
    rec_file_dir = GPConfig.DATA_DIR

    # Look for file
    f = []
    d = []
    for (dir_path, dir_names, file_names) in os.walk(rec_file_dir):
        f.extend(file_names)
        d.extend(dir_names)
        break

    split = "train" if training_split else "test"

    # Check if current dataset folder already exists. Create it otherwise
    if ds_name in d and os.path.exists(os.path.join(rec_file_dir, ds_name, split)):
        dataset_instances = []
        for (_, _, file_names) in os.walk(os.path.join(rec_file_dir, ds_name, split)):
            dataset_instances.extend([file.split('.csv')[0] for file in file_names])
    else:
        if read_only:
            return None
        safe_mkdir_recursive(os.path.join(rec_file_dir, ds_name, split))
        dataset_instances = []

    json_file_name = os.path.join(rec_file_dir, 'metadata.json')
    if not f:
        if read_only:
            return None
        # Metadata file not found -> make new one
        with open(json_file_name, 'w') as json_file:
            ds_instance_name = "dataset_001"
            json.dump({ds_name: {split: {ds_instance_name: params}}}, json_file, indent=4)
    else:
        # Metadata file existing
        with open(json_file_name) as json_file:
            metadata = json.load(json_file)

        # Check if current dataset name with data split exists in metadata:
        if ds_name in metadata.keys() and split in metadata[ds_name].keys():

            existing_instance_idx = -1
            for i, instance in enumerate(dataset_instances):
                if metadata[ds_name][split][instance] == params:
                    existing_instance_idx = i
                    if not read_only:
                        print("This configuration already exists in the dataset with the same name.")

            if existing_instance_idx == -1:

                if read_only:
                    return None

                if dataset_instances:
                    # Dataset exists but this is a new instance
                    existing_instances = [int(instance.split("_")[1]) for instance in dataset_instances]
                    max_instance_number = max(existing_instances)
                    ds_instance_name = "dataset_" + str(max_instance_number + 1).zfill(3)

                    # Add to metadata dictionary the new instance
                    metadata[ds_name][split][ds_instance_name] = params

                else:
                    # Edge case where, for some error, there was something added to the metadata file but no actual
                    # datasets were recorded. Remove entries from metadata and add them again.
                    ds_instance_name = "dataset_001"
                    metadata[ds_name][split] = {}
                    metadata[ds_name][split][ds_instance_name] = params

            else:
                # Dataset exists and there is an instance with the same configuration
                ds_instance_name = dataset_instances[existing_instance_idx]

        else:
            if read_only:
                return None

            ds_instance_name = "dataset_001"
            safe_mkdir_recursive(os.path.join(rec_file_dir, ds_name, split))

            # Add to metadata dictionary the new instance
            if ds_name in metadata.keys():
                metadata[ds_name][split] = {ds_instance_name: params}
            else:
                metadata[ds_name] = {split: {ds_instance_name: params}}

        if not read_only:
            with open(json_file_name, 'w') as json_file:
                json.dump(metadata, json_file, indent=4)

    return os.path.join(rec_file_dir, ds_name, split), ds_instance_name + '.csv'


def get_model_dir_and_file(model_options):
    directory = os.path.join(GPConfig.SAVE_DIR, str(model_options["git"]), str(model_options["model_name"]))

    model_params = model_options["params"]
    file_name = ''
    model_vars = list(model_params.keys())
    model_vars.sort()
    for i, param in enumerate(model_vars):
        if i > 0:
            file_name += '__'
        file_name += 'no_' if not model_params[param] else ''
        file_name += param

    return directory, file_name


def load_pickled_models(directory='', file_name='', model_options=None):
    """
    Loads a pre-trained model from the specified directory, contained in a given pickle filename. Otherwise, if
    the model_options dictionary is given, use its contents to reconstruct the directory location of the pre-trained
    model fitting the requirements.

    :param directory: directory where the model file is located
    :param file_name: file name of the pre-trained model
    :param model_options: dictionary with the keys: "noisy" (bool), "drag" (bool), "git" (string), "training_samples"
    (int), "payload" (bool).

    :return: a dictionary with the recovered models from the pickle files.
    """

    if model_options is not None:
        directory, file_name = get_model_dir_and_file(model_options)

    try:
        pickled_files = os.listdir(directory)
    except FileNotFoundError:
        return None

    loaded_models = []

    try:
        for file in pickled_files:
            if not file.startswith(file_name) and file != 'feats.csv':
                continue
            if '.pkl' not in file and '.csv' not in file:
                continue
            if '.pkl' in file:
                loaded_models.append(joblib.load(os.path.join(directory, file)))

    except IsADirectoryError:
        raise FileNotFoundError("Tried to load file from directory %s, but it was not found." % directory)

    if loaded_models is not None:
        if loaded_models:
            pre_trained_models = {"models": loaded_models}
        else:
            pre_trained_models = None
    else:
        pre_trained_models = None

    return pre_trained_models


def interpol_mse(t_1, x_1, t_2, x_2, n_interp_samples=1000):
    if np.all(t_1 == t_2):
        return np.mean(np.sqrt(np.sum((x_1 - x_2) ** 2, axis=1)))

    assert x_1.shape[1] == x_2.shape[1]

    t_min = max(t_1[0], t_2[0])
    t_max = min(t_1[-1], t_2[-1])

    t_interp = np.linspace(t_min, t_max, n_interp_samples)
    err = np.zeros((n_interp_samples, x_1.shape[1]))

    for dim in range(x_1.shape[1]):
        x1_interp = interp1d(t_1, x_1[:, dim], kind='cubic')
        x2_interp = interp1d(t_2, x_2[:, dim], kind='cubic')

        x1_sample = x1_interp(t_interp)
        x2_sample = x2_interp(t_interp)

        err[:, dim] = x1_sample - x2_sample

    return np.mean(np.sqrt(np.sum(err ** 2, axis=1)))


def euclidean_dist(x, y, thresh=None):
    """
    Measures the euclidean distance between points x and y. If a threshold value is provided, this function returns True
    if the calculated distance is smaller than the threshold, or False otherwise. If no threshold is provided, this
    function returns the computed distance.

    :param x: n-dimension point
    :param y: n-dimension point
    :param thresh: threshold
    :type thresh: float
    :return: If thresh is not None: whether x and y are closer to each other than the threshold.
    If thresh is None: the distance between x and y
    """

    dist = np.sqrt(np.sum((x - y) ** 2))

    if thresh is None:
        return dist

    return dist < thresh


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return np.array([qw, qx, qy, qz])


def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]


def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """

    if isinstance(q, np.ndarray):
        # if (q == np.zeros(4)).all():
        #     q = np.array([1, 0, 0, 0])
        q_norm = np.sqrt(np.sum(q ** 2))
    else:
        q_norm = cs.sqrt(cs.sumsqr(q))
    return 1 / q_norm * q


def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return cs.mtimes(rot_mat, v)


def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])

    else:
        rot_mat = cs.vertcat(
            cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
            cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

    return rot_mat


def q_dot_q(q, r):
    """
    Applies the rotation of quaternion r to quaternion q. In order words, rotates quaternion q by r. Quaternion format:
    wxyz.

    :param q: 4-length numpy array or CasADi MX. Initial rotation
    :param r: 4-length numpy array or CasADi MX. Applied rotation
    :return: The quaternion q rotated by r, with the same format as in the input.
    """

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rw, rx, ry, rz = r[0], r[1], r[2], r[3]

    t0 = rw * qw - rx * qx - ry * qy - rz * qz
    t1 = rw * qx + rx * qw - ry * qz + rz * qy
    t2 = rw * qy + rx * qz + ry * qw - rz * qx
    t3 = rw * qz - rx * qy + ry * qx + rz * qw

    if isinstance(q, np.ndarray):
        return np.array([t0, t1, t2, t3])
    else:
        return cs.vertcat(t0, t1, t2, t3)


def rotation_matrix_to_quat(rot):
    """
    Calculate a quaternion from a 3x3 rotation matrix.

    :param rot: 3x3 numpy array, representing a valid rotation matrix
    :return: a quaternion corresponding to the 3D rotation described by the input matrix. Quaternion format: wxyz
    """

    q = pyquaternion.Quaternion(matrix=rot)
    return np.array([q.w, q.x, q.y, q.z])


def undo_quaternion_flip(q_past, q_current):
    """
    Detects if q_current generated a quaternion jump and corrects it. Requires knowledge of the previous quaternion
    in the series, q_past
    :param q_past: 4-dimensional vector representing a quaternion in wxyz form.
    :param q_current: 4-dimensional vector representing a quaternion in wxyz form. Will be corrected if it generates
    a flip wrt q_past.
    :return: q_current with the flip removed if necessary
    """

    if np.sqrt(np.sum((q_past - q_current) ** 2)) > np.sqrt(np.sum((q_past + q_current) ** 2)):
        return -q_current
    return q_current


def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3D vector (PAMPC version)

    :param v: 3D numpy vector or CasADi MX
    :return: the corresponding skew-symmetric matrix of v with the same data type as v
    """

    if isinstance(v, np.ndarray):
        return np.array([[0, -v[0], -v[1], -v[2]],
                         [v[0], 0, v[2], -v[1]],
                         [v[1], -v[2], 0, v[0]],
                         [v[2], v[1], -v[0], 0]])

    return cs.vertcat(
        cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0))


def decompose_quaternion(q):
    """
    Decomposes a quaternion into a z rotation and an xy rotation
    :param q: 4-dimensional numpy array of CasADi MX (format qw, qx, qy, qz)
    :return: two 4-dimensional arrays (same format as input), where the first contains the xy rotation and the second
    the z rotation, in quaternion forms.
    """

    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, cs.MX):
        qz = unit_quat(cs.vertcat(w, 0, 0, z))
    else:
        qz = unit_quat(np.array([w, 0, 0, z]))
    qxy = q_dot_q(q, quaternion_inverse(qz))

    return qxy, qz


def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return cs.vertcat(w, -x, -y, -z)


def rotation_matrix_to_euler(r_mat):
    sy = math.sqrt(r_mat[0, 0] * r_mat[0, 0] + r_mat[1, 0] * r_mat[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(r_mat[2, 1], r_mat[2, 2])
        y = math.atan2(-r_mat[2, 0], sy)
        z = math.atan2(r_mat[1, 0], r_mat[0, 0])
    else:
        x = math.atan2(-r_mat[1, 2], r_mat[1, 1])
        y = math.atan2(-r_mat[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def prune_dataset(x, y, x_cap, bins, thresh, plot, labels=None):
    """
    Prunes the collected model error dataset with two filters. First, remove values where the input values (velocities)
    exceed 10. Second, create an histogram for each of the three axial velocity errors (y) with the specified number of
    bins and remove any data where the total amount of samples in that bin is less than the specified threshold ratio.
    :param x: dataset of input GP features (velocities). Dimensions N x n (N entries and n dimensions)
    :param y: dataset of errors. Dimensions N x m (N entries and m dimensions)
    :param x_cap: remove values from dataset if x > x_cap or x < -x_cap
    :param bins: number of bins used for histogram
    :param thresh: threshold ratio below which data from that bin will be removed
    :param plot: make a plot of the pruning
    :param labels: Labels to use for the plot
    :return: The indices kept after the pruning
    """

    n_bins = bins
    original_length = x.shape[0]

    plot_bins = []
    if plot:
        plt.figure()
        for i in range(y.shape[1]):
            plt.subplot(y.shape[1] + 1, 1, i + 1)
            h, bins = np.histogram(y[:, i], bins=n_bins)
            plot_bins.append(bins)
            plt.bar(bins[:-1], h, np.ones_like(h) * (bins[1] - bins[0]), align='edge', label='discarded')
            if labels is not None:
                plt.ylabel(labels[i])

    pruned_idx_unique = np.zeros(0, dtype=int)

    # Prune velocities (max axial velocity = x_cap m/s).
    if x_cap is not None:
        for i in range(x.shape[1]):
            pruned_idx = np.where(np.abs(x[:, i]) > x_cap)[0]
            pruned_idx_unique = np.unique(np.append(pruned_idx, pruned_idx_unique))

    # Prune by error histogram dimension wise (discard bins with less than 1% of the data)
    for i in range(y.shape[1]):
        h, bins = np.histogram(y[:, i], bins=n_bins)
        for j in range(len(h)):
            if h[j] / np.sum(h) < thresh:
                pruned_idx = np.where(np.logical_and(bins[j + 1] >= y[:, i], y[:, i] >= bins[j]))
                pruned_idx_unique = np.unique(np.append(pruned_idx, pruned_idx_unique))

    y_norm = np.sqrt(np.sum(y ** 2, 1))

    # Prune by error histogram norm
    h, error_bins = np.histogram(y_norm, bins=n_bins)
    h = h / np.sum(h)
    if plot:
        plt.subplot(y.shape[1] + 1, 1, y.shape[1] + 1)
        plt.bar(error_bins[:-1], h, np.ones_like(h) * (error_bins[1] - error_bins[0]), align='edge', label='discarded')

    for j in range(len(h)):
        if h[j] / np.sum(h) < thresh:
            pruned_idx = np.where(np.logical_and(error_bins[j + 1] >= y_norm, y_norm >= error_bins[j]))
            pruned_idx_unique = np.unique(np.append(pruned_idx, pruned_idx_unique))

    y = np.delete(y, pruned_idx_unique, axis=0)

    if plot:
        for i in range(y.shape[1]):
            plt.subplot(y.shape[1] + 1, 1, i + 1)
            h, bins = np.histogram(y[:, i], bins=plot_bins[i])
            plt.bar(bins[:-1], h, np.ones_like(h) * (bins[1] - bins[0]), align='edge', label='kept')
            plt.legend()

        plt.subplot(y.shape[1] + 1, 1, y.shape[1] + 1)
        h, bins = np.histogram(np.sqrt(np.sum(y ** 2, 1)), bins=error_bins)
        h = h / sum(h)
        plt.bar(bins[:-1], h, np.ones_like(h) * (bins[1] - bins[0]), align='edge', label='kept')
        plt.ylabel('Error norm')

    kept_idx = np.delete(np.arange(0, original_length), pruned_idx_unique)
    return kept_idx


def distance_maximizing_points_1d(points, n_train_points, dense_gp=None):
    """
    Heuristic function for sampling training points in 1D (one input feature and one output prediction dimensions)
    :param points: dataset points for the current cluster. Array of shape Nx1
    :param n_train_points: Integer. number of training points to sample.
    :param dense_gp: A GP object to sample the points from, or None of the points will be taken directly from the data.
    :return:
    """

    closest_points = np.zeros(n_train_points, dtype=int if dense_gp is None else float)

    if dense_gp is not None:
        n_train_points -= 1

    # Fit histogram in data with as many bins as the number of training points
    a, b = np.histogram(points, bins=n_train_points)
    hist_indices = np.digitize(points, b) - 1

    # Pick as training value the median or mean value of each bin
    for i in range(n_train_points):
        bin_values = points[np.where(hist_indices == i)]
        if len(bin_values) < 1:
            closest_points[i] = np.random.choice(np.arange(len(points)), 1)
            continue
        if divmod(len(bin_values), 2)[1] == 0:
            bin_values = bin_values[:-1]

        if dense_gp is None:
            # If no dense GP, sample median points in each bin from training set
            bin_median = np.median(bin_values)
            median_point_id = np.where(points == bin_median)[0]
            if len(median_point_id) > 1:
                closest_points[i] = median_point_id[0]
            else:
                closest_points[i] = median_point_id
        else:
            # If with GP, sample mean points in each bin from GP
            bin_mean = np.min(bin_values)
            closest_points[i] = bin_mean

    if dense_gp is not None:
        # Add dimension axis 0
        closest_points[-1] = np.max(points)
        closest_points = closest_points[np.newaxis, :]

    return closest_points


def distance_maximizing_points_2d(points, n_train_points, dense_gp, plot=False):
    if n_train_points > 30:
        n_clusters = max(int(n_train_points / 10), 30)
        n_samples = int(np.floor(n_train_points / n_clusters))
    else:
        n_clusters = n_train_points
        n_samples = 1

    kmeans = KMeans(n_clusters).fit_predict(points)

    closest_points = []
    for i in range(n_clusters):
        closest_points += np.random.choice(np.where(kmeans == i)[0], n_samples).tolist()

    # Remove exceeding points
    for _ in range(len(closest_points) - n_train_points):
        rnd_item = random.choice(closest_points)
        closest_points.remove(rnd_item)
    closest_points = np.array(closest_points)

    if plot:
        plt.figure()
        for i in range(n_clusters):
            cluster_points = points[np.where(kmeans == i)]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1])
        plt.scatter(points[closest_points, 0], points[closest_points, 1], marker='*', facecolors='w',
                    edgecolors='k', s=100, label='selected')
        plt.legend()
        plt.show()

    if dense_gp is None:
        return closest_points

    closest_points = points[closest_points].T
    return closest_points


def distance_maximizing_points(x_values, center, n_train_points=7, dense_gp=None, plot=False):
    if x_values.shape[1] == 1:
        return distance_maximizing_points_1d(x_values, n_train_points, dense_gp)

    if x_values.shape[1] >= 2:
        return distance_maximizing_points_2d(x_values, n_train_points, dense_gp, plot)

    # Compute PCA of data to find variability maximizing axes
    pca = PCA(n_components=3)
    pca.fit(x_values)
    pca_axes = pca.components_
    data_center = center

    # Apply PCA transformation
    points_pca = (x_values - data_center).dot(pca_axes.T)
    center = (center - data_center).dot(pca_axes.T)

    # Compute the corners of the cube containing the data in the PCA space
    p_min = center - (center - np.min(points_pca, 0))
    p_max = (np.max(points_pca, 0) - center) + center

    centroids = np.array([[center[0], center[1], center[2]]])

    pyramids = np.array([[p_max[0], center[1], center[2]], [center[0], p_max[1], center[2]],
                         [center[0], center[1], p_max[2]], [p_min[0], center[1], center[2]],
                         [center[0], p_min[1], center[2]], [center[0], center[1], p_min[2]]])

    cuboid = np.array([[p_max[0], p_max[1], p_max[2]], [p_max[0], p_max[1], p_min[2]],
                       [p_max[0], p_min[1], p_max[2]], [p_max[0], p_min[1], p_min[2]],
                       [p_min[0], p_max[1], p_max[2]], [p_min[0], p_max[1], p_min[2]],
                       [p_min[0], p_min[1], p_max[2]], [p_min[0], p_min[1], p_min[2]]])

    if n_train_points >= 15:
        centroids = np.concatenate((centroids, pyramids, cuboid), axis=0)
    elif n_train_points >= 9:
        centroids = np.concatenate((centroids, cuboid), axis=0)
    elif n_train_points >= 7:
        centroids = np.concatenate((centroids, pyramids), axis=0)
    else:
        centroids = centroids

    if dense_gp is None:

        closest_points = np.ones(centroids.shape[0], dtype=int) * -1

        # Find the closest points to all centroids
        for i in range(centroids.shape[0]):
            centroid = centroids[i, :]
            dist = np.sqrt(np.sum((points_pca - centroid) ** 2, 1))
            closest_point = int(np.argmin(dist))

            # Assert no repeated points. If repeated, find next closest one and check again
            while closest_point in closest_points:
                dist[closest_point] = np.inf
                closest_point = int(np.argmin(dist))
            closest_points[i] = closest_point

        # Convert centroids to data space
        centroids_ = centroids.dot(pca_axes) + data_center

        if not plot:
            return closest_points

        fig = plt.figure()

        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(points_pca[:, 0], points_pca[:, 1], points_pca[:, 2], 'b', label='data')

        ax.scatter(centroids[0, 0], centroids[0, 1], centroids[0, 2], s=50, label='center')
        ax.scatter(centroids[1:, 0], centroids[1:, 1], centroids[1:, 2], s=50, label='centroids')
        ax.scatter(points_pca[closest_points, 0], points_pca[closest_points, 1], points_pca[closest_points, 2],
                   s=50, label='selected')
        ax.set_title('PCA space')
        ax.legend()

        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(x_values[:, 0], x_values[:, 1], x_values[:, 2], 'b', label='data')

        closest_points_x = x_values[closest_points]
        ax.scatter(centroids_[0, 0], centroids_[0, 1], centroids_[0, 2], s=50, label='center')
        ax.scatter(centroids_[1:, 0], centroids_[1:, 1], centroids_[1:, 2], s=50, label='centroids')
        ax.scatter(closest_points_x[:, 0], closest_points_x[:, 1], closest_points_x[:, 2], s=50, label='selected')
        ax.set_title('Data space')
        ax.legend()
        plt.show()

        # Returns the indices of the closest points from the dataset to the ideal ones
        return closest_points

    else:

        # TODO: Use GP covariance for sampling
        # Convert centroids to data space
        centroids_ = centroids.dot(pca_axes) + data_center

        return centroids_.T


def sample_random_points(points, used_idx, points_to_sample, dense_gp=None):
    bins = min(10, int(len(points) / 10))
    bins = max(bins, 2)

    # Add remaining points as random points
    free_points = np.arange(0, points.shape[0], 1)
    gp_i_free_points = np.delete(free_points, used_idx)
    n_samples = min(points_to_sample, len(gp_i_free_points))

    # Compute histogram of data
    a, b = np.histogramdd(points[gp_i_free_points, :], bins)
    assignments = [np.minimum(np.digitize(points[gp_i_free_points, j], bins=b[j]) - 1, bins - 1)
                   for j in range(points.shape[1])]

    # Compute probability distribution based on inverse histogram
    probs = np.max(a) - a[tuple(assignments)]
    probs = probs / sum(probs)

    try:
        gp_i_free_points = np.random.choice(gp_i_free_points, n_samples, p=probs, replace=False)
    except ValueError:
        print('a')
    used_idx = np.append(used_idx, gp_i_free_points)

    return used_idx


def parse_xacro_file(xacro):
    """
    Reads a .xacro file describing a robot for Gazebo and returns a dictionary with its properties.
    :param xacro: full path of .xacro file to read
    :return: a dictionary of robot attributes
    """

    tree = XMLtree.parse(xacro)

    attrib_dict = {}

    for node in tree.getroot().getchildren():
        # Get attributes
        attributes = node.attrib

        if 'value' in attributes.keys():
            attrib_dict[attributes['name']] = attributes['value']

        if node.getchildren():
            try:
                attrib_dict[attributes['name']] = [child.attrib for child in node.getchildren()]
            except:
                continue

    return attrib_dict


def make_bx_matrix(x_dims, y_feats):
    """
    Generates the Bx matrix for the GP augmented MPC.

    :param x_dims: dimensionality of the state vector
    :param y_feats: array with the indices of the state vector x that are augmented by the GP regressor
    :return: The Bx matrix to map the GP output to the state space.
    """

    bx = np.zeros((x_dims, len(y_feats)))
    for i in range(len(y_feats)):
        bx[y_feats[i], i] = 1

    return bx


def make_bz_matrix(x_dims, u_dims, x_feats, u_feats):
    """
    Generates the Bz matrix for the GP augmented MPC.
    :param x_dims: dimensionality of the state vector
    :param u_dims: dimensionality of the input vector
    :param x_feats: array with the indices of the state vector x used to make the first part of the GP feature vector z
    :param u_feats: array with the indices of the input vector u used to make the second part of the GP feature vector z
    :return:  The Bz matrix to map from input x and u features to the z feature vector.
    """

    bz = np.zeros((len(x_feats), x_dims))
    for i in range(len(x_feats)):
        bz[i, x_feats[i]] = 1
    bzu = np.zeros((len(u_feats), u_dims))
    for i in range(len(u_feats)):
        bzu[i, u_feats[i]] = 1
    bz = np.concatenate((bz, np.zeros((len(x_feats), u_dims))), axis=1)
    bzu = np.concatenate((np.zeros((len(u_feats), x_dims)), bzu), axis=1)
    bz = np.concatenate((bz, bzu), axis=0)
    return bz


def quaternion_state_mse(x, x_ref, mask):
    """
    Calculates the MSE of the 13-dimensional state (p_xyz, q_wxyz, v_xyz, r_xyz) wrt. the reference state. The MSE of
    the quaternions are treated axes-wise.

    :param x: 13-dimensional state
    :param x_ref: 13-dimensional reference state
    :param mask: 12-dimensional masking for weighted MSE (p_xyz, q_xyz, v_xyz, r_xyz)
    :return: the mean squared error of both
    """

    q_error = q_dot_q(x[3:7], quaternion_inverse(x_ref[3:7]))
    e = np.concatenate((x[:3] - x_ref[:3], q_error[1:], x[7:10] - x_ref[7:10], x[10:] - x_ref[10:]))

    return np.sqrt((e * np.array(mask)).dot(e))


def separate_variables(traj):
    """
    Reshapes a trajectory into expected format.

    :param traj: N x 13 array representing the reference trajectory
    :return: A list with the components: Nx3 position trajectory array, Nx4 quaternion trajectory array, Nx3 velocity
    trajectory array, Nx3 body rate trajectory array
    """

    p_traj = traj[:, :3]
    a_traj = traj[:, 3:7]
    v_traj = traj[:, 7:10]
    r_traj = traj[:, 10:]
    return [p_traj, a_traj, v_traj, r_traj]
