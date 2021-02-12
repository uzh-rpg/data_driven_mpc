""" Gaussian Process custom implementation for the data-augmented MPC.

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


import numpy as np
import casadi as cs
import joblib

from tqdm import tqdm
from operator import itemgetter
from numpy.linalg import inv, cholesky, lstsq
from numpy.random import mtrand
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist, squareform

from src.utils.utils import safe_mknode_recursive, make_bz_matrix


class CustomKernelFunctions:

    def __init__(self, kernel_func, params=None):

        self.params = params
        self.kernel_type = kernel_func

        if self.kernel_type == 'squared_exponential':
            if params is None:
                self.params = {'l': [1.0], 'sigma_f': 1.0}
            self.kernel = self.squared_exponential_kernel
        else:
            raise NotImplementedError("only squared_exponential is supported")

        self.theta = self.get_trainable_parameters()

    def __call__(self, x_1, x_2):
        return self.kernel(x_1, x_2)

    def __str__(self):
        if self.kernel_type == 'squared_exponential':
            len_scales = np.reshape(self.params['l'], -1)
            len_scale_str = '['
            for i in range(len(len_scales)):
                len_scale_str += '%.3f, ' % len_scales[i] if i < len(len_scales) - 1 else '%.3f' % len_scales[i]
            len_scale_str += ']'
            summary = '%.3f' % self.params['sigma_f']
            summary += '**2*RBF(length_scale=' + len_scale_str + ')'
            return summary

        else:
            raise NotImplementedError("only squared_exponential is supported")

    def get_trainable_parameters(self):
        trainable_params = []
        if self.kernel_type == 'squared_exponential':
            trainable_params += \
                np.reshape(np.squeeze(self.params['l']), -1).tolist() if hasattr(self.params['l'], "__len__") \
                else [self.params['l']]
            trainable_params += [self.params['sigma_f']]
        return trainable_params

    @staticmethod
    def _check_length_scale(x, length_scale):
        length_scale = np.squeeze(length_scale).astype(float)
        if np.ndim(length_scale) > 1:
            raise ValueError("length_scale cannot be of dimension greater than 1")
        if np.ndim(length_scale) == 1 and x.shape[1] != length_scale.shape[0]:
            raise ValueError("Anisotropic kernel must have the same number of dimensions as data (%d!=%d)"
                             % (length_scale.shape[0], x.shape[1]))
        return length_scale

    def squared_exponential_kernel(self, x_1, x_2=None):
        """
        Anisotropic (diagonal length-scale) matrix squared exponential kernel. Computes a covariance matrix from points
        in x_1 and x_2.

        Args:
            x_1: Array of m points (m x d).
            x_2: Array of n points (n x d).

        Returns:
            Covariance matrix (m x n).
        """

        if isinstance(x_2, cs.MX):
            return self._squared_exponential_kernel_cs(x_1, x_2)

        # Length scale parameter
        len_scale = self.params['l'] if 'l' in self.params.keys() else 1.0

        # Vertical variation parameter
        sigma_f = self.params['sigma_f'] if 'sigma_f' in self.params.keys() else 1.0

        x_1 = np.atleast_2d(x_1)
        length_scale = self._check_length_scale(x_1, len_scale)
        if x_2 is None:
            dists = pdist(x_1 / length_scale, metric='sqeuclidean')
            k = sigma_f * np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            k = squareform(k)
            np.fill_diagonal(k, 1)
        else:
            dists = cdist(x_1 / length_scale, x_2 / length_scale, metric='sqeuclidean')
            k = sigma_f * np.exp(-.5 * dists)

        return k

    def _squared_exponential_kernel_cs(self, x_1, x_2):
        """
        Symbolic implementation of the anisotropic squared exponential kernel
        :param x_1: Array of m points (m x d).
        :param x_2: Array of n points (m x d).
        :return: Covariance matrix (m x n).
        """

        # Length scale parameter
        len_scale = self.params['l'] if 'l' in self.params.keys() else 1.0
        # Vertical variation parameter
        sigma_f = self.params['sigma_f'] if 'sigma_f' in self.params.keys() else 1.0

        if x_1.shape != x_2.shape and x_2.shape[0] == 1:
            tiling_ones = cs.MX.ones(x_1.shape[0], 1)
            d = x_1 - cs.mtimes(tiling_ones, x_2)
            dist = cs.sum2(d ** 2 / cs.mtimes(tiling_ones, cs.MX(len_scale ** 2).T))
        else:
            d = x_1 - x_2
            dist = cs.sum1(d ** 2 / cs.MX(len_scale ** 2))

        return sigma_f * cs.SX.exp(-.5 * dist)

    def diff(self, z, z_train):
        """
        Computes the symbolic differentiation of the kernel function, evaluated at point z and using the training
        dataset z_train. This function implements equation (80) from overleaf document, without the c^{v_x} vector,
        and for all the partial derivatives possible (m), instead of just one.

        :param z: evaluation point. Symbolic vector of length m
        :param z_train: training dataset. Symbolic matrix of shape n x m

        :return: an m x n matrix, which is the derivative of the exponential kernel function evaluated at point z
        against the training dataset.
        """

        if self.kernel_type != 'squared_exponential':
            raise NotImplementedError

        len_scale = self.params['l'] if len(self.params['l']) > 0 else self.params['l'] * cs.MX.ones(z_train.shape[1])
        len_scale = np.atleast_2d(len_scale ** 2)

        # Broadcast z vector to have the shape of z_train (tile z to to the number of training points n)
        z_tile = cs.mtimes(cs.MX.ones(z_train.shape[0], 1), z.T)

        # Compute k_zZ. Broadcast it to shape of z_tile and z_train, i.e. by the number of variables in z.
        k_zZ = cs.mtimes(cs.MX.ones(z_train.shape[1], 1), self.__call__(z_train, z.T).T)

        return - k_zZ * (z_tile - z_train).T / cs.mtimes(cs.MX.ones(z_train.shape[0], 1), len_scale).T


class CustomGPRegression:

    def __init__(self, x_features, u_features, reg_dim, mean=None, y_mean=None, kernel=None, sigma_n=1e-8,
                 n_restarts=1):
        """
        :param x_features: list of indices for the quadrotor state-derived features
        :param u_features: list of indices for the input state-derived features
        :param reg_dim: state dimension that this regressor is meant to be used for.
        :param mean: prior mean value
        :param y_mean: average y value for data normalization
        :param kernel: Kernel Function object
        :param sigma_n: noise sigma value
        :param n_restarts: number of optimization re-initializations
        """

        if kernel is None:
            kernel = CustomKernelFunctions('squared_exponential')

        # Avoid non-invertible error
        assert sigma_n != 0.0

        # For inference time
        self.x_features = x_features
        self.u_features = u_features
        self.reg_dim = reg_dim

        self.kernel_ = kernel
        self.kernel_type = kernel.kernel_type

        # Noise variance prior
        self.sigma_n = sigma_n

        # GP center of local feature space
        self.mean = mean
        self.y_mean = y_mean

        # Pre-computed training data kernel
        self._K = np.zeros((0, 0))
        self._K_inv = np.zeros((0, 0))
        self._K_inv_y = np.zeros((0, ))

        # Training dataset memory
        self._x_train = np.zeros((0, 0))
        self._y_train = np.zeros((0, ))

        # CasADi symbolic equivalents
        self._K_cs = None
        self._K_inv_cs = None
        self._K_inv_y_cs = None
        self._x_train_cs = None
        self._y_train_cs = None

        self.sym_pred = None
        self.sym_jacobian_dz = None

        self.n_restarts = n_restarts

    @property
    def kernel(self):
        return self.kernel_

    @kernel.setter
    def kernel(self, ker):
        self.kernel_ = ker

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, k):
        self._K = k
        self._K_cs = cs.DM(k)

    @property
    def K_inv(self):
        return self._K_inv

    @K_inv.setter
    def K_inv(self, k):
        self._K_inv = k
        self._K_inv_cs = cs.DM(k)

    @property
    def K_inv_y(self):
        return self._K_inv_y

    @K_inv_y.setter
    def K_inv_y(self, k):
        self._K_inv_y = k
        self._K_inv_y_cs = cs.DM(k)

    @property
    def x_train(self):
        return self._x_train

    @x_train.setter
    def x_train(self, k):
        self._x_train = k
        self._x_train_cs = cs.DM(k)

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, k):
        self._y_train = k
        self._y_train_cs = cs.DM(k)

    def log_marginal_likelihood(self, theta):

        l_params = np.exp(theta[:-1])
        sigma_f = np.exp(theta[-1])
        sigma_n = self.sigma_n

        kernel = CustomKernelFunctions(self.kernel_type, params={'l': l_params, 'sigma_f': sigma_f})
        k_train = kernel(self.x_train, self.x_train) + sigma_n ** 2 * np.eye(len(self.x_train))
        l_mat = cholesky(k_train)
        nll = np.sum(np.log(np.diagonal(l_mat))) + \
            0.5 * self.y_train.T.dot(lstsq(l_mat.T, lstsq(l_mat, self.y_train, rcond=None)[0], rcond=None)[0]) + \
            0.5 * len(self.x_train) * np.log(2 * np.pi)
        return nll

    def _nll(self, x_train, y_train):
        """
        Returns a numerically stable function implementation of the negative log likelihood using the cholesky
        decomposition of the kernel matrix. http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section 2.2,
        Algorithm 2.1.
        :param x_train: Array of m points (m x d).
        :param y_train: Array of m points (m x 1)
        :return: negative log likelihood (scalar) computing function
        """

        def nll_func(theta):

            l_params = np.exp(theta[:-2])
            sigma_f = np.exp(theta[-2])
            sigma_n = np.exp(theta[-1])

            kernel = CustomKernelFunctions(self.kernel_type, params={'l': l_params, 'sigma_f': sigma_f})
            k_train = kernel(x_train, x_train) + sigma_n ** 2 * np.eye(len(x_train))
            l_mat = cholesky(k_train)
            nll = np.sum(np.log(np.diagonal(l_mat))) + \
                0.5 * y_train.T.dot(lstsq(l_mat.T, lstsq(l_mat, y_train, rcond=None)[0], rcond=None)[0]) + \
                0.5 * len(x_train) * np.log(2 * np.pi)
            return nll

        return nll_func

    def _constrained_minimization(self, x_train, y_train, x_0, bounds):
        try:
            res = minimize(self._nll(x_train, y_train), x0=x_0, bounds=bounds, method='L-BFGS-B')
            return np.exp(res.x), res.fun
        except np.linalg.LinAlgError:
            return x_0, np.inf

    def fit(self, x_train, y_train):
        """
        Fit a GP regressor to the training dataset by minimizing the negative log likelihood of the training data

        :param x_train: Array of m points (m x d).
        :param y_train: Array of m points (m x 1)
        """

        initial_guess = self.kernel.get_trainable_parameters()
        initial_guess += [self.sigma_n]
        initial_guess = np.array(initial_guess)

        bounds = [(1e-5, 1e1) for _ in range(len(initial_guess) - 1)]
        bounds = bounds + [(1e-8, 1e0)]
        log_bounds = np.log(tuple(bounds))

        y_train -= self.y_mean

        optima = [self._constrained_minimization(x_train, y_train, initial_guess, log_bounds)]

        if self.n_restarts > 1:
            random_state = mtrand._rand
            for iteration in range(self.n_restarts - 1):
                theta_initial = random_state.uniform(log_bounds[:, 0], log_bounds[:, 1])
                optima.append(self._constrained_minimization(x_train, y_train, theta_initial, log_bounds))

        lml_values = list(map(itemgetter(1), optima))
        theta_opt = optima[int(np.argmin(lml_values))][0]

        # Update kernel with new parameters
        l_new = theta_opt[:-2]
        sigma_f_new = theta_opt[-2]
        self.sigma_n = theta_opt[-1]
        self.kernel = CustomKernelFunctions(self.kernel_type, params={'l': l_new, 'sigma_f': sigma_f_new})

        # Pre-compute kernel matrices
        self.K = self.kernel(x_train, x_train) + self.sigma_n ** 2 * np.eye(len(x_train))
        self.K_inv = inv(self.K)
        self.K_inv_y = self.K_inv.dot(y_train)

        # Update training dataset points
        self.x_train = x_train
        self.y_train = y_train

        self.compute_gp_jac()

    def compute_gp_jac(self):

        self.sym_jacobian_dz = self._linearized_state_estimate()

    def eval_gp_jac(self, z):

        if self.sym_jacobian_dz is None:
            self.compute_gp_jac()

        return self.sym_jacobian_dz(z)

    def _linearized_state_estimate(self):
        """
        Computes the symbolic linearization of the GP prediction expected state with respect to the inputs of the GP
        itself.

        :return: a CasADi function that computes the linearized GP prediction estimate wrt the input features of the
        GP regressor itself. The output of the function is a vector of shape m, where m is the number of regression
        features.
        """

        if self.kernel_type != 'squared_exponential':
            raise NotImplementedError

        # Symbolic variable for input state
        z = cs.MX.sym('z', self.x_train.shape[1])

        # Compute the kernel derivative:
        dgpdz = cs.mtimes(self.kernel.diff(z, self._x_train_cs), self._K_inv_y_cs)

        return cs.Function('f', [z], [dgpdz], ['z'], ['dgpdz'])

    def predict(self, x_test, return_std=False, return_cov=False):
        """
        Computes the sufficient statistics of the GP posterior predictive distribution
        from m training data X_train and Y_train and n new inputs X_s.

        Args:
            x_test: test input locations (n x d).
            return_std: boolean - return a standard deviation vector of size n
            return_cov: boolean - return a covariance vector of size n (sqrt of standard deviation)

        Returns:
            Posterior mean vector (n) and covariance diagonal or standard deviation vectors (n) if also requested.
        """

        # Ensure at least n=1
        x_test = np.atleast_2d(x_test) if isinstance(x_test, np.ndarray) else x_test

        if isinstance(x_test, cs.MX):
            return self._predict_sym(x_test=x_test, return_std=return_std, return_cov=return_cov)

        if isinstance(x_test, cs.DM):
            x_test = np.array(x_test).T

        k_s = self.kernel(x_test, self.x_train)
        k_ss = self.kernel(x_test, x_test) + 1e-8 * np.eye(len(x_test))

        # Posterior mean value
        mu_s = k_s.dot(self.K_inv_y) + self.y_mean

        # Posterior covariance
        cov_s = k_ss - k_s.dot(self.K_inv).dot(k_s.T)
        std_s = np.sqrt(np.diag(cov_s))

        if not return_std and not return_cov:
            return mu_s

        # Return covariance
        if return_cov:
            return mu_s, std_s ** 2

        # Return standard deviation
        return mu_s, std_s

    def _predict_sym(self, x_test, return_std=False, return_cov=False):
        """
        Computes the GP posterior mean and covariance at a given a test sample using CasADi symbolics.
        :param x_test: vector of size mx1, where m is the number of features used for the GP prediction

        :return: the posterior mean (scalar) and covariance (scalar).
        """

        k_s = self.kernel(self._x_train_cs, x_test.T)

        # Posterior mean value
        mu_s = cs.mtimes(k_s.T, self._K_inv_y_cs) + self.y_mean

        if not return_std and not return_cov:
            return {'mu': mu_s}

        k_ss = self.kernel(x_test, x_test) + 1e-8 * cs.MX.eye(x_test.shape[1])

        # Posterior covariance
        cov_s = k_ss - cs.mtimes(cs.mtimes(k_s.T, self._K_inv_cs), k_s)
        cov_s = cs.diag(cov_s)

        if return_std:
            return {'mu': mu_s, 'std': np.sqrt(cov_s)}

        return {'mu': mu_s, 'cov': cov_s}

    def sample_y(self, x_test, n_samples=3):
        """
        Sample a number of functions from the process at the given test points

        :param x_test: test input locations (n x d).
        :param n_samples: integer, number of samples to draw
        :return: the drawn samples from the gaussian process. An array of shape n x n_samples
        """

        mu, cov = self.predict(x_test, return_cov=True)

        # Draw three samples from the prior
        samples = np.random.multivariate_normal(mu.ravel(), np.diag(cov), n_samples)

        return samples.T

    def save(self, path):
        """
        Saves the current GP regressor to the specified path as a pickle file. Must be re-loaded with the function load
        :param path: absolute path to save the regressor to
        """

        saved_vars = {
            "kernel_params": self.kernel.params,
            "kernel_type": self.kernel.kernel_type,
            "x_train": self.x_train,
            "y_train": self.y_train,
            "k_inv_y": self.K_inv_y,
            "k_inv": self.K_inv,
            "sigma_n": self.sigma_n,
            "reg_dim": self.reg_dim,
            "x_features": self.x_features,
            "u_features": self.u_features,
            "mean": self.mean,
            "y_mean": self.y_mean
        }

        split_path = path.split('/')
        directory = '/'.join(split_path[:-1])
        file = split_path[-1]
        safe_mknode_recursive(directory, file, overwrite=True)

        with open(path, 'wb') as f:
            joblib.dump(saved_vars, f)

    def load(self, data_dict):
        """
        Load a pre-trained GP regressor
        :param data_dict: a dictionary with all the pre-trained matrices of the GP regressor
        """

        self.K_inv = data_dict['k_inv']
        self.K_inv_y = data_dict['k_inv_y']
        self.x_train = data_dict['x_train']
        self.y_train = data_dict['y_train']
        self.kernel_type = data_dict['kernel_type']
        self.kernel = CustomKernelFunctions(self.kernel_type, data_dict['kernel_params'])
        self.sigma_n = data_dict['sigma_n']
        self.mean = data_dict['mean'] if 'mean' in data_dict.keys() else np.array([0, 0, 0])
        self.y_mean = data_dict['y_mean'] if 'y_mean' in data_dict.keys() else np.array(0)
        self.compute_gp_jac()


class GPEnsemble:
    def __init__(self):
        """
        Sets up a GP regression ensemble. This essentially divides the prediction domain into different GP's, so that
        less training samples need to be used per GP.
        """

        self.out_dim = 0
        self.n_models_dict = {}

        # Make index to dim to make dimensions iterable
        self.dim_idx = np.zeros(0, dtype=int)

        # Dictionary of lists. Each element of the dictionary is indexed by the index of the GP output in the state
        # space, and contains a with all the GP's (one per cluster) used in that dimension.
        self.gp = {}

        # Store the centroids of all GP's
        self.gp_centroids = {}

        # Store the B_z matrices
        self.B_z_dict = {}

        # Whether the same clustering is used for all dimensions or not
        self.homogeneous = True

        # Whether the GP model has no ensembles in it (i.e. no GP has more than 1 cluster)
        self.no_ensemble = True

    @property
    def n_models(self):
        if self.homogeneous or self.no_ensemble:
            return self.n_models_dict[next(iter(self.n_models_dict))]
        return self.n_models_dict

    @property
    def B_z(self):
        return self.B_z_dict[next(iter(self.B_z_dict))] if self.homogeneous else self.B_z_dict

    def add_model(self, gp):
        """"
        :param gp: A list of n CustomGPRegression objects, where n is the number of GP's used to divide the feature
        space domain of the dimension in particular.
        :type gp: list
        """

        # Check if dimension is already "occupied" by another GP
        gp_dim = gp[0].reg_dim
        if gp_dim in self.gp.keys():
            raise ValueError("This dimension is already taken by another GP")

        self.out_dim += 1
        self.dim_idx = np.append(self.dim_idx, gp_dim)

        self.gp[gp_dim] = np.array(gp)

        # Store centroids and sort along first dimension for easier comparison
        self.gp_centroids[gp_dim] = np.array([gp_cluster.mean for gp_cluster in gp])
        sorted_cluster_idx = np.argsort(self.gp_centroids[gp_dim][:, 0])
        self.gp_centroids[gp_dim] = self.gp_centroids[gp_dim][sorted_cluster_idx]
        self.gp[gp_dim] = self.gp[gp_dim][sorted_cluster_idx]

        # Calculate if Ensemble is still homogeneous
        self.homogeneous = self.homogeneous_feature_space()

        # Check if current gp is an actual ensemble
        self.n_models_dict[gp_dim] = len(gp)
        if len(gp) > 1:
            self.no_ensemble = False

        # Pre-compute B_z matrix
        self.B_z_dict[gp_dim] = make_bz_matrix(x_dims=13, u_dims=4, x_feats=gp[0].x_features, u_feats=gp[0].u_features)

    def get_z(self, x, u, dim):
        """
        Computes the z features from the x and u vectors, and the target output dimension.
        :param x: state vector. Shape 13x1. Can be np.array or cs.MX.
        :param u: control input vector. Shape 4x1. Can be np.array or cs.MX.
        :param dim: output dimension target.
        :return: A vector of shape mx1 of the same format as inputs. m is determined by the B_z matrix for dim.
        """

        # Get input feature indices
        x_feats = self.gp[dim][0].x_features
        u_feats = self.gp[dim][0].u_features

        # Stack into a single matrix
        if isinstance(x, np.ndarray):
            z = np.concatenate((x[x_feats], u[u_feats]), axis=0)
        elif isinstance(x, cs.MX):
            z = cs.mtimes(self.B_z_dict[dim], cs.vertcat(x, u))
        else:
            raise TypeError

        return z

    def predict(self, x_test, u_test, return_std=False, return_cov=False, return_gp_id=False, return_z=False,
                progress_bar=False, gp_idx=None):
        """
        Runs GP inference. First, select the GP optimally for the test samples. Then, run inference on that GP.
        :param x_test: array of shape d x n. n is the number of test samples and d their dimension.
        :param u_test: array of shape d' x n. n is the number of test samples and d' their dimension.
        :param return_std: True if also return the standard deviation of the GP inference.
        :param return_cov: True if also return the covariance of the GP inference.
        :param return_gp_id: True if also return the id of the GP used for inference.
        :param return_z: True if also return the z features computed for inference.
        :param progress_bar: If True, a progress bar will be shown when evaluating the test data.
        :param gp_idx: Dictionary of indices with the same length as the GP output dimension indicating which GP to use
        for each one. If None, select best based on x_test.
        :type gp_idx: dict
        :return: m x n arrays, where m is the output dimension and n is the number of samples.
        """

        if return_std:
            assert not return_cov, "Can only return the std or the cov"
        if return_cov:
            assert not return_std, "Can only return the std or the cov"

        # Dictionary for function return
        outputs = {}

        # Build regression features and evaluation cluster indices for each GP output dimension
        z = {}
        gp_idx = {} if gp_idx is None else gp_idx

        if not self.homogeneous:
            for dim in self.gp.keys():

                z[dim] = self.get_z(x_test, u_test, dim)

                if dim not in gp_idx.keys():
                    # Calculate optimal GP clusters to use for each test point
                    gp_idx[dim] = self.select_gp(z=z[dim], dim=dim)
                    gp_idx[dim] = np.atleast_1d(gp_idx[dim])

        else:
            z_ = self.get_z(x_test, u_test, self.dim_idx[0])
            z = {k: v for k, v in zip(self.dim_idx, [z_] * self.out_dim)}

            if not bool(gp_idx):
                gp_idx_ = self.select_gp(z=z_, dim=self.dim_idx[0])
                gp_idx = {k: v for k, v in zip(self.dim_idx, [gp_idx_] * self.out_dim)}

        # Add stuff to output dictionary
        if return_z:
            outputs["z"] = z
        if return_gp_id:
            outputs["gp_id"] = gp_idx

        pred = []
        cov_or_std = []
        noise_prior = []

        # Test data loop
        range_data = tqdm(range(x_test.shape[1])) if progress_bar else range(x_test.shape[1])
        for j in range_data:

            pred_j = []
            cov_or_std_j = []
            noise_prior_j = []

            # Output dim loop
            for dim in self.gp.keys():
                out = self.gp[dim][gp_idx[dim][j]].predict(z[dim][:, j], return_std, return_cov)
                if not return_std and not return_cov:
                    if isinstance(out, dict):
                        pred_j += [out['mu']]
                    else:
                        pred_j += [out]
                else:
                    if isinstance(out, dict):
                        pred_j += [out['mu']]
                        cov_or_std_j += [out['cov'] if 'cov' in out.keys() else out['std']]
                    else:
                        pred_j += [out[0]]
                        cov_or_std_j += [out[1]]
                    noise_prior_j += [np.array([self.gp[dim][gp_idx[dim][j]].sigma_n])]

            pred += [pred_j]
            cov_or_std += [cov_or_std_j]
            noise_prior += [noise_prior_j]

        # Convert to CasADi symbolic or numpy matrix depending on the input type
        pred = cs.horzcat(*[cs.vertcat(*pred[i]) for i in range(x_test.shape[1])]) \
            if isinstance(x_test, cs.MX) else np.squeeze(np.array(pred)).T

        outputs["pred"] = pred

        if not return_std and not return_cov:
            return outputs

        # Convert to CasADi symbolic or numpy matrix depending on the input type
        cov_or_std = cs.horzcat(*[cs.vertcat(*cov_or_std[i]) for i in range(x_test.shape[1])]) \
            if isinstance(x_test, cs.MX) else np.squeeze(np.array(cov_or_std)).T
        noise_prior = cs.horzcat(*[cs.vertcat(*noise_prior[i]) for i in range(x_test.shape[1])]) \
            if isinstance(x_test, cs.MX) else np.squeeze(np.array(noise_prior)).T

        outputs["cov_or_std"] = cov_or_std
        outputs["noise_cov"] = noise_prior

        return outputs

    def select_gp(self, dim, x=None, u=None, z=None):
        """
        Selects the best GP's for computing inference at the given test points x for a given regression output
        dimension. This calculation is done by computing the distance of all n test points to all available GP's
        centroids and selecting the one minimizing the Euclidean distance.

        :param z: np array of shape d x n corresponding to the processed feature vector. If unknown one may call this
        method with x and u instead.
        :param x: np array of shape 13 x n corresponding to the query quadrotor states. Only necessary if z=None.
        :param u: np.array of shape 4 x n corresponding to the query quadrotor control vectors. Only necessary if
        z=None.
        :param dim: index of GP output dimension. If None, evaluate on all dimensions.
        :return: a numpy vector of length n, indicating which GP to use for every test sample x.
        """

        if dim is None:
            dim = self.dim_idx

        if isinstance(dim, np.ndarray):
            # If the ensemble is homogeneous only one evaluation is necessary
            if self.homogeneous or self.no_ensemble:
                return self.select_gp(dim[0], x, u, z)[0]

            return np.array([self.select_gp(i, x, u, z) for i in dim])

        if z is None:
            z = self.get_z(x, u, dim)
        z = np.atleast_2d(z)

        centroids = self.gp_centroids[dim]

        # Select subset of features for current dimension
        return np.argmin(np.sqrt(np.sum((z[np.newaxis, :, :] - centroids[:, :, np.newaxis]) ** 2, 1)), 0)

    def homogeneous_feature_space(self):
        if self.out_dim == 1:
            return True

        equal_clusters = True
        last_centroids = None
        for i, key in enumerate(self.gp_centroids.keys()):
            centroids = self.gp_centroids[key]
            if i == 0:
                last_centroids = centroids
                continue
            if np.any(last_centroids != centroids):
                equal_clusters = False
                break
            last_centroids = centroids

        return equal_clusters
