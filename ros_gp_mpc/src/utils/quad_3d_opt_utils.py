""" Set of utility functions for the quadrotor optimizer and simplified simulator.

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

import casadi as cs
import numpy as np
from src.quad_mpc.quad_3d import Quadrotor3D
from tqdm import tqdm


def discretize_dynamics_and_cost(t_horizon, n_points, m_steps_per_point, x, u, dynamics_f, cost_f, ind):
    """
    Integrates the symbolic dynamics and cost equations until the time horizon using a RK4 method.
    :param t_horizon: time horizon in seconds
    :param n_points: number of control input points until time horizon
    :param m_steps_per_point: number of integrations steps per control input
    :param x: 4-element list with symbolic vectors for position (3D), angle (4D), velocity (3D) and rate (3D)
    :param u: 4-element symbolic vector for control input
    :param dynamics_f: symbolic dynamics function written in CasADi symbolic syntax.
    :param cost_f: symbolic cost function written in CasADi symbolic syntax. If None, then cost 0 is returned.
    :param ind: Only used for trajectory tracking. Index of cost function to use.
    :return: a symbolic function that computes the dynamics integration and the cost function at n_control_inputs
    points until the time horizon given an initial state and
    """

    if isinstance(cost_f, list):
        # Select the list of cost functions
        cost_f = cost_f[ind * m_steps_per_point:(ind + 1) * m_steps_per_point]
    else:
        cost_f = [cost_f]

    # Fixed step Runge-Kutta 4 integrator
    dt = t_horizon / n_points / m_steps_per_point
    x0 = x
    q = 0

    for j in range(m_steps_per_point):
        k1 = dynamics_f(x=x, u=u)['x_dot']
        k2 = dynamics_f(x=x + dt / 2 * k1, u=u)['x_dot']
        k3 = dynamics_f(x=x + dt / 2 * k2, u=u)['x_dot']
        k4 = dynamics_f(x=x + dt * k3, u=u)['x_dot']
        x_out = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        x = x_out

        if cost_f and cost_f[j] is not None:
            q = q + cost_f[j](x=x, u=u)['q']

    return cs.Function('F', [x0, u], [x, q], ['x0', 'p'], ['xf', 'qf'])


def _forward_prop_core(x_0, u_seq, t_horizon, discrete_dynamics_f, dynamics_jac_f, B_x, gp_ensemble, covar, dt,
                       m_int_steps, use_model):
    """
    Propagates forward the state estimate described by the mean vector x_0 and the covariance matrix covar, and a
    sequence of inputs for the system u_seq. These inputs can either be numerical or symbolic.

    :param x_0: initial mean state of the state probability density function. Vector of length m
    :param u_seq: sequence of flattened N control inputs. I.e. vector of size N*4
    :param t_horizon: time horizon corresponding to sequence of inputs
    :param discrete_dynamics_f: symbolic function to compute the discrete dynamics of the system.
    :param dynamics_jac_f: symbolic function to compute the  dynamics jacobian of the system.
    :param B_x: Matrix to convert map from regressor output to state.
    :param gp_ensemble: The ensemble of GP's. Can be None if no GP's are used.
    :param covar: Initial covariance matrix of shape m x m, of the state probability density function.
    :param dt: Vector of N timestamps, the same length as w_opt / 2, corresponding to the time each input is applied.
    :param m_int_steps: number of intermediate integration steps per control node.
    :param use_model: The number (index) of the dynamics model to use from the available options.
    :return: The sequence of mean and covariance estimates for every corresponding input, as well as the computed
    cost for each stage.
    """

    # Reshape input sequence to a N x 4 (1D) vector. Assume control input dim = 4
    k = np.arange(int(u_seq.shape[0] / 4))
    input_sequence = cs.horzcat(u_seq[4 * k], u_seq[4 * k + 1], u_seq[4 * k + 2], u_seq[4 * k + 3])

    N = int(u_seq.shape[0] / 4)

    if dt is None:
        dt = t_horizon / N * np.ones((N, 1))

    if len(dt.shape) == 1:
        dt = np.expand_dims(dt, 1)

    # Initialize sequence of propagated states
    mu_x = [x_0]
    cov_x = [covar]
    cost_x = [0]

    for k in range(N):

        # Get current control input and current state mean and covariance
        u_k = input_sequence[k, :]
        mu_k = mu_x[k]
        sig_k = cov_x[k]

        # mu(k+1) vector from propagation equations. Pass state through nominal dynamics with GP mean augmentation if GP
        # is available. Otherwise use nominal dynamics only.
        f_func = discrete_dynamics_f(dt[k], 1, m_int_steps, k, use_gp=gp_ensemble is not None, use_model=use_model)

        fk = f_func(x0=mu_k, p=u_k)
        mu_next = fk['xf']
        stage_cost = fk['qf']

        # K(k+1) matrix from propagation equations
        K_mat = sig_k

        # Evaluate linearized dynamics at current state
        l_mat = dynamics_jac_f(mu_k, u_k) * dt[k] + np.eye(mu_k.shape[0])

        # Adjust matrices if GP model available.
        if gp_ensemble is not None:

            raise NotImplementedError  # TODO: re-implement covariance propagation with GP
            # Get subset of features for GP regressor if GP regressor available
            z_k = cs.mtimes(B_z, cs.vertcat(mu_k, u_k.T))

            # GP prediction. Stack predictions vertically along prediction dimension (e.g. vx, vy, ...)
            _, gp_covar_preds, gp_noise_prior = gp_ensemble.predict(z_k, return_cov=True, gp_idx=use_model)

            # Covariance forward propagation.
            l_mat = cs.horzcat(l_mat, B_x * dt[k])  # left-multiplying matrix
            sig_ld_comp = cs.mtimes(gp_prediction_jac(z_k, B_x, B_z, gp_ensemble, use_model), sig_k)
            K_mat = cs.vertcat(K_mat, sig_ld_comp)
            K_mat = cs.horzcat(K_mat, cs.vertcat(sig_ld_comp.T, cs.diag(gp_covar_preds + gp_noise_prior)))

        # Add next state estimate to lists
        mu_x += [mu_next]
        cov_x += [cs.mtimes(cs.mtimes(l_mat, K_mat), l_mat.T)]
        cost_x += [stage_cost]

    cov_x = [cov for cov in cov_x]
    return mu_x, cov_x, cost_x


def uncertainty_forward_propagation(x_0, u_seq, t_horizon, discrete_dynamics_f, dynamics_jac_f, B_x=None,
                                    gp_regressors=None, covar=None, dt=None, m_integration_steps=1, use_model=0):
    if covar is None:
        covar = np.zeros((len(x_0), len(x_0)))
    else:
        assert covar.shape == (len(x_0), len(x_0))

    x_0 = np.array(x_0)

    mu_x, cov_x, _ = _forward_prop_core(x_0, u_seq, t_horizon, discrete_dynamics_f, dynamics_jac_f, B_x,
                                        gp_regressors, covar, dt, m_integration_steps, use_model)

    mu_x = cs.horzcat(*mu_x)
    cov_x = cs.horzcat(*cov_x)

    mu_prop = np.array(mu_x).T
    cov_prop = np.array(cov_x).reshape((mu_prop.shape[1], mu_prop.shape[1], -1), order='F').transpose(2, 0, 1)
    return mu_prop, cov_prop


def gp_prediction_jac(z, Bx, Bz, gp_ensemble, gp_idx):
    """
    Computes the symbolic function for the Jacobian of the expected values of the joint GP regression model,
    evaluated at point z.

    :param z: A dictionary of symbolic vector at which the Jacobian must be evaluated. Each entry in the dictionary is
    indexed by the output dimension index. The dimension m_z on any given value must be the expected dimension of the
    feature inputs for the regressor.
    :param Bx: Matrix to convert map from regressor output to state.
    :param Bz: Matrix to map from (stacked) state vector and input vector to feature vector. If the gp_ensemble is not
    homogeneous, this parameter must be a dictionary specifying in each dimension which Bz matrix to use.
    :param gp_ensemble: GPEnsemble object with all the gp regressors
    :param gp_idx: which gp to compute the jacobian to from the ensemble
    :return: A Jacobian matrix of size n x m_x, where n is the number of variables regressed by the joint GP
    regressor and m_x is the dimension of the state vector.
    """

    # Figure out which variables are necessary to compute the Jacobian wrt.
    out_dims = np.matmul(Bx, np.ones((Bx.shape[1], 1)))
    if isinstance(Bz, dict):
        z_dims = {}
        for dim in Bz.keys():
            z_dims[dim] = np.matmul(Bz[dim][:, :len(out_dims)], out_dims)
    else:
        bz = np.matmul(Bz[:, :len(out_dims)], out_dims)
        z_dims = {k: v for k, v in zip(z.keys(), [bz] * len(z.keys()))}
        Bz = {k: v for k, v in zip(z.keys(), [Bz] * len(z.keys()))}

    jac = []
    for dim in gp_idx.keys():
        # Mapping from z to x
        inv_Bz = Bz[dim][:, :len(out_dims)].T

        gp = gp_ensemble.gp[dim][gp_idx[dim][0]]
        jac += [cs.mtimes(inv_Bz, gp.eval_gp_jac(z[dim]) * z_dims[dim])]

    return cs.horzcat(*jac).T


def simulate_plant(quad, w_opt, simulation_dt, simulate_func, t_horizon=None, dt_vec=None, progress_bar=False):
    """
    Given a sequence of n inputs, evaluates the simulated discrete-time plant model n steps into the future. The
    current drone state will not be changed by calling this method.
    :param quad: Quadrotor3D simulator object
    :type quad: Quadrotor3D
    :param w_opt: sequence of control n x m control inputs, where n is the number of steps and m is the
    dimensionality of a control input.
    :param simulation_dt: simulation step
    :param simulate_func: simulation function (inner loop) from the quadrotor MPC.
    :param t_horizon: time corresponding to the duration of the n control inputs. In the case that the w_opt comes
    from an MPC optimization, this parameter should be the MPC time horizon.
    :param dt_vec: a vector of timestamps, the same length as w_opt, corresponding to the total time each input is
    applied.
    :param progress_bar: boolean - whether to show a progress bar on the console or not.
    :return: the sequence of simulated quadrotor states.
    """

    # Default_parameters
    if t_horizon is None and dt_vec is None:
        raise ValueError("At least the t_horizon or dt should be provided")

    if t_horizon is None:
        t_horizon = np.sum(dt_vec)

    state_safe = quad.get_state(quaternion=True, stacked=True)

    # Compute true simulated trajectory
    total_sim_time = t_horizon
    sim_traj = []

    if dt_vec is None:
        change_control_input = np.arange(0, t_horizon, t_horizon / (w_opt.shape[0]))[1:]
        first_dt = t_horizon / w_opt.shape[0]
        sim_traj.append(quad.get_state(quaternion=True, stacked=True))
    else:
        change_control_input = np.cumsum(dt_vec)
        first_dt = dt_vec[0]

    t_start_ep = 1e-6
    int_range = tqdm(np.arange(t_start_ep, total_sim_time, simulation_dt)) if progress_bar else \
        np.arange(t_start_ep, total_sim_time, simulation_dt)

    current_ind = 0
    past_ind = 0
    for t_elapsed in int_range:
        ref_u = w_opt[current_ind, :].T
        simulate_func(ref_u)
        if t_elapsed + simulation_dt >= first_dt:
            current_ind = np.argwhere(change_control_input <= t_elapsed + simulation_dt)[-1, 0] + 1
            if past_ind != current_ind:
                sim_traj.append(quad.get_state(quaternion=True, stacked=True))
                past_ind = current_ind

    if dt_vec is None:
        sim_traj.append(quad.get_state(quaternion=True, stacked=True))
    sim_traj = np.squeeze(sim_traj)

    quad.set_state(state_safe)

    return sim_traj


def get_reference_chunk(reference_traj, reference_u, current_idx, n_mpc_nodes, reference_over_sampling):
    """
    Extracts the reference states and controls for the current MPC optimization given the over-sampled counterparts.

    :param reference_traj: The reference trajectory, which has been finely over-sampled by a factor of
    reference_over_sampling. It should be a vector of shape (Nx13), where N is the length of the trajectory in samples.
    :param reference_u: The reference controls, following the same requirements as reference_traj. Should be a vector
    of shape (Nx4).
    :param current_idx: Current index of the trajectory tracking. Should be an integer number between 0 and N-1.
    :param n_mpc_nodes: Number of MPC nodes considered in the optimization.
    :param reference_over_sampling: The over-sampling factor of the reference trajectories. Should be a positive
    integer.
    :return: Returns the chunks of reference selected for the current MPC iteration. Two numpy arrays will be returned:
        - An ((N+1)x13) array, corresponding to the reference trajectory. The first row is the state of current_idx.
        - An (Nx4) array, corresponding to the reference controls.
    """

    # Dense references
    ref_traj_chunk = reference_traj[current_idx:current_idx + (n_mpc_nodes + 1) * reference_over_sampling, :]
    ref_u_chunk = reference_u[current_idx:current_idx + n_mpc_nodes * reference_over_sampling, :]

    # Indices for down-sampling the reference to number of MPC nodes
    downsample_ref_ind = np.arange(0, min(reference_over_sampling * (n_mpc_nodes + 1), ref_traj_chunk.shape[0]),
                                   reference_over_sampling, dtype=int)

    # Sparser references (same dt as node separation)
    ref_traj_chunk = ref_traj_chunk[downsample_ref_ind, :]
    ref_u_chunk = ref_u_chunk[downsample_ref_ind[:max(len(downsample_ref_ind) - 1, 1)], :]

    return ref_traj_chunk, ref_u_chunk
