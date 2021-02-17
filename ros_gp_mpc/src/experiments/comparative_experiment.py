""" Runs the experimental setup to compare different data-learned models for the MPC on the Simplified Simulator.

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

import time
import argparse
import numpy as np

from config.configuration_parameters import SimpleSimConfig
from src.quad_mpc.quad_3d_mpc import Quad3DMPC
from src.quad_mpc.quad_3d import Quadrotor3D
from src.utils.quad_3d_opt_utils import get_reference_chunk
from src.utils.utils import load_pickled_models, interpol_mse, separate_variables
from src.utils.visualization import initialize_drone_plotter, draw_drone_simulation, trajectory_tracking_results, \
    get_experiment_files
from src.utils.visualization import mse_tracking_experiment_plot
from src.utils.trajectories import random_trajectory, lemniscate_trajectory, loop_trajectory
from src.model_fitting.rdrv_fitting import load_rdrv

global model_num


def prepare_quadrotor_mpc(simulation_options, version=None, name=None, reg_type="gp", quad_name=None,
                          t_horizon=1.0, q_diagonal=None, r_diagonal=None, q_mask=None):
    """
    Creates a Quad3DMPC for the custom simulator.
    @param simulation_options: Parameters for the Quadrotor3D object.
    @param version: loading version for the GP/RDRv model.
    @param name: name to load for the GP/RDRv model.
    @param reg_type: either `gp` or `rdrv`.
    @param quad_name: Name for the quadrotor. Default name will be used if not specified.
    @param t_horizon: Time horizon of MPC in seconds.
    @param q_diagonal: 12-dimensional diagonal of the Q matrix (p_xyz, a_xyz, v_xyz, w_xyz)
    @param r_diagonal: 4-dimensional diagonal of the R matrix (motor inputs 1-4)
    @param q_mask: State variable weighting mask (boolean). Which state variables compute towards state loss function?

    @return: A Quad3DMPC wrapper for the custom simulator.
    @rtype: Quad3DMPC
    """

    # Default Q and R matrix for LQR cost
    if q_diagonal is None:
        q_diagonal = np.array([10, 10, 10, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    if r_diagonal is None:
        r_diagonal = np.array([0.1, 0.1, 0.1, 0.1])
    if q_mask is None:
        q_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).T

    # Simulation integration step (the smaller the more "continuous"-like simulation.
    simulation_dt = 5e-4

    # Number of MPC optimization nodes
    n_mpc_nodes = 10

    # Calculate time between two MPC optimization nodes [s]
    node_dt = t_horizon / n_mpc_nodes

    # Quadrotor simulator
    my_quad = Quadrotor3D(**simulation_options)

    if version is not None and name is not None:

        load_ops = {"params": simulation_options}
        load_ops.update({"git": version, "model_name": name})

        # Load trained GP model
        if reg_type == "gp":
            pre_trained_models = load_pickled_models(model_options=load_ops)
            rdrv_d = None

        else:
            rdrv_d = load_rdrv(model_options=load_ops)
            pre_trained_models = None

    else:
        pre_trained_models = rdrv_d = None

    if quad_name is None:
        quad_name = "my_quad_" + str(globals()['model_num'])
        globals()['model_num'] += 1

    # Initialize quad MPC
    quad_mpc = Quad3DMPC(my_quad, t_horizon=t_horizon, optimization_dt=node_dt, simulation_dt=simulation_dt,
                         q_cost=q_diagonal, r_cost=r_diagonal, n_nodes=n_mpc_nodes,
                         pre_trained_models=pre_trained_models, model_name=quad_name, q_mask=q_mask, rdrv_d_mat=rdrv_d)

    return quad_mpc


def main(quad_mpc, av_speed, reference_type=None, plot=False):
    """

    :param quad_mpc:
    :type quad_mpc: Quad3DMPC
    :param av_speed:
    :param reference_type:
    :param plot:
    :return:
    """

    # Recover some necessary variables from the MPC object
    my_quad = quad_mpc.quad
    n_mpc_nodes = quad_mpc.n_nodes
    simulation_dt = quad_mpc.simulation_dt
    t_horizon = quad_mpc.t_horizon

    reference_over_sampling = 5
    mpc_period = t_horizon / (n_mpc_nodes * reference_over_sampling)

    # Choose the reference trajectory:
    if reference_type == "loop":
        # Circular trajectory
        reference_traj, reference_timestamps, reference_u = loop_trajectory(
            quad=my_quad, discretization_dt=mpc_period, radius=5, z=1, lin_acc=av_speed * 0.25, clockwise=True,
            yawing=False, v_max=av_speed, map_name=None, plot=plot)
    elif reference_type == "lemniscate":
        # Lemniscate trajectory
        reference_traj, reference_timestamps, reference_u = lemniscate_trajectory(
            quad=my_quad, discretization_dt=mpc_period, radius=5, z=1, lin_acc=av_speed * 0.25, clockwise=True,
            yawing=False, v_max=av_speed, map_name=None, plot=plot)
    else:
        # Get a random smooth position trajectory
        reference_traj, reference_timestamps, reference_u = random_trajectory(
            quad=my_quad, discretization_dt=mpc_period, seed=reference_type["random"], speed=av_speed, plot=plot)

    # Set quad initial state equal to the initial reference trajectory state
    quad_current_state = reference_traj[0, :].tolist()
    my_quad.set_state(quad_current_state)

    real_time_artists = None
    if plot:
        # Initialize real time plot stuff
        world_radius = np.max(np.abs(reference_traj[:, :2])) * 1.2
        real_time_artists = initialize_drone_plotter(n_props=n_mpc_nodes, quad_rad=my_quad.length,
                                                     world_rad=world_radius, full_traj=reference_traj)

    start_time = time.time()
    max_simulation_time = 10000

    ref_u = reference_u[0, :]
    quad_trajectory = np.zeros((len(reference_timestamps), len(quad_current_state)))
    u_optimized_seq = np.zeros((len(reference_timestamps), 4))

    # Sliding reference trajectory initial index
    current_idx = 0

    # Measure the MPC optimization time
    mean_opt_time = 0.0

    # Measure total simulation time
    total_sim_time = 0.0

    while (time.time() - start_time) < max_simulation_time and current_idx < reference_traj.shape[0]:

        quad_current_state = my_quad.get_state(quaternion=True, stacked=True)

        quad_trajectory[current_idx, :] = np.expand_dims(quad_current_state, axis=0)
        u_optimized_seq[current_idx, :] = np.reshape(ref_u, (1, -1))

        # ##### Optimization runtime (outer loop) ##### #
        # Get the chunk of trajectory required for the current optimization
        ref_traj_chunk, ref_u_chunk = get_reference_chunk(
            reference_traj, reference_u, current_idx, n_mpc_nodes, reference_over_sampling)

        model_ind = quad_mpc.set_reference(x_reference=separate_variables(ref_traj_chunk), u_reference=ref_u_chunk)

        # Optimize control input to reach pre-set target
        t_opt_init = time.time()
        w_opt, x_pred = quad_mpc.optimize(use_model=model_ind, return_x=True)
        mean_opt_time += time.time() - t_opt_init

        # Select first input (one for each motor) - MPC applies only first optimized input to the plant
        ref_u = np.squeeze(np.array(w_opt[:4]))

        if len(quad_trajectory) > 0 and plot and current_idx > 0:
            draw_drone_simulation(real_time_artists, quad_trajectory[:current_idx, :], my_quad, targets=None,
                                  targets_reached=None, pred_traj=x_pred, x_pred_cov=None)

        simulation_time = 1e-8

        # ##### Simulation runtime (inner loop) ##### #
        while simulation_time < mpc_period:
            simulation_time += simulation_dt
            total_sim_time += simulation_dt
            quad_mpc.simulate(ref_u)

        u_optimized_seq[current_idx, :] = np.reshape(ref_u, (1, -1))

        current_idx += 1

    quad_current_state = my_quad.get_state(quaternion=True, stacked=True)
    quad_trajectory[-1, :] = np.expand_dims(quad_current_state, axis=0)
    u_optimized_seq[-1, :] = np.reshape(ref_u, (1, -1))

    # Average elapsed time per optimization
    mean_opt_time /= current_idx

    rmse = interpol_mse(reference_timestamps, reference_traj[:, :3], reference_timestamps, quad_trajectory[:, :3])
    max_vel = np.max(np.sqrt(np.sum(reference_traj[:, 7:10] ** 2, 1)))

    with_gp = ' + GP ' if quad_mpc.gp_ensemble is not None else ' - GP '
    title = r'$v_{max}$=%.2f m/s | RMSE: %.4f m | %s ' % (max_vel, float(rmse), with_gp)

    if plot:
        trajectory_tracking_results(reference_timestamps, reference_traj, quad_trajectory,
                                    reference_u, u_optimized_seq, title)

    return rmse, max_vel, mean_opt_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_version", type=str, default="", nargs="+",
                        help="Versions to load for the regression models. By default it is an 8 digit git hash."
                             "Must specify the version for each model separated by spaces.")

    parser.add_argument("--model_name", type=str, default="", nargs="+",
                        help="Name of the regression models within the specified <model_version> folders. "
                             "Must specify the names for all models separated by spaces.")

    parser.add_argument("--model_type", type=str, default="", nargs="+",
                        help="Type of regression models (GP or RDRv linear). "
                             "Must be specified for all models separated by spaces.")

    parser.add_argument("--fast", dest="fast", action="store_true",
                        help="Set to True to run a fast experiment with less velocity samples.")
    parser.set_defaults(fast=False)

    input_arguments = parser.parse_args()

    globals()['model_num'] = 0

    # Trajectory options
    traj_type_vec = [{"random": 1}, "loop", "lemniscate"]
    traj_type_labels = ["Random", "Circle", "Lemniscate"]
    if input_arguments.fast:
        av_speed_vec = [[2.0, 3.5],
                        [2.0, 12.0],
                        [2.0, 12.0]]
    else:
        av_speed_vec = [[2.0, 2.7, 3.0, 3.2, 3.5],
                        [2.0, 4.5, 7.0, 9.5, 12.0],
                        [2.0, 4.5, 7.0, 9.5, 12.0]]

    # Model options
    git_list = input_arguments.model_version
    name_list = input_arguments.model_name
    type_list = input_arguments.model_type

    assert len(git_list) == len(name_list) == len(type_list)

    # Simulation options
    plot_sim = SimpleSimConfig.custom_sim_gui
    noisy_sim_options = SimpleSimConfig.simulation_disturbances
    perfect_sim_options = {"payload": False, "drag": False, "noisy": False, "motor_noise": False}
    model_vec = [
        {"simulation_options": perfect_sim_options, "model": None},
        {"simulation_options": noisy_sim_options, "model": None}]

    legends = ['perfect', 'nominal']
    for git, m_name, gp_or_rdrv in zip(git_list, name_list, type_list):
        model_vec += [{"simulation_options": noisy_sim_options,
                       "model": {"version": git, "name": m_name, "reg_type": gp_or_rdrv}}]
        legends += [gp_or_rdrv + ": " + m_name]

    y_label = "RMSE [m]"

    # Define result vectors
    mse = np.zeros((len(traj_type_vec), len(av_speed_vec[0]), len(model_vec)))
    v_max = np.zeros((len(traj_type_vec), len(av_speed_vec[0])))
    t_opt = np.zeros((len(traj_type_vec), len(av_speed_vec[0]), len(model_vec)))

    for n_train_id, model_type in enumerate(model_vec):

        if model_type["model"] is not None:
            custom_mpc = prepare_quadrotor_mpc(model_type["simulation_options"], **model_type["model"])
        else:
            custom_mpc = prepare_quadrotor_mpc(model_type["simulation_options"])

        for traj_id, traj_type in enumerate(traj_type_vec):

            for v_id, speed in enumerate(av_speed_vec[traj_id]):

                traj_params = {"av_speed": speed, "reference_type": traj_type, "plot": plot_sim}

                mse[traj_id, v_id, n_train_id], traj_v, opt_dt = main(custom_mpc, **traj_params)
                t_opt[traj_id, v_id, n_train_id] += opt_dt

                if v_max[traj_id, v_id] == 0:
                    v_max[traj_id, v_id] = traj_v

    _, err_file, v_file, t_file = get_experiment_files()
    np.save(err_file, mse)
    np.save(v_file, v_max)
    np.save(t_file, t_opt)

    # from src.utils.visualization import load_past_experiments
    # _, mse, v_max, t_opt = load_past_experiments()

    mse_tracking_experiment_plot(v_max, mse, traj_type_labels, model_vec, legends, [y_label], t_opt=t_opt, font_size=26)
