""" Executes aggressive maneuvers for collecting flight data on the Simplified Simulator to later train models on.

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
import sys
import time
import copy
import argparse
import itertools

import pandas as pd
import numpy as np
import casadi as cs

from src.utils.visualization import draw_drone_simulation, initialize_drone_plotter
from src.experiments.comparative_experiment import prepare_quadrotor_mpc
from src.utils.utils import safe_mknode_recursive, jsonify, euclidean_dist, get_data_dir_and_file
from config.configuration_parameters import SimpleSimConfig


def get_record_file_and_dir(record_dict_template, recording_options, simulation_setup, overwrite=True):
    dataset_name = recording_options["dataset_name"]
    training_split = recording_options["training_split"]

    # Directory and file name for data recording
    rec_file_dir, rec_file_name = get_data_dir_and_file(dataset_name, training_split, simulation_setup)

    overwritten = safe_mknode_recursive(rec_file_dir, rec_file_name, overwrite=overwrite)

    rec_dict = copy.deepcopy(record_dict_template)
    rec_file = os.path.join(rec_file_dir, rec_file_name)
    if overwrite or (not overwrite and not overwritten):
        for key in rec_dict.keys():
            rec_dict[key] = jsonify(rec_dict[key])

        df = pd.DataFrame(rec_dict)
        df.to_csv(rec_file, index=False, header=True)

        rec_dict = copy.deepcopy(record_dict_template)

    return rec_dict, rec_file


def make_record_dict(state_dim):
    blank_recording_dict = {
            "state_in": np.zeros((0, state_dim)),
            "state_ref": np.zeros((0, state_dim)),
            "error": np.zeros((0, state_dim)),
            "input_in": np.zeros((0, 4)),
            "state_out": np.zeros((0, state_dim)),
            "state_pred": np.zeros((0, state_dim)),
            "timestamp": np.zeros((0, 1)),
            "dt": np.zeros((0, 1)),
    }
    return blank_recording_dict


def check_out_data(rec_dict, state_final, x_pred, w_opt, dt):
    rec_dict["dt"] = np.append(rec_dict["dt"], dt)
    rec_dict["input_in"] = np.append(rec_dict["input_in"], w_opt[np.newaxis, :4], axis=0)
    rec_dict["state_out"] = np.append(rec_dict["state_out"], state_final, 0)

    if x_pred is not None:
        err = state_final - x_pred
        rec_dict["error"] = np.append(rec_dict["error"], err, axis=0)
        rec_dict["state_pred"] = np.append(rec_dict["state_pred"], x_pred[np.newaxis, :], axis=0)

    return rec_dict


def sample_random_target(x_current, world_radius, aggressive=True):
    """
    Creates a new target point to reach.
    :param x_current: current position of the quad. Only used if aggressive=True
    :param world_radius: radius of the area where points are sampled from
    :param aggressive: if aggressive=True, points will be sampled away from the current position. If False, then points
    will be sampled uniformly from the world area.
    :return: new sampled target point. A 3-dimensional numpy array.
    """

    if aggressive:

        # Polar 3D coordinates
        theta = np.random.uniform(0, 2 * np.pi, 1)
        psi = np.random.uniform(0, 2 * np.pi, 1)
        r = 1 * world_radius + np.random.uniform(-0.5, 0.5, 1) * world_radius

        # Transform to cartesian
        x = r * np.sin(theta) * np.cos(psi)
        y = r * np.sin(theta) * np.sin(psi)
        z = r * np.cos(theta)

        return x_current + np.array([x, y, z]).reshape((1, 3))

    else:
        return np.random.uniform(-world_radius, world_radius, (1, 3))


def main(model_options, recording_options, simulation_options, parameters):

    world_radius = 3

    if parameters["initial_state"] is None:
        initial_state = [0.0, 0.0, 0.0] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0]
    else:
        initial_state = parameters["initial_state"]
    sim_starting_pos = initial_state
    quad_current_state = sim_starting_pos

    if parameters["preset_targets"] is not None:
        targets = parameters["preset_targets"]
    else:
        targets = sample_random_target(np.array(initial_state[:3]), world_radius,
                                       aggressive=recording_options["recording"])

    quad_mpc = prepare_quadrotor_mpc(simulation_options, **model_options, t_horizon=0.5,
                                     q_mask=np.array([1, 1, 1, 0.01, 0.01, 0.01, 1, 1, 1, 0, 0, 0]))

    # Recover some necessary variables from the MPC object
    my_quad = quad_mpc.quad
    n_mpc_nodes = quad_mpc.n_nodes
    t_horizon = quad_mpc.t_horizon
    simulation_dt = quad_mpc.simulation_dt
    reference_over_sampling = 3
    control_period = t_horizon / (n_mpc_nodes * reference_over_sampling)

    my_quad.set_state(quad_current_state)

    # Real time plot params
    n_forward_props = n_mpc_nodes
    plot_sim_traj = False

    x_pred = None
    w_opt = None
    initial_guess = None

    # The optimization should be faster or equal than the duration of the optimization time step
    assert control_period <= t_horizon / n_mpc_nodes

    state = quad_mpc.get_state()

    # ####### Recording mode code ####### #
    recording = recording_options["recording"]
    state_dim = state.shape[0]
    blank_recording_dict = make_record_dict(state_dim)

    # Get recording file and directory
    if recording:
        if parameters["real_time_plot"]:
            parameters["real_time_plot"] = False
            print("Turned off real time plot during recording mode.")

        rec_dict, rec_file = get_record_file_and_dir(blank_recording_dict, recording_options, simulation_options)

    else:
        rec_dict = rec_file = None

    # Generate necessary art pack for real time plot
    if parameters["real_time_plot"]:
        real_time_art_pack = initialize_drone_plotter(n_props=n_forward_props, quad_rad=my_quad.x_f,
                                                      world_rad=world_radius)
    else:
        real_time_art_pack = None

    start_time = time.time()
    simulation_time = 0.0

    # Simulation tracking stuff
    targets_reached = np.array([False for _ in targets])
    quad_trajectory = np.array(quad_current_state).reshape(1, -1)

    n_iteration_count = 0

    print("Targets reached: ", end='')
    # All targets loop
    while False in targets_reached and (time.time() - start_time) < parameters["max_sim_time"]:
        current_target_i = np.where(targets_reached == False)[0][0]
        current_target = targets[current_target_i]
        current_target_reached = False

        quad_target_state = [list(current_target), [1, 0, 0, 0], [0, 0, 0], [0, 0, 0]]
        model_ind = quad_mpc.set_reference(quad_target_state)

        # Provide an initial guess without the uncertainty prop.
        if initial_guess is None:
            initial_guess = quad_mpc.optimize(use_model=model_ind)
            initial_guess = quad_mpc.reshape_input_sequence(initial_guess)

        # MPC loop
        while not current_target_reached and (time.time() - start_time) < parameters["max_sim_time"]:

            # Emergency recovery (quad controller gone out of control lol)
            if np.any(state[7:10] > 14) or n_iteration_count > 100:
                n_iteration_count = 0
                my_quad.set_state(list(itertools.chain.from_iterable(quad_target_state)))

            state = quad_mpc.get_state()
            if recording:
                rec_dict["state_in"] = np.append(rec_dict["state_in"], state.T, 0)
                rec_dict["timestamp"] = np.append(rec_dict["timestamp"], time.time() - start_time)
                stacked_ref = np.array(list(itertools.chain.from_iterable(quad_target_state)))[np.newaxis, :]
                rec_dict["state_ref"] = np.append(rec_dict["state_ref"], stacked_ref, 0)
                if simulation_time != 0.0:
                    rec_dict = check_out_data(rec_dict, state.T, x_pred, w_opt, simulation_time)

            simulation_time = 0.0

            # Optimize control input to reach pre-set target
            w_opt, x_pred_horizon = quad_mpc.optimize(use_model=model_ind, return_x=True)
            if np.any(w_opt > (my_quad.max_input_value + 0.01)) or np.any(w_opt < (my_quad.min_input_value - 0.01)):
                print("MPC constraints were violated")
            initial_guess = quad_mpc.reshape_input_sequence(w_opt)
            # Save initial guess for future optimization. It is a time-shift of the current optimized variables
            initial_guess = np.array(cs.vertcat(initial_guess[1:, :], cs.DM.zeros(4).T))

            # Select first input (one for each motor) - MPC applies only first optimized input to the plant
            ref_u = np.squeeze(np.array(w_opt[:4]))

            if recording:
                # Integrate first input. Will be used as nominal model prediction during next save
                x_pred, _ = quad_mpc.forward_prop(np.squeeze(state), w_opt=w_opt[:4],
                                                  t_horizon=control_period, use_gp=False)
                x_pred = x_pred[-1, :]

            if parameters["real_time_plot"]:
                prop_params = {"x_0": np.squeeze(state), "w_opt": w_opt, "use_model": model_ind, "t_horizon": t_horizon}
                x_int, _ = quad_mpc.forward_prop(**prop_params, use_gp=False)
                if plot_sim_traj:
                    x_sim = quad_mpc.simulate_plant(quad_mpc.reshape_input_sequence(w_opt))
                else:
                    x_sim = None
                draw_drone_simulation(real_time_art_pack, quad_trajectory, my_quad, targets,
                                      targets_reached, x_sim, x_int, x_pred_horizon, follow_quad=False)

            while simulation_time < control_period:

                # Simulation runtime (inner loop)
                simulation_time += simulation_dt
                quad_mpc.simulate(ref_u)

                quad_current_state = quad_mpc.get_state()

                # Target is reached
                if euclidean_dist(current_target[0:3], quad_current_state[0:3, 0], thresh=0.05):
                    print("*", end='')
                    sys.stdout.flush()
                    n_iteration_count = 0

                    # Check out data immediately as new target will be optimized in next step
                    if recording and len(rec_dict['state_in']) > len(rec_dict['input_in']):
                        x_pred, _ = quad_mpc.forward_prop(np.squeeze(state), w_opt=w_opt[:4], t_horizon=simulation_time,
                                                          use_gp=False)
                        x_pred = x_pred[-1, :]
                        rec_dict = check_out_data(rec_dict, quad_mpc.get_state().T, x_pred, w_opt, simulation_time)

                    # Reset optimization time -> Ask for new optimization for next target in next dt
                    simulation_time = 0.0

                    # Mark current target as reached
                    current_target_reached = True
                    targets_reached[current_target_i] = True

                    # Remove initial guess
                    initial_guess = None

                    if parameters["preset_targets"] is None:
                        new_target = sample_random_target(quad_current_state[:3], world_radius, aggressive=recording)
                        targets = np.append(targets, new_target, axis=0)
                        targets_reached = np.append(targets_reached, False)

                    # Reset PID integral and past errors
                    quad_mpc.reset()

                    break

            n_iteration_count += 1

            if parameters["real_time_plot"]:
                quad_trajectory = np.append(quad_trajectory, quad_current_state.T, axis=0)
                if len(quad_trajectory) > 300:
                    quad_trajectory = np.delete(quad_trajectory, 0, 0)

        # Current target was reached. Remove incomplete recordings
        if recording:
            if len(rec_dict['state_in']) > len(rec_dict['input_in']):
                rec_dict["state_in"] = rec_dict["state_in"][:-1]
                rec_dict["timestamp"] = rec_dict["timestamp"][:-1]
                rec_dict["state_ref"] = rec_dict["state_ref"][:-1]

            for key in rec_dict.keys():
                rec_dict[key] = jsonify(rec_dict[key])

            df = pd.DataFrame(rec_dict)
            df.to_csv(rec_file, index=True, mode='a', header=False)

            rec_dict = copy.deepcopy(blank_recording_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_version", type=str, default="",
                        help="Version to load for the regression models. By default it is an 8 digit git hash.")

    parser.add_argument("--model_name", type=str, default="",
                        help="Name of the regression model within the specified <model_version> folder.")

    parser.add_argument("--model_type", type=str, default="gp", choices=["gp", "rdrv"],
                        help="Type of regression model (GP or RDRv linear)")

    parser.add_argument("--recording", dest="recording", action="store_true",
                        help="Set to True to enable recording mode.")
    parser.set_defaults(recording=False)

    parser.add_argument("--dataset_name", type=str, default="simplified_sim_dataset",
                        help="Name for the generated dataset.")

    parser.add_argument("--simulation_time", type=float, default=300,
                        help="Total duration of the simulation in seconds.")

    args = parser.parse_args()

    np.random.seed(0)

    acados_config = {
        "solver_type": "SQP",
        "terminal_cost": True
    }

    run_options = {
        "model_options": {
            "version": args.model_version,
            "name": args.model_name,
            "reg_type": args.model_type,
            "quad_name": "my_quad"
        },
        "recording_options": {
            "recording": args.recording,
            "dataset_name": args.dataset_name,
            "training_split": True,
        },
        "simulation_options": SimpleSimConfig.simulation_disturbances,
        "parameters": {
            "real_time_plot": SimpleSimConfig.custom_sim_gui,
            "max_sim_time": args.simulation_time,
            "preset_targets": None,
            "initial_state": None,
            "acados_options": acados_config
        }
    }

    main(**run_options)
