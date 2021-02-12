""" Tracks a specified trajectory on the simplified simulator using the data-augmented MPC.

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
import json
import argparse
import numpy as np
from tqdm import tqdm
from src.utils.utils import separate_variables
from src.utils.quad_3d_opt_utils import get_reference_chunk
from src.utils.trajectories import loop_trajectory, lemniscate_trajectory, check_trajectory
from src.utils.visualization import initialize_drone_plotter, draw_drone_simulation, trajectory_tracking_results
from src.experiments.comparative_experiment import prepare_quadrotor_mpc
from config.configuration_parameters import SimpleSimConfig


def main(args):
    params = {
        "version": args.model_version,
        "name": args.model_name,
        "reg_type": args.model_type,
        "quad_name": "my_quad"
    }

    # Load the disturbances for the custom offline simulator.
    simulation_options = SimpleSimConfig.simulation_disturbances

    debug_plots = SimpleSimConfig.pre_run_debug_plots
    tracking_results_plot = SimpleSimConfig.result_plots
    sim_gui = SimpleSimConfig.custom_sim_gui

    quad_mpc = prepare_quadrotor_mpc(simulation_options, **params)

    # Recover some necessary variables from the MPC object
    my_quad = quad_mpc.quad
    n_mpc_nodes = quad_mpc.n_nodes
    t_horizon = quad_mpc.t_horizon
    simulation_dt = quad_mpc.simulation_dt
    reference_over_sampling = 5
    control_period = t_horizon / (n_mpc_nodes * reference_over_sampling)

    if args.trajectory == "loop":
        reference_traj, reference_timestamps, reference_u = loop_trajectory(
            my_quad, control_period, radius=args.trajectory_radius, z=1, lin_acc=args.acceleration, clockwise=True,
            yawing=False, v_max=args.max_speed, map_name=None, plot=debug_plots)

    elif args.trajectory == "lemniscate":
        reference_traj, reference_timestamps, reference_u = lemniscate_trajectory(
            my_quad, control_period, radius=args.trajectory_radius, z=1, lin_acc=args.acceleration, clockwise=True,
            yawing=False, v_max=args.max_speed, map_name=None, plot=debug_plots)

    else:
        raise ValueError("Unknown trajectory {}. Options are `lemniscate` and `loop`".format(args.trajectory))

    if not check_trajectory(reference_traj, reference_u, reference_timestamps, debug_plots):
        return

    # Set quad initial state equal to the initial reference trajectory state
    quad_current_state = reference_traj[0, :].tolist()
    my_quad.set_state(quad_current_state)

    real_time_artists = None
    if sim_gui:
        # Initialize real time plot stuff
        world_radius = np.max(np.abs(reference_traj[:, :2])) * 1.2
        real_time_artists = initialize_drone_plotter(n_props=n_mpc_nodes, quad_rad=my_quad.length,
                                                     world_rad=world_radius, full_traj=reference_traj)

    ref_u = reference_u[0, :]
    quad_trajectory = np.zeros((len(reference_timestamps), len(quad_current_state)))
    u_optimized_seq = np.zeros((len(reference_timestamps), 4))

    # Sliding reference trajectory initial index
    current_idx = 0

    # Measure the MPC optimization time
    mean_opt_time = 0.0

    # Measure total simulation time
    total_sim_time = 0.0

    print("\nRunning simulation...")
    for current_idx in tqdm(range(reference_traj.shape[0])):

        quad_current_state = my_quad.get_state(quaternion=True, stacked=True)

        quad_trajectory[current_idx, :] = np.expand_dims(quad_current_state, axis=0)

        # ##### Optimization runtime (outer loop) ##### #
        # Get the chunk of trajectory required for the current optimization
        ref_traj_chunk, ref_u_chunk = get_reference_chunk(
            reference_traj, reference_u, current_idx, n_mpc_nodes, reference_over_sampling)

        # Set the reference for the OCP
        model_ind = quad_mpc.set_reference(x_reference=separate_variables(ref_traj_chunk), u_reference=ref_u_chunk)

        # Optimize control input to reach pre-set target
        t_opt_init = time.time()
        w_opt, x_pred = quad_mpc.optimize(use_model=model_ind, return_x=True)
        mean_opt_time += time.time() - t_opt_init

        # Select first input (one for each motor) - MPC applies only first optimized input to the plant
        ref_u = np.squeeze(np.array(w_opt[:4]))
        u_optimized_seq[current_idx, :] = np.reshape(ref_u, (1, -1))

        if len(quad_trajectory) > 0 and sim_gui and current_idx > 0:
            draw_drone_simulation(real_time_artists, quad_trajectory[:current_idx, :], my_quad, targets=None,
                                  targets_reached=None, pred_traj=x_pred, x_pred_cov=None)

        simulation_time = 0.0

        # ##### Simulation runtime (inner loop) ##### #
        while simulation_time < control_period:
            simulation_time += simulation_dt
            total_sim_time += simulation_dt
            quad_mpc.simulate(ref_u)

    u_optimized_seq[current_idx, :] = np.reshape(ref_u, (1, -1))

    quad_current_state = my_quad.get_state(quaternion=True, stacked=True)
    quad_trajectory[-1, :] = np.expand_dims(quad_current_state, axis=0)
    u_optimized_seq[-1, :] = np.reshape(ref_u, (1, -1))

    # Average elapsed time per optimization
    mean_opt_time = mean_opt_time / current_idx * 1000
    tracking_rmse = np.mean(np.sqrt(np.sum((reference_traj[:, :3] - quad_trajectory[:, :3]) ** 2, axis=1)))

    if tracking_results_plot:
        v_max = np.max(reference_traj[:, 7:10])

        with_gp = ' + GP ' if quad_mpc.gp_ensemble is not None else ' - GP '
        title = r'$v_{max}$=%.2f m/s | RMSE: %.4f m | %s ' % (v_max, float(tracking_rmse), with_gp)
        trajectory_tracking_results(reference_timestamps, reference_traj, quad_trajectory, reference_u, u_optimized_seq,
                                    title)

    v_max_abs = np.max(np.sqrt(np.sum(reference_traj[:, 7:10] ** 2, 1)))

    print("\n:::::::::::::: SIMULATION SETUP ::::::::::::::\n")
    print("Simulation: Applied disturbances: ")
    print(json.dumps(simulation_options))
    if quad_mpc.gp_ensemble is not None:
        print("\nModel: Using GP regression model: ", params["version"] + '/' + params["name"])
    elif quad_mpc.rdrv is not None:
        print("\nModel: Using RDRv regression model: ", params["version"] + '/' + params["name"])
    else:
        print("\nModel: No regression model loaded")

    print("\nReference: Executed trajectory", '`' + args.trajectory + '`', "with a peak axial velocity of",
          args.max_speed, "m/s, and a maximum speed of %2.3f m/s" % v_max_abs)

    print("\n::::::::::::: SIMULATION RESULTS :::::::::::::\n")
    print("Mean optimization time: %.3f ms" % mean_opt_time)
    print("Tracking RMSE: %.4f m\n" % tracking_rmse)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_version", type=str, default="",
                        help="Version to load for the regression models. By default it is an 8 digit git hash.")

    parser.add_argument("--model_name", type=str, default="",
                        help="Name of the regression model within the specified <model_version> folder.")

    parser.add_argument("--model_type", type=str, default="gp", choices=["gp", "rdrv"],
                        help="Type of regression model (GP or RDRv linear)")

    parser.add_argument("--trajectory", type=str, default="loop", choices=["loop", "lemniscate"],
                        help='path to other necessary data files (eg. vocabularies)')

    parser.add_argument("--max_speed", type=float, default=8,
                        help="Maximum axial speed executed during the flight in m/s. For the `loop` trajectory, "
                             "velocities are feasible up to 14 m/s, and for the `lemniscate` up to 8 m/s")

    parser.add_argument("--acceleration", type=float, default=1,
                        help="Acceleration of the reference trajectory. Higher accelerations will shorten the execution"
                             "time of the tracking")

    parser.add_argument("--trajectory_radius", type=float, default=5, help="Radius of the reference trajectories")
    input_arguments = parser.parse_args()

    main(input_arguments)
