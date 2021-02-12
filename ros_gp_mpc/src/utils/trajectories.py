""" Trajectory generation functions. For the circle, lemniscate and random trajectories.

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
from src.utils.utils import undo_quaternion_flip, rotation_matrix_to_quat
from src.utils.utils import quaternion_inverse, q_dot_q
from src.utils.trajectory_generator import draw_poly, get_full_traj, fit_multi_segment_polynomial_trajectory
from src.utils.keyframe_3d_gen import random_periodical_trajectory
from config.configuration_parameters import DirectoryConfig
from src.quad_mpc.quad_3d import Quadrotor3D
import matplotlib.pyplot as plt
import os
import yaml
import json

import rospy


def check_trajectory(trajectory, inputs, tvec, plot=False):
    """

    @param trajectory:
    @param inputs:
    @param tvec:
    @param plot:
    @return:
    """

    print("Checking trajectory integrity...")

    dt = np.expand_dims(np.gradient(tvec, axis=0), axis=1)
    numeric_derivative = np.gradient(trajectory, axis=0) / dt

    errors = np.zeros((dt.shape[0], 3))

    num_bodyrates = []

    for i in range(dt.shape[0]):
        # 1) check if velocity is consistent with position
        numeric_velocity = numeric_derivative[i, 0:3]
        analytic_velocity = trajectory[i, 7:10]
        errors[i, 0] = np.linalg.norm(numeric_velocity - analytic_velocity)
        if not np.allclose(analytic_velocity, numeric_velocity, atol=1e-2, rtol=1e-2):
            print("inconsistent linear velocity")
            print(numeric_velocity)
            print(analytic_velocity)
            return False

        # 2) check if attitude is consistent with acceleration
        gravity = 9.81
        numeric_thrust = numeric_derivative[i, 7:10] + np.array([0.0, 0.0, gravity])
        numeric_thrust = numeric_thrust / np.linalg.norm(numeric_thrust)
        analytic_attitude = trajectory[i, 3:7]
        if np.abs(np.linalg.norm(analytic_attitude) - 1.0) > 1e-6:
            print("quaternion does not have unit norm!")
            print(analytic_attitude)
            print(np.linalg.norm(analytic_attitude))
            return False

        e_z = np.array([0.0, 0.0, 1.0])
        q_w = 1.0 + np.dot(e_z, numeric_thrust)
        q_xyz = np.cross(e_z, numeric_thrust)
        numeric_attitude = 0.5 * np.array([q_w] + q_xyz.tolist())
        numeric_attitude = numeric_attitude / np.linalg.norm(numeric_attitude)
        # the two attitudes can only differ in yaw --> check x,y component
        q_diff = q_dot_q(quaternion_inverse(analytic_attitude), numeric_attitude)
        errors[i, 1] = np.linalg.norm(q_diff[1:3])
        if not np.allclose(q_diff[1:3], np.zeros(2, ), atol=1e-3, rtol=1e-3):
            print("Attitude and acceleration do not match!")
            print(analytic_attitude)
            print(numeric_attitude)
            print(q_diff)
            return False

        # 3) check if bodyrates agree with attitude difference
        numeric_bodyrates = 2.0 * q_dot_q(quaternion_inverse(trajectory[i, 3:7]), numeric_derivative[i, 3:7])[1:]
        num_bodyrates.append(numeric_bodyrates)
        analytic_bodyrates = trajectory[i, 10:13]
        errors[i, 2] = np.linalg.norm(numeric_bodyrates - analytic_bodyrates)
        if not np.allclose(numeric_bodyrates, analytic_bodyrates, atol=0.05, rtol=0.05):
            print("inconsistent angular velocity")
            print(numeric_bodyrates)
            print(analytic_bodyrates)
            return False

    print("Trajectory check successful")
    print("Maximum linear velocity error: %.5f" % np.max(errors[:, 0]))
    print("Maximum attitude error: %.5f" % np.max(errors[:, 1]))
    print("Maximum angular velocity error: %.5f" % np.max(errors[:, 2]))

    if plot:
        num_bodyrates = np.stack(num_bodyrates)
        plt.figure()
        for i in range(3):
            plt.subplot(3, 2, i * 2 + 1)
            plt.plot(numeric_derivative[:, i], label='numeric')
            plt.plot(trajectory[:, 7 + i], label='analytic')
            plt.ylabel('m/s')
            if i == 0:
                plt.title("Velocity check")
            plt.legend()

        for i in range(3):
            plt.subplot(3, 2, i * 2 + 2)
            plt.plot(num_bodyrates[:, i], label='numeric')
            plt.plot(trajectory[:, 10 + i], label='analytic')
            plt.ylabel('rad/s')
            if i == 0:
                plt.title("Body rate check")
            plt.legend()
        plt.suptitle('Integrity check of reference trajectory')
        plt.show()

    return True


def minimum_snap_trajectory_generator(traj_derivatives, yaw_derivatives, t_ref, quad, map_limits, plot):
    """
    Follows the Minimum Snap Trajectory paper to generate a full trajectory given the position reference and its
    derivatives, and the yaw trajectory and its derivatives.

    :param traj_derivatives: np.array of shape 4x3xN. N corresponds to the length in samples of the trajectory, and:
        - The 4 components of the first dimension correspond to position, velocity, acceleration and jerk.
        - The 3 components of the second dimension correspond to x, y, z.
    :param yaw_derivatives: np.array of shape 2xN. N corresponds to the length in samples of the trajectory. The first
    row is the yaw trajectory, and the second row is the yaw time-derivative trajectory.
    :param t_ref: vector of length N, containing the reference times (starting from 0) for the trajectory.
    :param quad: Quadrotor3D object, corresponding to the quadrotor model that will track the generated reference.
    :type quad: Quadrotor3D
    :param map_limits: dictionary of map limits if available, None otherwise.
    :param plot: True if show a plot of the generated trajectory.
    :return: tuple of 3 arrays:
        - Nx13 array of generated reference trajectory. The 13 dimension contains the components: position_xyz,
        attitude_quaternion_wxyz, velocity_xyz, body_rate_xyz.
        - N array of reference timestamps. The same as in the input
        - Nx4 array of reference controls, corresponding to the four motors of the quadrotor.
    """

    discretization_dt = t_ref[1] - t_ref[0]
    len_traj = traj_derivatives.shape[2]

    # Add gravity to accelerations
    gravity = 9.81
    thrust = traj_derivatives[2, :, :].T + np.tile(np.array([[0, 0, 1]]), (len_traj, 1)) * gravity
    # Compute body axes
    z_b = thrust / np.sqrt(np.sum(thrust ** 2, 1))[:, np.newaxis]

    yawing = np.any(yaw_derivatives[0, :] != 0)

    rate = np.zeros((len_traj, 3))
    f_t = np.zeros((len_traj, 1))
    for i in range(len_traj):
        f_t[i, 0] = quad.mass * z_b[i].dot(thrust[i, :].T)

    if yawing:
        # yaw is defined as the projection of the body-x axis on the horizontal plane
        x_c = np.concatenate((np.cos(yaw_derivatives[0, :])[:, np.newaxis],
                              np.sin(yaw_derivatives[0, :])[:, np.newaxis],
                              np.zeros(len_traj)[:, np.newaxis]), 1)
        y_b = np.cross(z_b, x_c)
        y_b = y_b / np.sqrt(np.sum(y_b ** 2, axis=1))[:, np.newaxis]
        x_b = np.cross(y_b, z_b)

        # Rotation matrix (from body to world)
        b_r_w = np.concatenate((x_b[:, :, np.newaxis], y_b[:, :, np.newaxis], z_b[:, :, np.newaxis]), -1)
        q = []
        for i in range(len_traj):
            # Transform to quaternion
            q.append(rotation_matrix_to_quat(b_r_w[i]))
            if i > 1:
                q[-1] = undo_quaternion_flip(q[-2], q[-1])
        q = np.stack(q)

        # Compute angular rate vector
        # Total thrust acceleration must be equal to the projection of the quadrotor acceleration into the Z body axis
        a_proj = np.zeros((len_traj, 1))

        for i in range(len_traj):
            a_proj[i, 0] = z_b[i].dot(traj_derivatives[3, :, i])

        h_omega = quad.mass / f_t * (traj_derivatives[3, :, :].T - a_proj * z_b)
        for i in range(len_traj):
            rate[i, 0] = -h_omega[i].dot(y_b[i])
            rate[i, 1] = h_omega[i].dot(x_b[i])
            rate[i, 2] = -yaw_derivatives[1, i] * np.array([0, 0, 1]).dot(z_b[i])

    else:
        # new way to compute attitude:
        # https://math.stackexchange.com/questions/2251214/calculate-quaternions-from-two-directional-vectors
        e_z = np.array([[0.0, 0.0, 1.0]])
        q_w = 1.0 + np.sum(e_z * z_b, axis=1)
        q_xyz = np.cross(e_z, z_b)
        q = 0.5 * np.concatenate([np.expand_dims(q_w, axis=1), q_xyz], axis=1)
        q = q / np.sqrt(np.sum(q ** 2, 1))[:, np.newaxis]

        # Use numerical differentiation of quaternions
        q_dot = np.gradient(q, axis=0) / discretization_dt
        w_int = np.zeros((len_traj, 3))
        for i in range(len_traj):
            w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q[i, :]), q_dot[i])[1:]
        rate[:, 0] = w_int[:, 0]
        rate[:, 1] = w_int[:, 1]
        rate[:, 2] = w_int[:, 2]

        go_crazy_about_yaw = True
        if go_crazy_about_yaw:
            print("Maximum yawrate before adaption: %.3f" % np.max(np.abs(rate[:, 2])))
            q_new = q
            yaw_corr_acc = 0.0
            for i in range(1, len_traj):
                yaw_corr = -rate[i, 2] * discretization_dt
                yaw_corr_acc += yaw_corr
                q_corr = np.array([np.cos(yaw_corr_acc / 2.0), 0.0, 0.0, np.sin(yaw_corr_acc / 2.0)])
                q_new[i, :] = q_dot_q(q[i, :], q_corr)
                w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q[i, :]), q_dot[i])[1:]

            q_new_dot = np.gradient(q_new, axis=0) / discretization_dt
            for i in range(1, len_traj):
                w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q_new[i, :]), q_new_dot[i])[1:]

            q = q_new
            rate[:, 0] = w_int[:, 0]
            rate[:, 1] = w_int[:, 1]
            rate[:, 2] = w_int[:, 2]
            print("Maximum yawrate after adaption: %.3f" % np.max(np.abs(rate[:, 2])))

    # Compute inputs
    rate_dot = np.gradient(rate, axis=0) / discretization_dt

    rate_x_Jrate = np.array([(quad.J[2] - quad.J[1]) * rate[:, 2] * rate[:, 1],
                             (quad.J[0] - quad.J[2]) * rate[:, 0] * rate[:, 2],
                             (quad.J[1] - quad.J[0]) * rate[:, 1] * rate[:, 0]]).T

    tau = rate_dot * quad.J[np.newaxis, :] + rate_x_Jrate
    b = np.concatenate((tau, f_t), axis=-1)
    a_mat = np.concatenate((quad.y_f[np.newaxis, :], -quad.x_f[np.newaxis, :],
                            quad.z_l_tau[np.newaxis, :], np.ones_like(quad.z_l_tau)[np.newaxis, :]), 0)

    reference_u = np.zeros((len_traj, 4))
    for i in range(len_traj):
        reference_u[i, :] = np.linalg.solve(a_mat, b[i, :])

    full_pos = traj_derivatives[0, :, :].T
    full_vel = traj_derivatives[1, :, :].T
    reference_traj = np.concatenate((full_pos, q, full_vel, rate), 1)

    if map_limits is None:
        # Locate starting point right at x=0 and y=0.
        reference_traj[:, 0] -= reference_traj[0, 0]
        reference_traj[:, 1] -= reference_traj[0, 1]

    else:
        x_max_range = map_limits["x"][1] - map_limits["x"][0]
        y_max_range = map_limits["y"][1] - map_limits["y"][0]
        z_max_range = map_limits["z"][1] - map_limits["z"][0]

        x_center = x_max_range / 2 + map_limits["x"][0]
        y_center = y_max_range / 2 + map_limits["y"][0]
        z_center = z_max_range / 2 + map_limits["z"][0]

        # Center circle to center of map XY plane
        reference_traj[:, :3] += np.array([x_center, y_center, 0])
        reference_traj[:, 2] = z_center

    if plot:
        draw_poly(reference_traj, reference_u, t_ref)

    # Change format of reference input to motor activation, in interval [0, 1]
    reference_u = reference_u / quad.max_thrust

    return reference_traj, t_ref, reference_u


def load_map_limits_from_file(map_limits):
    if map_limits is not None and map_limits != "None":
        config_path = DirectoryConfig.CONFIG_DIR
        params_file = os.path.join(config_path, map_limits + '.yaml')
        try:
            with open(params_file) as file:
                limits = yaml.full_load(file)
                map_limits = {"x": [limits["x_min"], limits["x_max"]],
                              "y": [limits["y_min"], limits["y_max"]],
                              "z": [limits["z_min"], limits["z_max"]]}
                rospy.loginfo("Using world limits: " + json.dumps(limits))
        except FileNotFoundError:
            warn_msg = "Tried to load environment limits: '%s', but the file was not found. Using default limits." \
                       % map_limits
            rospy.logwarn(warn_msg)
            map_limits = None
    else:
        map_limits = None

    return map_limits


def straight_trajectory(quad, discretization_dt, speed):
    n = 3
    pos_traj = np.zeros((n, 3))
    pos_traj[:, 1] = np.linspace(-3, 3, n)
    pos_traj[:, 2] = 1
    att_traj = np.zeros_like(pos_traj)

    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed

    poly_pos_traj = fit_multi_segment_polynomial_trajectory(pos_traj.T, att_traj[:, -1].T)
    traj, yaw, t_ref = get_full_traj(poly_pos_traj, target_dt=av_dt, int_dt=discretization_dt)

    reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, None, False)
    return reference_traj, t_ref, reference_u


def random_trajectory(quad, discretization_dt, seed, speed, map_name=None, plot=False):
    map_limits = load_map_limits_from_file(map_name)

    # Get a random smooth position trajectory
    pos_traj, att_traj = random_periodical_trajectory(random_state=seed, map_limits=map_limits, plot=False)

    if map_limits is None:
        # Locate starting point right at x=0 and y=0.
        pos_traj[:, 0] -= pos_traj[0, 0]
        pos_traj[:, 1] -= pos_traj[0, 1]

    # Ensure no negative z position
    min_z_pos = np.min(pos_traj[:, -1])
    if min_z_pos < 0:
        pos_traj[:, -1] = pos_traj[:, -1] + 2 * min_z_pos * np.sign(min_z_pos)

    # Calculate feasible time based on the average distance between two position track points:
    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed

    # Calculate the polynomial fit to the position trajectory, and compute the full kinematic reference. This
    # trajectory is sampled according to the frequency that will be needed for the MPC.
    poly_pos_traj = fit_multi_segment_polynomial_trajectory(pos_traj.T, att_traj[:, -1].T)

    traj, yaw, t_ref = get_full_traj(poly_pos_traj, av_dt, discretization_dt)

    reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, map_limits, False)
    if plot:
        draw_poly(reference_traj, reference_u, t_ref, pos_traj.T)

    return reference_traj, t_ref, reference_u


def loop_trajectory(quad, discretization_dt, radius, z, lin_acc, clockwise, yawing, v_max, map_name, plot):
    """
    Creates a circular trajectory on the x-y plane that increases speed by 1m/s at every revolution.

    :param quad: Quadrotor model
    :param discretization_dt: Sampling period of the trajectory.
    :param radius: radius of loop trajectory in meters
    :param z: z position of loop plane in meters
    :param lin_acc: linear acceleration of trajectory (and successive deceleration) in m/s^2
    :param clockwise: True if the rotation will be done clockwise.
    :param yawing: True if the quadrotor yaws along the trajectory. False for 0 yaw trajectory.
    :param v_max: Maximum speed at peak velocity. Revolutions needed will be calculated automatically.
    :param map_name: Name of map to load its limits
    :param plot: Whether to plot an analysis of the planned trajectory or not.
    :return: The full 13-DoF trajectory with time and input vectors
    """

    # Apply map limits to radius
    map_limits = load_map_limits_from_file(map_name)
    if map_limits is not None:
        x_max_range = map_limits["x"][1] - map_limits["x"][0]
        y_max_range = map_limits["y"][1] - map_limits["y"][0]

        max_radius = min(x_max_range / 2, y_max_range / 2)

        radius = min(radius, max_radius)

    assert z > 0

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    ramp_t_vec = np.arange(0, ramp_up_t, discretization_dt)
    ramp_up_alpha = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * ramp_t_vec) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    coasting_t_vec = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    coasting_alpha = np.ones_like(coasting_t_vec) * alpha_acc
    # Transition phase: decelerate
    transition_t_vec = np.arange(0, 2 * ramp_up_t, discretization_dt)
    transition_alpha = alpha_acc * np.cos(np.pi / (2 * ramp_up_t) * transition_t_vec)
    transition_t_vec += coasting_t_vec[-1] + discretization_dt
    # Deceleration phase
    down_coasting_t_vec = transition_t_vec[-1] + np.arange(0, coasting_duration, discretization_dt) + discretization_dt
    down_coasting_alpha = -np.ones_like(down_coasting_t_vec) * alpha_acc
    # Bring to rest phase
    ramp_up_t_vec = down_coasting_t_vec[-1] + np.arange(0, ramp_up_t, discretization_dt) + discretization_dt
    ramp_up_alpha_end = ramp_up_alpha - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate((ramp_t_vec, coasting_t_vec, transition_t_vec, down_coasting_t_vec, ramp_up_t_vec))
    alpha_vec = np.concatenate((
        ramp_up_alpha, coasting_alpha, transition_alpha, down_coasting_alpha, ramp_up_alpha_end))

    # Calculate derivative of angular acceleration (alpha_vec)
    ramp_up_alpha_dt = alpha_acc * np.pi / (2 * ramp_up_t) * np.sin(np.pi / ramp_up_t * ramp_t_vec)
    coasting_alpha_dt = np.zeros_like(coasting_alpha)
    transition_alpha_dt = - alpha_acc * np.pi / (2 * ramp_up_t) * np.sin(np.pi / (2 * ramp_up_t) * transition_t_vec)
    alpha_dt = np.concatenate((
        ramp_up_alpha_dt, coasting_alpha_dt, transition_alpha_dt, coasting_alpha_dt, ramp_up_alpha_dt))

    if not clockwise:
        alpha_vec *= -1
        alpha_dt *= -1

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.sin(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_y = radius * np.cos(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_z = np.ones_like(pos_traj_x) * z

    vel_traj_x = (radius * w_vec * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_y = - (radius * w_vec * np.sin(angle_vec))[np.newaxis, np.newaxis, :]

    acc_traj_x = radius * (alpha_vec * np.cos(angle_vec) - w_vec ** 2 * np.sin(angle_vec))[np.newaxis, np.newaxis, :]
    acc_traj_y = - radius * (alpha_vec * np.sin(angle_vec) + w_vec ** 2 * np.cos(angle_vec))[np.newaxis, np.newaxis, :]

    jerk_traj_x = radius * (alpha_dt * np.cos(angle_vec) - alpha_vec * np.sin(angle_vec) * w_vec -
                            np.cos(angle_vec) * w_vec ** 3 - 2 * np.sin(angle_vec) * w_vec * alpha_vec)
    jerk_traj_y = - radius * (np.cos(angle_vec) * w_vec * alpha_vec + np.sin(angle_vec) * alpha_dt -
                              np.sin(angle_vec) * w_vec ** 3 + 2 * np.cos(angle_vec) * w_vec * alpha_vec)
    jerk_traj_x = jerk_traj_x[np.newaxis, np.newaxis, :]
    jerk_traj_y = jerk_traj_y[np.newaxis, np.newaxis, :]

    if yawing:
        yaw_traj = -angle_vec
    else:
        yaw_traj = np.zeros_like(angle_vec)

    traj = np.concatenate((
        np.concatenate((pos_traj_x, pos_traj_y, pos_traj_z), 1),
        np.concatenate((vel_traj_x, vel_traj_y, np.zeros_like(vel_traj_x)), 1),
        np.concatenate((acc_traj_x, acc_traj_y, np.zeros_like(acc_traj_x)), 1),
        np.concatenate((jerk_traj_x, jerk_traj_y, np.zeros_like(jerk_traj_x)), 1)), 0)

    yaw = np.concatenate((yaw_traj[np.newaxis, :], w_vec[np.newaxis, :]), 0)

    return minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, map_limits, plot)


def lemniscate_trajectory(quad, discretization_dt, radius, z, lin_acc, clockwise, yawing, v_max, map_name, plot):
    """

    :param quad:
    :param discretization_dt:
    :param radius:
    :param z:
    :param lin_acc:
    :param clockwise:
    :param yawing:
    :param v_max:
    :param map_name:
    :param plot:
    :return:
    """

    # Apply map limits to radius
    map_limits = load_map_limits_from_file(map_name)
    if map_limits is not None:
        x_max_range = map_limits["x"][1] - map_limits["x"][0]
        y_max_range = map_limits["y"][1] - map_limits["y"][0]

        max_radius = min(x_max_range / 2, y_max_range / 2)

        radius = min(radius, max_radius)

    assert z > 0

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    ramp_t_vec = np.arange(0, ramp_up_t, discretization_dt)
    ramp_up_alpha = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * ramp_t_vec) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    coasting_t_vec = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    coasting_alpha = np.ones_like(coasting_t_vec) * alpha_acc
    # Transition phase: decelerate
    transition_t_vec = np.arange(0, 2 * ramp_up_t, discretization_dt)
    transition_alpha = alpha_acc * np.cos(np.pi / (2 * ramp_up_t) * transition_t_vec)
    transition_t_vec += coasting_t_vec[-1] + discretization_dt
    # Deceleration phase
    down_coasting_t_vec = transition_t_vec[-1] + np.arange(0, coasting_duration, discretization_dt) + discretization_dt
    down_coasting_alpha = -np.ones_like(down_coasting_t_vec) * alpha_acc
    # Bring to rest phase
    ramp_up_t_vec = down_coasting_t_vec[-1] + np.arange(0, ramp_up_t, discretization_dt) + discretization_dt
    ramp_up_alpha_end = ramp_up_alpha - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate((ramp_t_vec, coasting_t_vec, transition_t_vec, down_coasting_t_vec, ramp_up_t_vec))
    alpha_vec = np.concatenate((
        ramp_up_alpha, coasting_alpha, transition_alpha, down_coasting_alpha, ramp_up_alpha_end))

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Adaption: we achieve the highest spikes in the bodyrates when passing through the 'center' part of the figure-8
    # This leads to negative reference thrusts.
    # Let's see if we can alleviate this by adapting the z-reference in these parts to add some acceleration in the
    # z-component
    z_dim = 0.0

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.cos(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_y = radius * (np.sin(angle_vec) * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    pos_traj_z = - z_dim * np.cos(4.0 * angle_vec)[np.newaxis, np.newaxis, :] + z

    vel_traj_x = -radius * (w_vec * np.sin(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_y = radius * (w_vec * np.cos(angle_vec) ** 2 - w_vec * np.sin(angle_vec) ** 2)[np.newaxis, np.newaxis, :]
    vel_traj_z = 4.0 * z_dim * w_vec * np.sin(4.0 * angle_vec)[np.newaxis, np.newaxis, :]

    acc_traj_x = -radius * (alpha_vec * np.sin(angle_vec) + w_vec ** 2 * np.cos(angle_vec))
    acc_traj_y = radius * (alpha_vec * np.cos(angle_vec) ** 2 - 2.0 * w_vec ** 2 * np.cos(angle_vec) * np.sin(
        angle_vec) - alpha_vec * np.sin(angle_vec) ** 2 - 2.0 * w_vec ** 2 * np.sin(angle_vec) * np.cos(angle_vec))
    acc_traj_z = 16.0 * z_dim * (w_vec ** 2 * np.cos(4.0 * angle_vec) + alpha_vec * np.sin(4.0 * angle_vec))
    acc_traj_x = acc_traj_x[np.newaxis, np.newaxis, :]
    acc_traj_y = acc_traj_y[np.newaxis, np.newaxis, :]
    acc_traj_z = acc_traj_z[np.newaxis, np.newaxis, :]

    traj = np.concatenate((
        np.concatenate((pos_traj_x, pos_traj_y, pos_traj_z), 1),
        np.concatenate((vel_traj_x, vel_traj_y, vel_traj_z), 1),
        np.concatenate((acc_traj_x, acc_traj_y, acc_traj_z), 1)), 0)

    yaw = np.zeros_like(traj)

    return minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, map_limits, plot)
