""" Class for interfacing the data-augmented MPC with ROS.

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

from src.utils.utils import parse_xacro_file
from src.quad_mpc.quad_3d import Quadrotor3D
from src.quad_mpc.quad_3d_mpc import Quad3DMPC
import numpy as np
import std_msgs.msg
from quadrotor_msgs.msg import ControlCommand
import rospy
import os


def custom_quad_param_loader(quad_name):

    this_path = os.path.dirname(os.path.realpath(__file__))
    params_file = os.path.join(this_path, '..', '..', 'config', quad_name + '.xacro')

    # Get parameters for drone
    attrib = parse_xacro_file(params_file)

    quad = Quadrotor3D(noisy=False, drag=False, payload=False, motor_noise=False)
    quad.mass = float(attrib['mass']) + float(attrib['mass_rotor']) * 4
    quad.J = np.array([float(attrib['body_inertia'][0]['ixx']),
                       float(attrib['body_inertia'][0]['iyy']),
                       float(attrib['body_inertia'][0]['izz'])])
    quad.length = float(attrib['arm_length'])

    quad.max_thrust = float(attrib["max_rot_velocity"]) ** 2 * float(attrib["motor_constant"])
    quad.c = float(attrib['moment_constant'])

    # x configuration
    if quad_name != "hummingbird":
        h = np.cos(np.pi / 4) * quad.length
        quad.x_f = np.array([h, -h, -h, h])
        quad.y_f = np.array([-h, -h, h, h])
        quad.z_l_tau = np.array([-quad.c, quad.c, -quad.c, quad.c])

    # + configuration
    else:
        quad.x_f = np.array([quad.length, 0, -quad.length, 0])
        quad.y_f = np.array([0, quad.length, 0, -quad.length])
        quad.z_l_tau = -np.array([-quad.c, quad.c, -quad.c, quad.c])

    return quad


class ROSGPMPC:
    def __init__(self, t_horizon, n_mpc_nodes, opt_dt, quad_name, point_reference=False, gp_models=None, rdrv=None):

        quad = custom_quad_param_loader(quad_name)

        # Initialize quad MPC
        if point_reference:
            acados_config = {
                "solver_type": "SQP",
                "terminal_cost": True
            }
        else:
            acados_config = {
                "solver_type": "SQP_RTI",
                "terminal_cost": False
            }

        q_diagonal = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
        r_diagonal = np.array([1.0, 1.0, 1.0, 1.0])

        q_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).T

        quad_mpc = Quad3DMPC(quad, t_horizon=t_horizon, optimization_dt=opt_dt, n_nodes=n_mpc_nodes,
                             pre_trained_models=gp_models, model_name=quad_name, solver_options=acados_config,
                             q_mask=q_mask, q_cost=q_diagonal, r_cost=r_diagonal, rdrv_d_mat=rdrv)

        self.quad_name = quad_name
        self.quad = quad
        self.quad_mpc = quad_mpc

        # Last optimization
        self.last_w = None

    def set_state(self, x):
        """
        Set quadrotor state estimate from an odometry measurement
        :param x: measured state from odometry. List with 13 components with the format: [p_xyz, q_wxyz, v_xyz, w_xyz]
        """

        self.quad.set_state(x)

    def set_gp_state(self, x):
        """
        Set a quadrotor state estimate from an odometry measurement. While the input state in the function set_state()
        will be used as initial state for the MPC optimization, the input state of this function will be used to
        evaluate the GP's. If this function is never called, then the GP's will be evaluated with the state from
        set_state() too.
        :param x: measured state from odometry. List with 13 components with the format: [p_xyz, q_wxyz, v_xyz, w_xyz]
        """

        self.quad.set_gp_state(x)

    def set_reference(self, x_ref, u_ref):
        """
        Set a reference state for the optimizer.
        :param x_ref: list with 4 sub-components (position, angle quaternion, velocity, body rate). If these four
        are lists, then this means a single target point is used. If they are Nx3 and Nx4 (for quaternion) numpy arrays,
        then they are interpreted as a sequence of N tracking points.
        :param u_ref: Optional target for the optimized control inputs
        """

        return self.quad_mpc.set_reference(x_reference=x_ref, u_reference=u_ref)

    def optimize(self, model_data):

        w_opt, x_opt = self.quad_mpc.optimize(use_model=model_data, return_x=True)

        # Remember solution for next optimization
        self.last_w = self.quad_mpc.reshape_input_sequence(w_opt)

        next_control = ControlCommand()
        next_control.header = std_msgs.msg.Header()
        next_control.header.stamp = rospy.Time.now()
        next_control.control_mode = 2
        next_control.armed = True
        next_control.collective_thrust = np.sum(w_opt[:4]) * self.quad.max_thrust / self.quad.mass
        next_control.bodyrates.x = x_opt[1, -3]
        next_control.bodyrates.y = x_opt[1, -2]
        next_control.bodyrates.z = x_opt[1, -1]
        next_control.rotor_thrusts = w_opt[:4] * self.quad.max_thrust

        # Something is off with the colibri
        if self.quad_name == "colibri":
            next_control.collective_thrust -= 1.8

        return next_control, w_opt
