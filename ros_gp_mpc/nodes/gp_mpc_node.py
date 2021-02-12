#!/usr/bin/env python3.6
""" ROS node for the data-augmented MPC, to use in the Gazebo simulator and real world experiments.

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

import json
import time
import rospy
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Empty
from geometry_msgs.msg import PoseStamped
from ros_gp_mpc.msg import ReferenceTrajectory
from quadrotor_msgs.msg import ControlCommand
from src.quad_mpc.create_ros_gp_mpc import ROSGPMPC
from gazebo_msgs.srv import GetPhysicsProperties
from src.utils.utils import jsonify, interpol_mse, quaternion_state_mse, load_pickled_models, v_dot_q, \
    separate_variables
from src.utils.visualization import trajectory_tracking_results, mse_tracking_experiment_plot, \
    load_past_experiments, get_experiment_files
from src.experiments.point_tracking_and_record import make_record_dict, get_record_file_and_dir, check_out_data
from src.model_fitting.rdrv_fitting import load_rdrv


def odometry_parse(odom_msg):
    p = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z]
    q = [odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
         odom_msg.pose.pose.orientation.z]
    v = [odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z]
    w = [odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z]

    return p, q, v, w


def make_raw_optitrack_dict():
    rec_dict_raw = make_record_dict(state_dim=7)
    # Remove unnecessary entries
    keys = list(rec_dict_raw.keys())
    for key in keys:
        if key not in ["state_in", "timestamp"]:
            rec_dict_raw.pop(key)
    return rec_dict_raw


def odometry_skipped_warning(last_seq, current_seq, stage):
    skip_msg = "Odometry skipped at %s step. Last: %d, current: %d" % (stage, last_seq, current_seq)
    rospy.logwarn(skip_msg)


class GPMPCWrapper:
    def __init__(self, quad_name, environment="gazebo", recording_options=None, load_options=None, use_ekf=False,
                 rdrv=None, plot=False, reset_experiment=False):

        if recording_options is None:
            recording_options = {"recording": False}

        # If on a simulation environment, figure out if physics are slowed down
        if environment == "gazebo":
            try:
                get_gazebo_physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
                resp = get_gazebo_physics()
                physics_speed = resp.max_update_rate * resp.time_step
                rospy.loginfo("Physics running at %.2f normal speed" % physics_speed)
            except rospy.ServiceException as e:
                print("Service call failed: %s" % e)
                physics_speed = 1
        else:
            physics_speed = 1
        self.physics_speed = physics_speed

        self.environment = environment
        self.plot = plot
        self.recording_options = recording_options

        # Control at 50 (sim) or 60 (real) hz. Use time horizon=1 and 10 nodes
        self.n_mpc_nodes = rospy.get_param('~n_nodes', default=10)
        self.t_horizon = rospy.get_param('~t_horizon', default=1.0)
        self.control_freq_factor = rospy.get_param('~control_freq_factor', default=5 if environment == "gazebo" else 6)
        self.opt_dt = self.t_horizon / (self.n_mpc_nodes * self.control_freq_factor)

        # Load trained GP model
        if load_options is not None:
            rospy.loginfo("Attempting to load GP model from:\n   git: {}\n   name: {}\n   meta: {}".format(
                load_options["git"], load_options["model_name"], load_options["params"]))
            pre_trained_models = load_pickled_models(model_options=load_options)
            if pre_trained_models is None:
                rospy.logwarn("Model parameters specified did not match with any pre-trained GP")
        else:
            pre_trained_models = None
        self.pre_trained_models = pre_trained_models
        self.git_v = load_options["git"]
        if self.pre_trained_models is not None:
            rospy.loginfo("Successfully loaded GP model")
            self.model_name = load_options["model_name"]
        elif rdrv is not None:
            self.model_name = "rdrv"
        else:
            self.model_name = "Nominal"

        # Initialize GP MPC for point tracking
        self.gp_mpc = ROSGPMPC(self.t_horizon, self.n_mpc_nodes, self.opt_dt, quad_name=quad_name,
                               point_reference=False, gp_models=pre_trained_models, rdrv=rdrv)

        # Last state obtained from odometry
        self.x = None

        # Elapsed time between two recordings
        self.last_update_time = time.time()

        # Last references. Use hovering activation as input reference
        self.last_x_ref = None
        self.last_u_ref = None

        # Reference trajectory variables
        self.x_ref = None
        self.t_ref = None
        self.u_ref = None
        self.current_idx = 0
        self.quad_trajectory = None
        self.quad_controls = None
        self.w_control = None

        # Provisional reference for "waiting_for_reference" hovering mode
        self.x_ref_prov = None

        # To measure optimization elapsed time
        self.optimization_dt = 0

        # Thread for MPC optimization
        self.mpc_thread = threading.Thread()

        # Trajectory tracking experiment. Dims: seed x av_v x n_samples
        if reset_experiment:
            self.metadata_dict = {}
        else:
            self.metadata_dict, _, _, _ = load_past_experiments()
        self.mse_exp = np.zeros((0, 0, 0, 1))
        self.t_opt = np.zeros((0, 0, 0))
        self.mse_exp_v_max = np.zeros((0, 0))
        self.ref_traj_name = ""
        self.ref_v = 0
        self.run_traj_counter = 0

        # Keep track of status of MPC object
        self.odom_available = False

        # Binary variable to run MPC only once every other odometry callback
        self.optimize_next = True

        # Binary variable to completely skip an odometry if in flying arena
        self.skip_next = False

        # Remember the sequence number of the last odometry message received.
        self.last_odom_seq_number = 0

        # Measure if trajectory starting point is reached
        self.x_initial_reached = False

        # Variables for recording mode
        self.recording_warmup = True
        self.x_pred = None
        self.w_opt = None

        # Odometry estimate for GP. None by default if same odometry as control reference should be used
        self.gp_odom = None

        # Get recording file and directory
        blank_recording_dict = make_record_dict(state_dim=13)
        if recording_options["recording"]:
            record_raw_optitrack = recording_options["record_raw_optitrack"]
            overwrite = recording_options["overwrite"]
            metadata = {self.environment: "default"}

            rec_dict, rec_file = get_record_file_and_dir(
                blank_recording_dict, recording_options, simulation_setup=metadata, overwrite=overwrite)

            # If flying with the optitrack system, also record raw optitrack estimates
            if self.environment == "flying_room" or self.environment == 'arena' and record_raw_optitrack:
                rec_dict_raw = make_raw_optitrack_dict()
                metadata = {self.environment: "optitrack_raw"}
                rec_dict_raw, rec_file_raw = get_record_file_and_dir(
                    rec_dict_raw, recording_options, simulation_setup=metadata, overwrite=overwrite)
            else:
                rec_dict_raw = rec_file_raw = None

        else:
            record_raw_optitrack = False
            rec_dict = rec_file = None
            rec_dict_raw = rec_file_raw = None

        self.rec_dict = rec_dict
        self.rec_file = rec_file
        self.rec_dict_raw = rec_dict_raw
        self.rec_file_raw = rec_file_raw

        self.landing = False
        self.override_land = False
        self.ground_level = False
        self.controller_off = False

        # Setup node publishers and subscribers. The odometry (sub) and control (pub) topics will vary depending on
        # which environment is being used
        ekf_odom_topic = None
        if self.environment == "gazebo":
            odom_topic = "/" + quad_name + "/ground_truth/odometry"
            raw_topic = None
        elif self.environment == "arena":
            # Assume arena world setup
            odom_topic = "/" + quad_name + "/state_estimate"
            raw_topic = "/vicon/" + quad_name
            if use_ekf:
                ekf_odom_topic = "/" + quad_name + "/state_estimate_ekf"
        else:
            # Assume real world setup
            odom_topic = "/" + quad_name + "/state_estimate"
            raw_topic = "/optitrack/" + quad_name
            if use_ekf:
                ekf_odom_topic = "/" + quad_name + "/state_estimate_ekf"

        land_topic = "/" + quad_name + "/autopilot/land"
        control_topic = "/" + quad_name + "/autopilot/control_command_input"
        off_topic = "/" + quad_name + "/autopilot/off"

        reference_topic = "reference"
        status_topic = "busy"

        # Publishers
        self.control_pub = rospy.Publisher(control_topic, ControlCommand, queue_size=1, tcp_nodelay=True)
        self.status_pub = rospy.Publisher(status_topic, Bool, queue_size=1)
        self.off_pub = rospy.Publisher(off_topic, Empty, queue_size=1)

        # Subscribers
        self.land_sub = rospy.Subscriber(land_topic, Empty, self.land_callback)
        self.ref_sub = rospy.Subscriber(reference_topic, ReferenceTrajectory, self.reference_callback)
        if ekf_odom_topic:
            # We get a second odometry estimate which is smoothed. Used to evaluate the GP's.
            self.odom_sub = rospy.Subscriber(
                odom_topic, Odometry, self.odometry_callback, queue_size=1, tcp_nodelay=True)
            self.ekf_odom_sub = rospy.Subscriber(
                ekf_odom_topic, Odometry, self.ekf_odom_callback, queue_size=1, tcp_nodelay=True)
        else:
            self.odom_sub = rospy.Subscriber(
                odom_topic, Odometry, self.odometry_callback, queue_size=1, tcp_nodelay=True)
        if raw_topic is not None and record_raw_optitrack:
            self.raw_sub = rospy.Subscriber(raw_topic, PoseStamped, self.raw_odometry_callback)

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            # Publish if MPC is busy with a current trajectory
            msg = Bool()
            msg.data = not (self.x_ref is None and self.odom_available)
            self.status_pub.publish(msg)
            rate.sleep()

    def land_callback(self, _):
        """
        Trigger landing sequence.
        :param _: empty message
        """

        rospy.loginfo("Controller disabled. Landing with RPG MPC")
        self.override_land = True

    def rest_state(self):
        """
        Set quad reference to hover state at position (0, 0, 0.1)
        """
        self.last_x_ref = [[self.x[0], self.x[1], self.x[2]], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.last_u_ref = self.gp_mpc.quad.g[-1] * self.gp_mpc.quad.mass / (self.gp_mpc.quad.max_thrust * 4)
        self.last_u_ref = self.last_u_ref[0] * np.array([1, 1, 1, 1])

    def run_mpc(self, odom, recording=True):
        """
        :param odom: message from subscriber.
        :type odom: Odometry
        :param recording: If False, some messages were skipped between this and last optimization. Don't record any data
        during this optimization if in recording mode.
        """

        if not self.odom_available:
            return

        # Measure time between initial state was checked in and now
        dt = odom.header.stamp.to_time() - self.last_update_time

        model_data, x_guess, u_guess = self.set_reference()

        # Run MPC and publish control
        try:
            tic = time.time()
            next_control, w_opt = self.gp_mpc.optimize(model_data)
            self.optimization_dt += time.time() - tic
            # print("MPC thread. Seq: %d. Topt: %.4f" % (odom.header.seq, (time.time() - tic) * 1000))
            self.control_pub.publish(next_control)
            if self.x_initial_reached and self.current_idx < self.w_control.shape[0]:
                self.w_control[self.current_idx, 0] = next_control.bodyrates.x
                self.w_control[self.current_idx, 1] = next_control.bodyrates.y
                self.w_control[self.current_idx, 2] = next_control.bodyrates.z

        except KeyError:
            self.recording_warmup = True
            # Should not happen anymore.
            rospy.logwarn("Tried to run an MPC optimization but MPC is not ready yet.")
            return

        if w_opt is not None:

            # Check out final states. self.recording_warmup can only be true in recording mode.
            if not self.recording_warmup and recording and self.x_initial_reached:
                x_out = np.array(self.x)[np.newaxis, :]
                self.rec_dict = check_out_data(self.rec_dict, x_out, None, self.w_opt, dt)

            self.w_opt = w_opt
            if self.x_initial_reached and self.current_idx < self.quad_controls.shape[0]:
                self.quad_controls[self.current_idx, :] = np.expand_dims(self.w_opt[:4], axis=0)

    def check_out_initial_state(self, odom):
        """
        Add the initial state to the recording dictionary and start counting until next optimization
        :param odom: message from subscriber.
        :type odom: Odometry
        """

        if self.w_opt is not None:
            self.last_update_time = odom.header.stamp.to_time()
            self.rec_dict["state_in"] = np.append(self.rec_dict["state_in"], np.array(self.x)[np.newaxis, :], 0)
            self.rec_dict["timestamp"] = np.append(self.rec_dict["timestamp"], odom.header.stamp.to_time())
            if self.current_idx < self.x_ref.shape[0]:
                self.rec_dict["state_ref"] = np.append(self.rec_dict["state_ref"], self.x_ref[np.newaxis, self.current_idx, :], 0)
            self.recording_warmup = False

    def reference_callback(self, msg):
        """
        Callback for receiving a reference trajectory
        :param msg: message from subscriber
        :type msg: ReferenceTrajectory
        """

        seq_len = msg.seq_len

        if seq_len == 0:
            # Hover-in-place mode
            self.x_ref = self.x[:7]
            self.u_ref = None
            self.t_ref = None

            off_msg = Empty()
            self.off_pub.publish(off_msg)
            self.controller_off = True

            # If this is the end of a reference tracking experiment, generate the results plot
            if self.mse_exp.shape[0] != 0 and self.run_traj_counter > 0:
                self.plot_tracking_mse_experiment()

            self.landing = False
            rospy.loginfo("No more references will be received")
            return

        # Save reference name
        self.ref_traj_name = msg.traj_name
        self.ref_v = msg.v_input

        # Save reference trajectory, relative times and inputs
        self.x_ref = np.array(msg.trajectory).reshape(seq_len, -1)
        self.t_ref = np.array(msg.dt)
        self.u_ref = np.array(msg.inputs).reshape(seq_len, -1)
        self.quad_trajectory = np.zeros((len(self.t_ref), len(self.x)))
        self.quad_controls = np.zeros((len(self.t_ref), 4))

        self.w_control = np.zeros((len(self.t_ref), 3))

        rospy.loginfo("New trajectory received. Time duration: %.2f s" % self.t_ref[-1])

    def sync_odom_callback(self, msg1, msg2):
        """
        Synchronized callback function for the odometry estimates. TODO: is this the best way?
        :param msg1: Odometry estimate for flight control.
        :type msg1: Odometry
        :param msg2: Odometry estimate for GP inference.
        :type msg2: Odometry
        """

        p, q, v, w = odometry_parse(msg2)
        self.gp_odom = p + q + v + w
        self.odometry_callback(msg1)

    def ekf_odom_callback(self, msg):

        p, q, v, w = odometry_parse(msg)
        self.gp_odom = p + q + v + w

    def odometry_callback(self, msg):
        """
        Callback function for Odometry subscriber
        :param msg: message from subscriber.
        :type msg: Odometry
        """

        if self.controller_off:
            return

        p, q, v, w = odometry_parse(msg)

        # Change velocity to world frame if in gazebo environment
        if self.environment == "gazebo":
            v_w = v_dot_q(np.array(v), np.array(q)).tolist()
        else:
            v_w = v

        self.x = p + q + v_w + w

        try:
            # Update the state estimate of the quad
            self.gp_mpc.set_state(self.x)

            # If an estimate specifically for the GP's is available, also update it
            if self.gp_odom is not None:
                self.gp_mpc.set_gp_state(self.gp_odom)

        except AttributeError:
            # The GP MPC object instantiation is still not finished
            return

        if self.override_land:
            return

        # If the above try passed then MPC is ready
        self.odom_available = True

        # We only optimize once every two odometry messages
        if not self.optimize_next:
            self.mpc_thread.join()

            # If currently on trajectory tracking, pay close attention to any skipped messages.
            if self.x_initial_reached:

                # Count how many messages were skipped (ideally 0)
                skipped_messages = int(msg.header.seq - self.last_odom_seq_number - 1)
                if skipped_messages > 0:
                    warn_msg = "Recording time skipped messages: %d" % skipped_messages
                    rospy.logwarn(warn_msg)

                # Adjust current index in trajectory
                self.current_idx += divmod(skipped_messages, 2)[0]
                # If odd number of skipped messages, do optimization
                if skipped_messages > 0 and skipped_messages % 2 == 1:

                    if self.recording_options["recording"]:
                        self.check_out_initial_state(msg)

                    # Run MPC now
                    self.run_mpc(msg)
                    self.last_odom_seq_number = msg.header.seq
                    self.optimize_next = False
                    return

            self.optimize_next = True
            if self.recording_options["recording"] and self.x_initial_reached:
                self.check_out_initial_state(msg)
            return

        # Run MPC
        if msg.header.seq > self.last_odom_seq_number + 2 and self.last_odom_seq_number > 0 and self.x_initial_reached:
            # If one message was skipped at this point, then the reference is already late. Compensate by
            # optimizing twice in a row and hope to do it fast...
            if self.recording_options["recording"] and self.x_initial_reached:
                self.check_out_initial_state(msg)
            self.run_mpc(msg)
            self.optimize_next = True
            self.last_odom_seq_number = msg.header.seq
            odometry_skipped_warning(self.last_odom_seq_number, msg.header.seq, "optimization")
            return

        def _thread_func():
            self.run_mpc(msg)
        self.mpc_thread = threading.Thread(target=_thread_func(), args=(), daemon=True)
        self.mpc_thread.start()

        self.last_odom_seq_number = msg.header.seq
        self.optimize_next = False

    def raw_odometry_callback(self, msg):
        """
        Callback function for the raw Optitrack subscriber. Adds the data to the raw data dictionary.
        :param msg: Raw data from Optitrack estimator
        :type msg: PoseStamped
        """

        if not self.recording_options["recording"] or not self.x_initial_reached:
            return

        x = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                      msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])

        self.rec_dict_raw["state_in"] = np.append(self.rec_dict_raw["state_in"], x[np.newaxis, :], 0)
        self.rec_dict_raw["timestamp"] = np.append(self.rec_dict_raw["timestamp"], msg.header.stamp.to_time())

    def hover_here(self, x):
        self.rest_state()
        x_ref = [x[:3], x[3:7], [0, 0, 0], [0, 0, 0]]
        u_ref = self.last_u_ref
        x_guess = np.tile(np.concatenate(x_ref)[np.newaxis, :], (self.n_mpc_nodes, 1))
        u_guess = np.tile(self.last_u_ref[np.newaxis, :], (self.n_mpc_nodes, 1))
        return self.gp_mpc.set_reference(x_ref, u_ref), x_guess, u_guess

    def set_reference(self):

        if self.environment == "gazebo":
            th = 0.1
        else:
            th = 0.5
        mask = [1] * 9 + [0] * 3

        x_ref = self.last_x_ref
        u_ref = self.last_u_ref

        x_guess = None
        u_guess = None

        if not self.odom_available:
            return

        # Check if landing mode
        if self.landing:
            dz = np.sign(0.1 - self.x[2])
            dz = dz * 0.1 if self.environment != "gazebo" else dz * 0.3
            x_ref[0][2] = min(0.1, self.x[2] + dz) if dz > 0 else max(0.1, self.x[2] + dz)

            # Check if z position is close to target.
            if abs(self.x[2] - 0.1) < 0.05:

                executed_x_ref = self.x_ref
                executed_u_ref = self.u_ref
                executed_t_ref = self.t_ref

                self.x_ref = None
                self.u_ref = None
                self.t_ref = None

                self.x_initial_reached = False

                if self.recording_options["recording"]:
                    self.save_recording_data()

                # Calculate MSE of position tracking and maximum axial velocity achieved
                rmse = interpol_mse(executed_t_ref, executed_x_ref[:, :3], executed_t_ref, self.quad_trajectory[:, :3])
                self.optimization_dt /= self.current_idx

                if self.ref_traj_name in self.metadata_dict.keys():
                    if self.model_name in self.metadata_dict[self.ref_traj_name].keys():
                        self.metadata_dict[self.ref_traj_name][self.model_name][self.ref_v] = [rmse,
                                                                                               self.optimization_dt]
                    else:
                        self.metadata_dict[self.ref_traj_name][self.model_name] = {
                            self.ref_v: [rmse, self.optimization_dt]}
                else:
                    self.metadata_dict[self.ref_traj_name] = {
                        self.model_name: {self.ref_v: [rmse, self.optimization_dt]}}

                n_trajectories = len(self.metadata_dict.keys())
                n_models = len(self.metadata_dict[self.ref_traj_name].keys())
                n_vel = len(self.metadata_dict[self.ref_traj_name][self.model_name].keys())

                # Figure out dimensions of data so far
                self.mse_exp = np.zeros((n_trajectories, n_vel, n_models, 1))
                self.t_opt = np.zeros((n_trajectories, n_vel, n_models))
                self.mse_exp_v_max = np.zeros((n_trajectories, n_vel))

                # Add data to array
                # Dimensions of mse_exp: n_trajectories x n_average_speeds x n_models x n_sim_options
                for traj_id, traj_type in enumerate(self.metadata_dict.keys()):
                    for model_id, model_type in enumerate(self.metadata_dict[traj_type].keys()):
                        for vel_id, vel in enumerate(self.metadata_dict[traj_type][model_type].keys()):
                            self.mse_exp[traj_id, vel_id, model_id, 0] = self.metadata_dict[traj_type][model_type][vel][0]
                            self.t_opt[traj_id, vel_id, model_id] = self.optimization_dt
                            self.mse_exp_v_max[traj_id, vel_id] = vel

                v_max = np.max(self.quad_trajectory[:, 7:10])
                rospy.loginfo("Tracking complete. Total RMSE: %.5f m. Max axial vel: %.3f. "
                              "Mean optimization time: %.3f ms" % (rmse, v_max, self.optimization_dt * 1000))

                self.current_idx = 0
                if self.plot:
                    with_gp = ' + GP ' if self.pre_trained_models is not None else ' - GP '
                    tit = r'$v_{max}$=%.2f m/s | RMSE: %.4f | %s ' % (v_max, float(rmse), with_gp)
                    trajectory_tracking_results(executed_t_ref, executed_x_ref, self.quad_trajectory, executed_u_ref,
                                                self.quad_controls, w_control=self.w_control, title=tit)

                # Stop landing. Quad is close to ground level
                self.landing = False
                self.ground_level = True

            return self.gp_mpc.set_reference(x_ref, u_ref), x_guess, u_guess

        # Check if reference trajectory is set up. If not, pick current position and keep hover
        if self.x_ref is None:

            self.ground_level = False
            # We are waiting for a new reference. Set in provisional hover mode at current position
            if self.x_ref_prov is None:
                rospy.loginfo("Entering provisional hovering mode while to reference is available at: ")
                self.x_ref_prov = self.x
                rospy.loginfo(self.x_ref_prov)

            # Provisional hovering mode
            return self.hover_here(self.x_ref_prov)

        if self.x_ref_prov is not None:
            self.x_ref_prov = None
            rospy.loginfo("Abandoning provisional hovering mode.")

        # Check if reference is hovering mode
        if isinstance(self.x_ref, list):
            return self.hover_here(self.x_ref)

        # Trajectory tracking mode. Check if target reached
        if quaternion_state_mse(np.array(self.x), self.x_ref[0, :], mask) < th and not self.x_initial_reached:
            # Initial position of trajectory has been reached
            self.x_initial_reached = True
            self.odom_available = False
            self.optimization_dt = 0
            rospy.loginfo("Reached initial position of trajectory.")
            model_data = self.gp_mpc.set_reference(separate_variables(self.x_ref[:1, :]), self.u_ref[:1, :])
            return model_data, x_guess, u_guess

        # Raise the drone towards the initial position of the trajectory
        if not self.x_initial_reached:
            dx = 0.3 * np.sign(self.x_ref[0, 0] - self.x[0])
            dy = 0.3 * np.sign(self.x_ref[0, 1] - self.x[1])
            dz = 0.3 * np.sign(self.x_ref[0, 2] - self.x[2])
            x_ref[0][0] = min(self.x_ref[0, 0], self.x[0] + dx) if dx > 0 else max(self.x_ref[0, 0], self.x[0] + dx)
            x_ref[0][1] = min(self.x_ref[0, 1], self.x[1] + dy) if dy > 0 else max(self.x_ref[0, 1], self.x[1] + dy)
            x_ref[0][2] = min(self.x_ref[0, 2], self.x[2] + dz) if dz > 0 else max(self.x_ref[0, 2], self.x[2] + dz)

        elif self.current_idx < self.x_ref.shape[0]:

            self.quad_trajectory[self.current_idx, :] = np.expand_dims(self.x, axis=0)

            # Trajectory tracking
            ref_traj = self.x_ref[self.current_idx:self.current_idx + self.n_mpc_nodes * self.control_freq_factor, :]
            ref_u = self.u_ref[self.current_idx:self.current_idx + self.n_mpc_nodes * self.control_freq_factor, :]

            # Indices for down-sampling the reference to number of MPC nodes
            downsample_ref_ind = np.arange(0, min(self.control_freq_factor * self.n_mpc_nodes, ref_traj.shape[0]),
                                           self.control_freq_factor, dtype=int)

            # Sparser references (same dt as node separation)
            x_ref = ref_traj[downsample_ref_ind, :]
            u_ref = ref_u[downsample_ref_ind, :]

            # Initial guesses
            u_guess = u_ref
            x_guess = x_ref
            while u_guess.shape[0] < self.n_mpc_nodes:
                x_guess = np.concatenate((x_guess, x_guess[-1:, :]), axis=0)
                u_guess = np.concatenate((u_guess, u_guess[-1:, :]), axis=0)

            x_ref = separate_variables(x_ref)

            self.current_idx += 1

        # End of reference reached
        elif self.current_idx == self.x_ref.shape[0]:

            # Add one to the completed trajectory counter
            self.run_traj_counter += 1

            # Lower drone to a safe height
            self.landing = True
            self.rest_state()
            x_ref = self.last_x_ref
            u_ref = self.last_u_ref

            # Stop recording
            self.x_initial_reached = False
            self.recording_warmup = True

        self.last_x_ref = x_ref
        self.last_u_ref = u_ref
        return self.gp_mpc.set_reference(x_ref, u_ref), x_guess, u_guess

    def plot_tracking_mse_experiment(self):

        metadata_file, _, _, _ = get_experiment_files()

        # Save data for reload
        with open(metadata_file, 'w') as json_file:
            json.dump(self.metadata_dict, json_file, indent=4)

        # Sort seeds dictionary by value
        traj_type_labels = [k for k in self.metadata_dict.keys()]
        model_type_labels = [k for k in self.metadata_dict[traj_type_labels[0]].keys()]

        mse_tracking_experiment_plot(v_max=self.mse_exp_v_max, mse=self.mse_exp, traj_type_vec=traj_type_labels,
                                     train_samples_vec=model_type_labels, legends=model_type_labels,
                                     y_labels=["RotorS"], t_opt=self.t_opt)

    def save_recording_data(self):

        # Remove exceeding data entry if needed
        if len(self.rec_dict['state_in']) > len(self.rec_dict['input_in']):
            self.rec_dict['state_in'] = self.rec_dict['state_in'][:-1]
            self.rec_dict['timestamp'] = self.rec_dict['timestamp'][:-1]

        # Compute predictions offline to avoid extra overhead while in trajectory tracking control
        rospy.loginfo("Filling in dataset and saving...")
        for i in tqdm(range(len(self.rec_dict['input_in']))):
            x_0 = self.rec_dict['state_in'][i]
            x_f = self.rec_dict['state_out'][i]
            u = self.rec_dict['input_in'][i]
            dt = self.rec_dict['dt'][i]
            x_pred, _ = self.gp_mpc.quad_mpc.forward_prop(x_0, u, t_horizon=dt, use_gp=False)
            x_pred = x_pred[-1, np.newaxis, :]

            self.rec_dict['state_pred'] = np.append(self.rec_dict['state_pred'], x_pred, axis=0)
            self.rec_dict['error'] = np.append(self.rec_dict['error'], x_f - x_pred, axis=0)

        # Save datasets
        x_dim = self.rec_dict["state_in"].shape[1]

        for key in self.rec_dict.keys():
            print(key, " ", self.rec_dict[key].shape)
            self.rec_dict[key] = jsonify(self.rec_dict[key])
        df = pd.DataFrame(self.rec_dict)
        df.to_csv(self.rec_file, index=True, mode='a', header=False)

        if self.rec_dict_raw is not None:
            data_len = min(self.rec_dict_raw["state_in"].shape[0], len(self.rec_dict_raw["timestamp"]))

            # To ensure same length of all entries
            for key in self.rec_dict_raw.keys():
                self.rec_dict_raw[key] = self.rec_dict_raw[key][:data_len]
                print(key, " ", self.rec_dict_raw[key].shape)
                self.rec_dict_raw[key] = jsonify(self.rec_dict_raw[key])

            df = pd.DataFrame(self.rec_dict_raw)
            df.to_csv(self.rec_file_raw, index=True, mode='a', header=False)

            self.rec_dict_raw = make_raw_optitrack_dict()

        # Reset recording dictionaries
        self.rec_dict = make_record_dict(x_dim)


def main():
    rospy.init_node("gp_mpc")

    # Recording parameters
    recording_options = {
        "recording": rospy.get_param('~recording', default=True),
        "dataset_name": "deleteme",
        "training_split": True,
        "overwrite": True,
        "record_raw_optitrack": True
    }

    dataset_name = rospy.get_param('~dataset_name', default=None)
    overwrite = rospy.get_param('~overwrite', default=None)
    training = rospy.get_param('~training_split', default=None)
    raw_optitrack = rospy.get_param('~record_raw_optitrack', default=None)
    if dataset_name is not None:
        recording_options["dataset_name"] = dataset_name
    if overwrite is not None:
        recording_options["overwrite"] = overwrite
    if training is not None:
        recording_options["training_split"] = training
    if raw_optitrack is not None:
        recording_options["record_raw_optitrack"] = raw_optitrack

    # GP loading parameters
    load_options = {
        "git": "b6e73a5",
        "model_name": "",
        "params": None
    }
    git_id = rospy.get_param('~model_version', default=None)
    model_name = rospy.get_param('~model_name', default=None)
    model_type = rospy.get_param('~model_type', default="gp")
    if git_id is not None:
        load_options["git"] = git_id
    if model_name is not None:
        load_options["model_name"] = str(model_name)

    plot = True
    plot = rospy.get_param('~plot', default=None) if rospy.get_param('~plot', default=None) is not None else plot

    env = rospy.get_param('~environment', default='gazebo')
    default_quad = "hummingbird" if env == "gazebo" else "colibri"
    load_options["params"] = {env: "default"}

    if model_type == "rdrv":
        rdrv = load_rdrv(model_options=load_options)
    else:
        rdrv = None

    quad_name = rospy.get_param('~quad_name', default=None)
    quad_name = quad_name if quad_name is not None else default_quad

    # Change if needed. This is currently the supported combination.
    if env == "gazebo":
        assert quad_name == "hummingbird"
        ekf_sync = False
    else:
        assert quad_name == "colibri"
        ekf_sync = rospy.get_param('~use_ekf_synchronization', default=False)

    # Reset experiments switch
    reset = rospy.get_param('~reset_experiment', default=True)

    GPMPCWrapper(quad_name, env, recording_options, load_options, use_ekf=ekf_sync, rdrv=rdrv, plot=plot,
                 reset_experiment=reset)


if __name__ == "__main__":
    main()
