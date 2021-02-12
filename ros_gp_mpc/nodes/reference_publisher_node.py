#!/usr/bin/env python3.6
""" Node wrapper for publishing trajectories for the MPC pipeline to track.

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

from std_msgs.msg import Bool
from ros_gp_mpc.msg import ReferenceTrajectory
from src.quad_mpc.create_ros_gp_mpc import custom_quad_param_loader
from src.utils.trajectories import loop_trajectory, random_trajectory, lemniscate_trajectory
import numpy as np
import rospy


class ReferenceGenerator:

    def __init__(self):

        self.gp_mpc_busy = True

        rospy.init_node("reference_generator")

        plot = rospy.get_param('~plot', default=True)

        quad_name = rospy.get_param('~quad_name', default='hummingbird')
        quad = custom_quad_param_loader(quad_name)

        # Configuration for random flight mode
        n_seeds = rospy.get_param('~n_seeds', default=1)
        v_list = rospy.get_param('~v_list', default=[3.5])
        if isinstance(v_list, str):
            v_list = v_list.split('[')[1].split(']')[0]
            v_list = [float(v.strip()) for v in v_list.split(',')]

        # Select if generate "random" trajectories, "hover" mode or increasing speed "loop" mode
        mode = rospy.get_param('~mode', default="random")
        if mode != "random":
            n_seeds = 1

        # Load parameters of loop trajectory
        loop_r = rospy.get_param('~loop_r', default=1.5)
        loop_z = rospy.get_param('~loop_z', default=1)
        loop_v_max = rospy.get_param('~loop_v_max', default=3)
        loop_a = rospy.get_param('~loop_lin_a', default=0.075)
        loop_cc = rospy.get_param('~loop_clockwise', default=True)
        loop_yawing = rospy.get_param('~loop_yawing', default=True)

        # Load world limits if any
        map_limits = rospy.get_param('~world_limits', default=None)

        # Control at 50 hz. Use time horizon=1 and 10 nodes
        n_mpc_nodes = rospy.get_param('~n_nodes', default=10)
        t_horizon = rospy.get_param('~t_horizon', default=1.0)
        control_freq_factor = rospy.get_param('~control_freq_factor', default=5 if quad_name == "hummingbird" else 6)
        opt_dt = t_horizon / (n_mpc_nodes * control_freq_factor)

        reference_topic = "reference"
        status_topic = "busy"
        reference_pub = rospy.Publisher(reference_topic, ReferenceTrajectory, queue_size=1)
        rospy.Subscriber(status_topic, Bool, self.status_callback)

        v_ind = 0
        seed = 0

        # Calculate total number of trajectories
        n_trajectories = n_seeds * len(v_list)
        curr_trajectory_ind = 0

        rate = rospy.Rate(0.2)
        while not rospy.is_shutdown():

            if not self.gp_mpc_busy and mode == "hover":
                rospy.loginfo("Sending hover-in-place command")
                msg = ReferenceTrajectory()
                reference_pub.publish(msg)
                rospy.signal_shutdown("All trajectories were sent to the MPC")
                break

            if not self.gp_mpc_busy and curr_trajectory_ind == n_trajectories:
                msg = ReferenceTrajectory()
                reference_pub.publish(msg)
                rospy.signal_shutdown("All trajectories were sent to the MPC")
                break

            if not self.gp_mpc_busy and mode == "loop":
                rospy.loginfo("Sending increasing speed loop trajectory")
                x_ref, t_ref, u_ref = loop_trajectory(quad, opt_dt, v_max=loop_v_max, radius=loop_r, z=loop_z,
                                                      lin_acc=loop_a, clockwise=loop_cc, map_name=map_limits,
                                                      yawing=loop_yawing, plot=plot)

                msg = ReferenceTrajectory()
                msg.traj_name = "circle"
                msg.v_input = loop_v_max
                msg.seq_len = x_ref.shape[0]
                msg.trajectory = np.reshape(x_ref, (-1,)).tolist()
                msg.dt = t_ref.tolist()
                msg.inputs = np.reshape(u_ref, (-1,)).tolist()

                reference_pub.publish(msg)
                curr_trajectory_ind += 1
                self.gp_mpc_busy = True

            elif not self.gp_mpc_busy and mode == "lemniscate":
                rospy.loginfo("Sending increasing speed lemniscate trajectory")
                x_ref, t_ref, u_ref = lemniscate_trajectory(quad, opt_dt, v_max=loop_v_max, radius=loop_r, z=loop_z,
                                                            lin_acc=loop_a, clockwise=loop_cc, map_name=map_limits,
                                                            yawing=loop_yawing, plot=plot)

                msg = ReferenceTrajectory()
                msg.traj_name = "lemniscate"
                msg.v_input = loop_v_max
                msg.seq_len = x_ref.shape[0]
                msg.trajectory = np.reshape(x_ref, (-1,)).tolist()
                msg.dt = t_ref.tolist()
                msg.inputs = np.reshape(u_ref, (-1,)).tolist()

                reference_pub.publish(msg)
                curr_trajectory_ind += 1
                self.gp_mpc_busy = True

            elif not self.gp_mpc_busy and mode == "random":

                speed = v_list[v_ind]
                log_msg = "Random trajectory generator %d/%d. Seed: %d. Mean vel: %.3f m/s" % \
                          (curr_trajectory_ind + 1, n_trajectories, seed, speed)
                rospy.loginfo(log_msg)

                x_ref, t_ref, u_ref = random_trajectory(quad, opt_dt, seed=seed, speed=speed, map_name=map_limits,
                                                        plot=plot)
                msg = ReferenceTrajectory()
                msg.traj_name = "random_" + str(seed)
                msg.v_input = speed
                msg.seq_len = x_ref.shape[0]
                msg.trajectory = np.reshape(x_ref, (-1, )).tolist()
                msg.dt = t_ref.tolist()
                msg.inputs = np.reshape(u_ref, (-1, )).tolist()

                reference_pub.publish(msg)
                curr_trajectory_ind += 1
                self.gp_mpc_busy = True

                if v_ind + 1 < len(v_list):
                    v_ind += 1
                else:
                    seed += 1
                    v_ind = 0

            elif not self.gp_mpc_busy:
                raise ValueError("Unknown trajectory type: %s" % mode)

            rate.sleep()

    def status_callback(self, msg):
        """
        Callback function for tracking if the gp_mpc node is busy
        :param msg: Message from the subscriber
        :type msg: Bool
        """
        self.gp_mpc_busy = msg.data


if __name__ == "__main__":

    ReferenceGenerator()
