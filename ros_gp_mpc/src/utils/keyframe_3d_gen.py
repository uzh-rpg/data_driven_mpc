""" Generates a set of keypoints to generate a piece-wise polynomial trajectory between each pair.

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


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def apply_map_limits(x, y, z, limits):

    # Find out which axis is the most constrained
    x_max_range = limits["x"][1] - limits["x"][0]
    y_max_range = limits["y"][1] - limits["y"][0]
    z_max_range = limits["z"][1] - limits["z"][0]

    x_actual_range = np.max(x) - np.min(x)
    y_actual_range = np.max(y) - np.min(y)
    z_actual_range = np.max(z) - np.min(z)

    # One or more of the ranges violates the constraints.
    if x_actual_range > x_max_range or y_actual_range > y_max_range or z_actual_range > z_max_range:
        shrink_ratio = max(x_actual_range / x_max_range, y_actual_range / y_max_range, z_actual_range / z_max_range)
        x = (x - np.mean(x)) / shrink_ratio
        y = (y - np.mean(y)) / shrink_ratio
        z = (z - np.mean(z)) / shrink_ratio

        x += (limits["x"][0] - np.min(x))
        y += (limits["y"][0] - np.min(y))
        z += (limits["z"][0] - np.min(z))

    return x, y, z


def center_and_scale(x, y, z, x_max, x_min, y_max, y_min, z_max, z_min):

    x -= (x_min + (x_max - x_min) / 2)
    y -= (y_min + (y_max - y_min) / 2)
    z -= (z_min + (z_max - z_min) / 2)

    scaling = np.mean([x_max - x_min, y_max - y_min, z_max - z_min])
    x = x * 6 / scaling
    y = y * 6 / scaling
    z = z * 6 / scaling

    return x, y, z


def random_periodical_trajectory(plot=False, random_state=None, map_limits=None):

    if random_state is None:
        random_state = np.random.randint(0, 9999)

    kernel_z = ExpSineSquared(length_scale=5.5, periodicity=60)
    kernel_y = ExpSineSquared(length_scale=4.5, periodicity=30) + ExpSineSquared(length_scale=4.0, periodicity=15)
    kernel_x = ExpSineSquared(length_scale=4.5, periodicity=30) + ExpSineSquared(length_scale=4.5, periodicity=60)

    gp_x = GaussianProcessRegressor(kernel=kernel_x)
    gp_y = GaussianProcessRegressor(kernel=kernel_y)
    gp_z = GaussianProcessRegressor(kernel=kernel_z)

    # High resolution sampling for track boundaries
    inputs_x = np.linspace(0, 60, 100)
    inputs_y = np.linspace(0, 30, 100)
    inputs_z = np.linspace(0, 60, 100)

    x_sample_hr = gp_x.sample_y(inputs_x[:, np.newaxis], 1, random_state=random_state)
    y_sample_hr = gp_y.sample_y(inputs_y[:, np.newaxis], 1, random_state=random_state)
    z_sample_hr = gp_z.sample_y(inputs_z[:, np.newaxis], 1, random_state=random_state)

    max_x_coords = np.max(x_sample_hr, 0)
    max_y_coords = np.max(y_sample_hr, 0)
    max_z_coords = np.max(z_sample_hr, 0)

    min_x_coords = np.min(x_sample_hr, 0)
    min_y_coords = np.min(y_sample_hr, 0)
    min_z_coords = np.min(z_sample_hr, 0)

    x_sample_hr, y_sample_hr, z_sample_hr = center_and_scale(
        x_sample_hr, y_sample_hr, z_sample_hr,
        max_x_coords, min_x_coords, max_y_coords, min_y_coords, max_z_coords, min_z_coords)

    # Additional constraint on map limits
    if map_limits is not None:
        x_sample_hr, y_sample_hr, z_sample_hr = apply_map_limits(x_sample_hr, y_sample_hr, z_sample_hr, map_limits)

    # Low resolution for control points
    lr_ind = np.linspace(0, len(x_sample_hr) - 1, 10, dtype=int)
    lr_ind[-1] = 0
    x_sample_lr = x_sample_hr[lr_ind, :]
    y_sample_lr = y_sample_hr[lr_ind, :]
    z_sample_lr = z_sample_hr[lr_ind, :]

    x_sample_diff = x_sample_hr[lr_ind + 1, :] - x_sample_lr
    y_sample_diff = y_sample_hr[lr_ind + 1, :] - y_sample_lr
    z_sample_diff = z_sample_hr[lr_ind + 1, :] - z_sample_lr

    # Get angles at keypoints
    a_x = np.arctan2(z_sample_diff, y_sample_diff) * 0
    a_y = np.arctan2(x_sample_diff, z_sample_diff) * 0
    a_z = (np.arctan2(y_sample_diff, x_sample_diff) - np.pi/4) * 0

    if plot:
        # Plot checking
        # Build rotation matrices
        rx = [np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]) for a in a_x[:, 0]]
        ry = [np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) for a in a_y[:, 0]]
        rz = [np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) for a in a_z[:, 0]]

        main_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        quiver_axes = np.zeros((len(lr_ind), 3, 3))

        for i in range(len(lr_ind)):
            r_mat = rz[i].dot(ry[i].dot(rx[i]))
            rot_body = r_mat.dot(main_axes)
            quiver_axes[i, :, :] = rot_body

        shortest_axis = min(np.max(x_sample_hr) - np.min(x_sample_hr),
                            np.max(y_sample_hr) - np.min(y_sample_hr),
                            np.max(z_sample_hr) - np.min(z_sample_hr))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_sample_lr, y_sample_lr, z_sample_lr)
        ax.plot(x_sample_hr[:, 0], y_sample_hr[:, 0], z_sample_hr[:, 0], '-', alpha=0.5)
        ax.quiver(x_sample_lr[:, 0], y_sample_lr[:, 0], z_sample_lr[:, 0],
                  x_sample_diff[:, 0], y_sample_diff[:, 0], z_sample_diff[:, 0], color='g',
                  length=shortest_axis / 6, normalize=True, label='traj. norm')
        ax.quiver(x_sample_lr, y_sample_lr, z_sample_lr,
                  quiver_axes[:, 0, :], quiver_axes[:, 1, :], quiver_axes[:, 2, :], color='b',
                  length=shortest_axis / 6, normalize=True, label='quad. att.')
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=16)
        ax.set_xlabel('x [m]', size=16, labelpad=10)
        ax.set_ylabel('y [m]', size=16, labelpad=10)
        ax.set_zlabel('z [m]', size=16, labelpad=10)
        ax.set_xlim([np.min(x_sample_hr), np.max(x_sample_hr)])
        ax.set_ylim([np.min(y_sample_hr), np.max(y_sample_hr)])
        ax.set_zlim([np.min(z_sample_hr), np.max(z_sample_hr)])
        ax.set_title('Source keypoints', size=18)
        plt.show()

    curve = np.concatenate((x_sample_lr, y_sample_lr, z_sample_lr), 1)
    attitude = np.concatenate((a_x, a_y, a_z), 1)

    return curve, attitude


if __name__ == "__main__":
    limits = {
        "x": [-0.6, 4],
        "y": [-2, 2],
        "z": [0.1, 2]
    }
    random_periodical_trajectory(plot=True, map_limits=limits)
