""" Miscellaneous visualization functions.

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
import tikzplotlib

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from config.configuration_parameters import DirectoryConfig as PathConfig
from src.utils.utils import v_dot_q, quaternion_to_euler, quaternion_inverse, q_dot_q, safe_mknode_recursive, \
    safe_mkdir_recursive
import os


def angle_to_rot_mat(angle):
    """
    Computes the 2x2 rotation matrix from the scalar angle
    :param angle: scalar angle in radians
    :return: the corresponding 2x2 rotation matrix
    """

    s = np.sin(angle)
    c = np.cos(angle)
    return np.array([[c, -s], [s, c]])


def draw_arrow(x_base, y_base, x_body, y_body):
    """
    Returns the coordinates for drawing a 2D arrow given its origin point and its length.
    :param x_base: x coordinate of the arrow origin
    :param y_base: y coordinate of the arrow origin
    :param x_body: x length of the arrow
    :param y_body: y length of the arrow
    :return: a tuple of x, y coordinates to plot the arrow
    """

    len_arrow = np.sqrt(x_body ** 2 + y_body ** 2)
    beta = np.arctan2(y_body, x_body)
    beta_rot = angle_to_rot_mat(beta)
    lower_arrow = beta_rot.dot(np.array([[-np.cos(np.pi / 6)], [-np.sin(np.pi / 6)]]) * len_arrow / 3)
    upper_arrow = beta_rot.dot(np.array([[-np.cos(np.pi / 6)], [np.sin(np.pi / 6)]]) * len_arrow / 3)

    return ([x_base, x_base + x_body, x_base + x_body + lower_arrow[0, 0],
             x_base + x_body, x_base + x_body + upper_arrow[0, 0]],
            [y_base, y_base + y_body, y_base + y_body + lower_arrow[1, 0],
             y_base + y_body, y_base + y_body + upper_arrow[1, 0]])


def draw_drone(pos, q_rot, x_f, y_f):

    # Define quadrotor extremities in body reference frame
    x1 = np.array([x_f[0], y_f[0], 0])
    x2 = np.array([x_f[1], y_f[1], 0])
    x3 = np.array([x_f[2], y_f[2], 0])
    x4 = np.array([x_f[3], y_f[3], 0])

    # Convert to world reference frame and add quadrotor center point:
    x1 = v_dot_q(x1, q_rot) + pos
    x2 = v_dot_q(x2, q_rot) + pos
    x3 = v_dot_q(x3, q_rot) + pos
    x4 = v_dot_q(x4, q_rot) + pos

    # Build set of coordinates for plotting
    return ([x1[0], x3[0], pos[0], x2[0], x4[0]],
            [x1[1], x3[1], pos[1], x2[1], x4[1]],
            [x1[2], x3[2], pos[2], x2[2], x4[2]])


def draw_covariance_ellipsoid(center, covar):
    """
    :param center: 3-dimensional array. Center of the ellipsoid
    :param covar: 3x3 covariance matrix. If the covariance is diagonal, the ellipsoid will have radii equal to the
    three diagonal axis along axes x, y, z respectively.
    :return:
    """

    # find the rotation matrix and radii of the axes
    _, radii, rotation = np.linalg.svd(covar)

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 20)
    v = np.linspace(0.0, np.pi, 20)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    x = np.reshape(x, -1)
    y = np.reshape(y, -1)
    z = np.reshape(z, -1)
    return x, y, z


def visualize_data_distribution(x_data, y_data, clusters, x_pruned, y_pruned):
    """
    Visualizes the distribution of the training dataset and the assignation of the GP prediction clusters.
    :param x_data: numpy array of shape N x 3, where N is the number of training points. Feature variables.
    :param y_data: numpy array of shape N x 3, where N is the number of training points. Regressed variables.
    :param x_pruned: numpy array of shape M x 3, where M is the number of pruned training points. Feature variables.
    :param y_pruned: numpy array of shape M x 3, where M is the number of pruned training points. Regressed variables.
    :param clusters: A dictionary where each entry is indexed by the cluster number, and contains a list of all the
    indices of the points in x_pruned belonging to that cluster.
    """

    if x_data.shape[1] < 3:
        return

    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    c = np.sqrt(np.sum(y_data ** 2, 1))
    scatter = ax.scatter(x_data[:, 0], x_data[:, 1], x_data[:, 2], c=c, alpha=0.6)
    ax.set_title('Raw data: Correction magnitude')
    ax.set_xlabel(r'$v_x\: [m/s]$')
    ax.set_ylabel(r'$v_y\: [m/s]$')
    ax.set_zlabel(r'$v_z\: [m/s]$')
    fig.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.75)

    ax = fig.add_subplot(132, projection='3d')
    c = np.sqrt(np.sum(y_pruned ** 2, 1))
    scatter = ax.scatter(x_pruned[:, 0], x_pruned[:, 1], x_pruned[:, 2], c=c, alpha=0.6)
    ax.set_title('Pruned data: Correction magnitude')
    ax.set_xlabel(r'$v_x\: [m/s]$')
    ax.set_ylabel(r'$v_y\: [m/s]$')
    ax.set_zlabel(r'$v_z\: [m/s]$')
    fig.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.75)

    n_clusters = len(clusters.keys())

    ax = fig.add_subplot(133, projection='3d')
    for i in range(int(n_clusters)):
        ax.scatter(x_pruned[clusters[i], 0], x_pruned[clusters[i], 1], x_pruned[clusters[i], 2], alpha=0.6)
    ax.set_title('Cluster assignations')
    ax.set_xlabel(r'$v_x\: [m/s]$')
    ax.set_ylabel(r'$v_y\: [m/s]$')
    ax.set_zlabel(r'$v_z\: [m/s]$')

    plt.show()


def visualize_gp_inference(x_data, u_data, y_data, gp_ensemble, vis_features_x, y_dims, labels):
    # WARNING: This function is extremely limited to the case where the regression is performed using just the
    # velocity state as input features and as output dimensions.

    predictions = gp_ensemble.predict(x_data.T, u_data.T)
    predictions = np.atleast_2d(np.atleast_2d(predictions["pred"])[y_dims])

    if len(vis_features_x) > 1:
        y_pred = np.sqrt(np.sum(predictions ** 2, 0))
        y_mse = np.sqrt(np.sum(y_data ** 2, 1))
    else:
        y_pred = predictions[0, :]
        y_mse = y_data[:, 0]

    v_min = min(np.min(y_pred), np.min(y_mse))
    v_max = max(np.max(y_pred), np.max(y_mse))

    fig = plt.figure()

    font_size = 16

    if len(vis_features_x) == 1:
        # Feature dimension is only 1

        # Compute windowed average
        n_bins = 20
        _, b = np.histogram(x_data[:, vis_features_x], bins=n_bins)
        hist_indices = np.digitize(x_data[:, vis_features_x], b)
        win_average = np.zeros(n_bins)
        for i in range(n_bins):
            win_average[i] = np.mean(y_mse[np.where(hist_indices == i + 1)[0]])
        bin_midpoints = b[:-1] + np.diff(b)[0] / 2

        ax = [fig.add_subplot(121), fig.add_subplot(122)]

        ax[0].scatter(x_data[:, vis_features_x], y_mse)
        ax[0].set_xlabel(labels[0])
        ax[0].set_ylabel('RMSE')
        ax[0].set_title('Post-processed dataset')

        ax[1].scatter(x_data[:, vis_features_x], y_pred, label='GP')
        ax[1].plot(bin_midpoints, win_average, label='window average')
        ax[1].set_xlabel(labels[0])
        ax[1].set_title('Predictions')
        ax[1].legend()

        return

    elif len(vis_features_x) >= 3:
        ax = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
        im = ax[0].scatter(x_data[:, vis_features_x[0]], x_data[:, vis_features_x[1]], x_data[:, vis_features_x[2]], c=y_mse,
                           cmap='viridis', alpha=0.6, vmin=v_min, vmax=v_max)
        ax[0].set_xlabel(labels[0], size=font_size - 4, labelpad=10)
        ax[0].set_ylabel(labels[1], size=font_size - 4, labelpad=10)
        ax[0].set_zlabel(labels[2], size=font_size - 4, labelpad=10)
        ax[0].set_title(r'Nominal MPC error $\|\mathbf{a}^e\|$', size=font_size)
        ax[0].view_init(65, 15)

        ax[1].scatter(x_data[:, vis_features_x[0]], x_data[:, vis_features_x[1]], x_data[:, vis_features_x[2]], c=y_pred,
                      cmap='viridis', alpha=0.6, vmin=v_min, vmax=v_max)
        ax[1].set_xlabel(labels[0], size=font_size - 4, labelpad=10)
        ax[1].set_ylabel(labels[1], size=font_size - 4, labelpad=10)
        ax[1].set_zlabel(labels[2], size=font_size - 4, labelpad=10)
        ax[1].set_title(r'GP prediction mangnitude $\|\tilde{\mathbf{a}}^e\|$', size=font_size)
        ax[1].view_init(65, 15)

        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        cbar = fig.add_axes([0.90, 0.05, 0.03, 0.8])
        fig.colorbar(im, cax=cbar)
        cbar.get_yaxis().labelpad = 15
        cbar.set_ylabel(r'$\|\mathbf{a}^e\|\left[\frac{m}{s^2}\right]$', size=font_size, labelpad=20, rotation=270)
        cbar.tick_params(labelsize=font_size - 4)

    # Create values for the regressed variables
    x = np.linspace(min(x_data[:, vis_features_x[0]]), max(x_data[:, vis_features_x[0]]), 100)
    y = np.linspace(min(x_data[:, vis_features_x[1]]), max(x_data[:, vis_features_x[1]]), 100)
    # x = np.linspace(-8, 8, 50)
    # y = np.linspace(-8, 8, 50)
    x_mesh, y_mesh = np.meshgrid(x, y)
    x = np.reshape(x_mesh, (-1, 1))
    y = np.reshape(y_mesh, (-1, 1))
    z = np.zeros_like(x)
    x_sample = np.concatenate((x, y, z), 1)

    # Generate complete mock x features. Only vis_features are non-zero
    x_mock = np.tile(np.zeros_like(z), (1, x_data.shape[1]))
    x_mock[:, np.array(vis_features_x)] = x_sample

    # Also create mock u features
    u_mock = np.tile(np.zeros_like(z), (1, u_data.shape[1]))

    if len(vis_features_x) != 3:
        plt.show()
        return

    # Generate animated plot showing prediction of the multiple clusters.
    # Cluster coloring only possible if all the output dimensions have exactly the same clusters.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print("Grid sampling...")
    outs = gp_ensemble.predict(x_mock.T, u_mock.T, return_gp_id=True, progress_bar=True)
    y_pred = np.atleast_2d(np.atleast_2d(outs["pred"])[y_dims])
    gp_ids = outs["gp_id"]
    y_sample = np.sqrt(np.sum(y_pred ** 2, 0))
    y_sample = np.reshape(y_sample, x_mesh.shape)

    gp_ids = np.reshape(gp_ids[next(iter(gp_ids))], x_mesh.shape)

    def init():
        # create the new map
        cmap = cm.get_cmap('jet')
        cmaplist = [cmap(j) for j in range(cmap.N)]
        cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

        # define the bins and normalize
        capped_n_clusters = min(np.amax(gp_ids) + 2, 20)
        bounds = np.linspace(0, np.amax(gp_ids) + 1, capped_n_clusters)
        norm = BoundaryNorm(bounds, cmap.N)

        my_col = cm.get_cmap('jet')(gp_ids / (np.amax(gp_ids) + 1))

        ax.plot_surface(x_mesh, y_mesh, y_sample, facecolors=my_col, linewidth=0, rstride=1, cstride=1,
                        antialiased=False, alpha=0.7, cmap=cmap, norm=norm)
        ax2 = fig.add_axes([0.90, 0.2, 0.03, 0.6])
        ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
        ax2.set_ylabel('Cluster assignment ID', size=14)
        ax2.tick_params(labelsize=16)

        ax.tick_params(labelsize=14)
        ax.set_xlabel(labels[0], size=16, labelpad=10)
        ax.set_ylabel(labels[1], size=16, labelpad=10)
        ax.set_zlabel(r'$\|\tilde{\mathbf{a}}^e\|\: \left[\frac{m}{s^2}\right]$', size=16, labelpad=10)
        ax.set_title(r'GP correction. Slice $v_z=0 \:\: \left[\frac{m}{s}\right]$', size=18)
        return fig,

    def animate(i):
        ax.view_init(elev=30., azim=i*3)
        return fig

    _ = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=False)

    plt.show()


def initialize_drone_plotter(world_rad, quad_rad, n_props, full_traj=None):

    fig = plt.figure(figsize=(10, 10), dpi=96)
    fig.show()

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    ax = fig.add_subplot(111, projection='3d')

    if full_traj is not None:
        ax.plot(full_traj[:, 0], full_traj[:, 1], full_traj[:, 2], '--', color='tab:blue', alpha=0.5)
        ax.set_xlim([ax.get_xlim()[0] - 2 * quad_rad, ax.get_xlim()[1] + 2 * quad_rad])
        ax.set_ylim([ax.get_ylim()[0] - 2 * quad_rad, ax.get_ylim()[1] + 2 * quad_rad])
        ax.set_zlim([ax.get_zlim()[0] - 2 * quad_rad, ax.get_zlim()[1] + 2 * quad_rad])
    else:
        ax.set_xlim([-world_rad, world_rad])
        ax.set_ylim([-world_rad, world_rad])
        ax.set_zlim([-world_rad, world_rad])

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    fig.canvas.draw()
    plt.draw()

    # cache the background
    background = fig.canvas.copy_from_bbox(ax.bbox)

    artists = {
        "trajectory": ax.plot([], [])[0], "drone": ax.plot([], [], 'o-')[0],
        "drone_x": ax.plot([], [], 'o-', color='r')[0],
        "missing_targets": ax.plot([], [], [], color='r', marker='o', linestyle='None', markersize=12)[0],
        "reached_targets": ax.plot([], [], [], color='g', marker='o', linestyle='None', markersize=12)[0],
        "sim_trajectory": [ax.plot([], [], [], '-', color='tab:blue', alpha=0.9 - i * 0.2 / n_props)[0]
                           for i in range(n_props)],
        "int_trajectory": [ax.plot([], [], [], '-', color='tab:orange', alpha=0.9 - i * 0.5 / n_props)[0]
                           for i in range(n_props + 1)],
        "prop_trajectory": [ax.plot([], [], [], '-', color='tab:red', alpha=0.9 - i * 0.2 / n_props)[0]
                            for i in range(n_props)],
        "prop_covariance": [ax.plot([], [], [], color='r', alpha=0.5 - i * 0.45 / n_props)[0]
                            for i in range(n_props)],
        "projection_lines": [ax.plot([], [], [], '-', color='tab:blue', alpha=0.2)[0],
                             ax.plot([], [], [], '-', color='tab:blue', alpha=0.2)[0]],
        "projection_target": [ax.plot([], [], [], marker='o', color='r', linestyle='None', alpha=0.2)[0],
                              ax.plot([], [], [], marker='o', color='r', linestyle='None', alpha=0.2)[0]]}

    art_pack = fig, ax, artists, background, world_rad
    return art_pack


def draw_drone_simulation(art_pack, x_trajectory, quad, targets, targets_reached, sim_traj=None,
                          int_traj=None, pred_traj=None, x_pred_cov=None, follow_quad=False):

    fig, ax, artists, background, world_rad = art_pack

    trajectories_artist = artists["trajectory"] if "trajectory" in artists.keys() else []
    projected_traj_artists = artists["projection_lines"] if "projection_lines" in artists.keys() else []
    drone_sketch_artist = artists["drone"] if "drone" in artists.keys() else []
    drone_sketch_artist_x_motor = artists["drone_x"] if "drone_x" in artists.keys() else[]
    targets_artist = artists["missing_targets"] if "missing_targets" in artists.keys() else []
    reached_targets_artist = artists["reached_targets"] if "reached_targets" in artists.keys() else []
    projected_tar_artists = artists["projection_target"] if "projection_target" in artists.keys() else []
    sim_traj_artists = artists["sim_trajectory"] if "sim_trajectory" in artists.keys() else []
    int_traj_artists = artists["int_trajectory"] if "int_trajectory" in artists.keys() else []
    pred_traj_artists = artists["prop_trajectory"] if "prop_trajectory" in artists.keys() else []
    cov_artists = artists["prop_covariance"] if "prop_covariance" in artists.keys() else []

    # restore background
    fig.canvas.restore_region(background)

    def draw_fading_traj(traj, traj_artists):
        traj = np.squeeze(np.array(traj))
        for j in range(min(traj.shape[0] - 1, len(traj_artists))):
            traj_artists[j].set_data([traj[j, 0], traj[j + 1, 0]], [traj[j, 1], traj[j + 1, 1]])
            traj_artists[j].set_3d_properties([traj[j, 2], traj[j + 1, 2]])

    # Draw missing and reached targets
    if targets is not None and targets_reached is not None:
        reached = targets[targets_reached, :]
        reached = reached[-2:, :]
        missing = targets[targets_reached == False, :]
        reached_targets_artist.set_data(reached[:, 0], reached[:, 1])
        reached_targets_artist.set_3d_properties(reached[:, 2])
        targets_artist.set_data(missing[:, 0], missing[:, 1])
        targets_artist.set_3d_properties(missing[:, 2])
        ax.draw_artist(targets_artist)
        ax.draw_artist(reached_targets_artist)

        # Draw projected target
        if missing.any():
            projected_tar_artists[0].set_data([missing[0, 0]], [ax.get_ylim()[1]])
            projected_tar_artists[0].set_3d_properties([missing[0, 2]])
            projected_tar_artists[1].set_data([ax.get_xlim()[0]], [missing[0, 1]])
            projected_tar_artists[1].set_3d_properties([missing[0, 2]])
            [ax.draw_artist(projected_tar_artist) for projected_tar_artist in projected_tar_artists]

    # Draw quadrotor trajectory
    trajectory_start_pt = max(len(x_trajectory) - 100, 0)
    trajectories_artist.set_data(x_trajectory[trajectory_start_pt:, 0], x_trajectory[trajectory_start_pt:, 1])
    trajectories_artist.set_3d_properties(x_trajectory[trajectory_start_pt:, 2])
    ax.draw_artist(trajectories_artist)

    # Draw projected trajectory
    projected_traj_artists[0].set_data(x_trajectory[trajectory_start_pt:, 0], ax.get_ylim()[1])
    projected_traj_artists[0].set_3d_properties(x_trajectory[trajectory_start_pt:, 2])
    projected_traj_artists[1].set_data(ax.get_xlim()[0], x_trajectory[trajectory_start_pt:, 1])
    projected_traj_artists[1].set_3d_properties(x_trajectory[trajectory_start_pt:, 2])
    [ax.draw_artist(projected_traj_artist) for projected_traj_artist in projected_traj_artists]

    # Draw drone art
    drone_art = draw_drone(x_trajectory[-1, 0:3], x_trajectory[-1, 3:7], quad.x_f, quad.y_f)
    drone_sketch_artist_x_motor.set_data(drone_art[0][0], drone_art[1][0])
    drone_sketch_artist_x_motor.set_3d_properties(drone_art[2][0])
    drone_sketch_artist.set_data(drone_art[0], drone_art[1])
    drone_sketch_artist.set_3d_properties(drone_art[2])
    ax.draw_artist(drone_sketch_artist)
    ax.draw_artist(drone_sketch_artist_x_motor)

    if int_traj is not None:
        draw_fading_traj(int_traj, int_traj_artists)
        for int_traj_artist in int_traj_artists:
            ax.draw_artist(int_traj_artist)

    if sim_traj is not None:
        draw_fading_traj(sim_traj, sim_traj_artists)
        for sim_traj_artist in sim_traj_artists:
            ax.draw_artist(sim_traj_artist)

    if pred_traj is not None:
        draw_fading_traj(pred_traj, pred_traj_artists)
        for pred_traj_artist in pred_traj_artists:
            ax.draw_artist(pred_traj_artist)

    if x_pred_cov is not None:
        n_std = 3
        x_std = np.sqrt(x_pred_cov[:, 0, 0]) * n_std
        y_std = np.sqrt(x_pred_cov[:, 1, 1]) * n_std
        z_std = np.sqrt(x_pred_cov[:, 2, 2]) * n_std
        for i, cov_artist in enumerate(cov_artists):
            center = pred_traj[i+1, 0:3]
            radii = np.diag(np.array([x_std[i], y_std[i], z_std[i]]))
            x, y, z = draw_covariance_ellipsoid(center, radii)
            cov_artist.set_data(x, y)
            cov_artist.set_3d_properties(z)
        for cov_artist in cov_artists:
            ax.draw_artist(cov_artist)

    if follow_quad:
        ax.set_xlim([x_trajectory[-1, 0] - world_rad, x_trajectory[-1, 0] + world_rad])
        ax.set_ylim([x_trajectory[-1, 1] - world_rad, x_trajectory[-1, 1] + world_rad])
        ax.set_zlim([x_trajectory[-1, 2] - world_rad, x_trajectory[-1, 2] + world_rad])

    # fill in the axes rectangle
    fig.canvas.blit(ax.bbox)
    
    
def trajectory_tracking_results(t_ref, x_ref, x_executed, u_ref, u_executed, title, w_control=None, legend_labels=None,
                                quat_error=True):

    if legend_labels is None:
        legend_labels = ['reference', 'simulated']

    with_ref = True if x_ref is not None else False

    fig, ax = plt.subplots(3, 4, sharex='all', figsize=(7, 9))

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    labels = ['x', 'y', 'z']
    for i in range(3):
        ax[i, 0].plot(t_ref, x_executed[:, i], label=legend_labels[1])
        if with_ref:
            ax[i, 0].plot(t_ref, x_ref[:, i], label=legend_labels[0])
        ax[i, 0].legend()
        ax[i, 0].set_ylabel(labels[i])
    ax[0, 0].set_title(r'$p\:[m]$')
    ax[2, 0].set_xlabel(r'$t [s]$')

    q_euler = np.stack([quaternion_to_euler(x_executed[j, 3:7]) for j in range(x_executed.shape[0])])
    for i in range(3):
        ax[i, 1].plot(t_ref, q_euler[:, i], label=legend_labels[1])
    if with_ref:
        ref_euler = np.stack([quaternion_to_euler(x_ref[j, 3:7]) for j in range(x_ref.shape[0])])
        q_err = []
        for i in range(t_ref.shape[0]):
            q_err.append(q_dot_q(x_executed[i, 3:7], quaternion_inverse(x_ref[i, 3:7])))
        q_err = np.stack(q_err)

        for i in range(3):
            ax[i, 1].plot(t_ref, ref_euler[:, i], label=legend_labels[0])
            if quat_error:
                ax[i, 1].plot(t_ref, q_err[:, i + 1], label='quat error')
    for i in range(3):
        ax[i, 1].legend()
    ax[0, 1].set_title(r'$\theta\:[rad]$')
    ax[2, 1].set_xlabel(r'$t [s]$')

    for i in range(3):
        ax[i, 2].plot(t_ref, x_executed[:, i + 7], label=legend_labels[1])
        if with_ref:
            ax[i, 2].plot(t_ref, x_ref[:, i + 7], label=legend_labels[0])
        ax[i, 2].legend()
    ax[0, 2].set_title(r'$v\:[m/s]$')
    ax[2, 2].set_xlabel(r'$t [s]$')

    for i in range(3):
        ax[i, 3].plot(t_ref, x_executed[:, i + 10], label=legend_labels[1])
        if with_ref:
            ax[i, 3].plot(t_ref, x_ref[:, i + 10], label=legend_labels[0])
        if w_control is not None:
            ax[i, 3].plot(t_ref, w_control[:, i], label='control')
        ax[i, 3].legend()
    ax[0, 3].set_title(r'$\omega\:[rad/s]$')
    ax[2, 3].set_xlabel(r'$t [s]$')

    plt.suptitle(title)

    if u_ref is not None and u_executed is not None:
        ax = plt.subplots(1, 4, sharex="all", sharey="all")[1]
        for i in range(4):
            ax[i].plot(t_ref, u_ref[:, i], label='ref')
            ax[i].plot(t_ref, u_executed[:, i], label='simulated')
            ax[i].set_xlabel(r'$t [s]$')
            tit = 'Control %d' % (i + 1)
            ax[i].set_title(tit)
            ax[i].legend()
    plt.show()


def mse_tracking_experiment_plot(v_max, mse, traj_type_vec, train_samples_vec, legends, y_labels, t_opt=None,
                                 font_size=16):

    # Check if there is the variants dimension in the data
    if len(mse.shape) == 4:
        variants_dim = mse.shape[3]
    else:
        variants_dim = 1

    fig, axes = plt.subplots(variants_dim, len(traj_type_vec), sharex='col', sharey='none',
                             figsize=(17, 2.5 * variants_dim + 2))
    if variants_dim == 1 and len(traj_type_vec) > 1:
        axes = axes[np.newaxis, :]
    elif variants_dim == 1:
        axes = np.expand_dims(axes, 0)
        axes = np.expand_dims(axes, 0)
    elif len(traj_type_vec) == 1:
        axes = axes[:, np.newaxis]

    for seed_id, track_seed in enumerate(traj_type_vec):
        for j in range(variants_dim):
            for i, _ in enumerate(train_samples_vec):
                mse_data = mse[seed_id, :, i, j] if len(mse.shape) == 4 else mse[seed_id, :, i]
                label = legends[i] if seed_id == 0 and j == 0 else None
                if legends[i] == 'perfect':
                    axes[j, seed_id].plot(v_max[seed_id, :], mse_data, '--o', linewidth=4, label=label)
                else:
                    axes[j, seed_id].plot(v_max[seed_id, :], mse_data, '--o', label=label)
            if seed_id == 0:
                axes[j, seed_id].set_ylabel(y_labels[j], size=font_size)
            if j == 0:
                axes[j, seed_id].set_title('RMSE [m] | ' + str(track_seed), size=font_size+2)

            axes[j, seed_id].grid()
            axes[j, seed_id].tick_params(labelsize=font_size)

        axes[variants_dim - 1, seed_id].set_xlabel('max vel [m/s]', size=font_size)

    legend_cols = len(train_samples_vec)
    fig.legend(loc="upper center", fancybox=True, borderaxespad=0.05, ncol=legend_cols, mode="expand",
               title_fontsize=font_size, fontsize=font_size - 4)
    plt.tight_layout(h_pad=1.4)
    plt.subplots_adjust(top=0.7 + 0.05 * variants_dim)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_save_dir = dir_path + '/../../results/images/'
    safe_mkdir_recursive(img_save_dir, overwrite=False)

    tikzplotlib.save(img_save_dir + "mse.tex")
    fig.savefig(img_save_dir + 'mse', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)

    if t_opt is None:
        return

    v = v_max.reshape(-1)
    ind_v = np.argsort(v, axis=0)

    fig = plt.figure(figsize=(17, 4.5))
    for i, n_train in enumerate(train_samples_vec):
        plt.plot(v[ind_v], t_opt.reshape(t_opt.shape[0] * t_opt.shape[1], -1)[ind_v, i], label=legends[i])
    fig.legend(loc="upper center", fancybox=True, borderaxespad=0.05, ncol=legend_cols, mode="expand",
               title_fontsize=font_size, fontsize=font_size)
    plt.ylabel('Mean MPC loop time (s)', fontsize=font_size)
    plt.xlabel('Max vel [m/s]', fontsize=font_size)

    fig.savefig(img_save_dir + 't_opt', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)


def load_past_experiments():

    metadata_file, mse_file, v_file, t_opt_file = get_experiment_files()

    try:
        with open(metadata_file) as json_file:
            metadata = json.load(json_file)
    except FileNotFoundError:
        metadata = None

    mse = np.load(mse_file)
    v = np.load(v_file)
    t_opt = np.load(t_opt_file)

    return metadata, mse, v, t_opt


def get_experiment_files():
    results_path = PathConfig.RESULTS_DIR
    metadata_file = os.path.join(results_path, 'experiments', 'metadata.json')
    mse_file = os.path.join(results_path, 'experiments', 'mse.npy')
    mean_v_file = os.path.join(results_path, 'experiments', 'mean_v.npy')
    t_opt_file = os.path.join(results_path, 'experiments', 't_opt.npy')

    if not os.path.exists(metadata_file):
        safe_mknode_recursive(os.path.join(results_path, 'experiments'), 'metadata.json', overwrite=False)
    if not os.path.exists(mse_file):
        safe_mknode_recursive(os.path.join(results_path, 'experiments'), 'mse.npy', overwrite=False)
    if not os.path.exists(mean_v_file):
        safe_mknode_recursive(os.path.join(results_path, 'experiments'), 'mean_v.npy', overwrite=False)
    if not os.path.exists(t_opt_file):
        safe_mknode_recursive(os.path.join(results_path, 'experiments'), 't_opt.npy', overwrite=False)

    return metadata_file, mse_file, mean_v_file, t_opt_file
