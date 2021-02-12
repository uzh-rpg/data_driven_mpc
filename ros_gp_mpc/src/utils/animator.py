""" Class for generating a comprehensive post-processed visualization of experimental flight results.

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

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from config.configuration_parameters import DirectoryConfig


class Dynamic3DTrajectory:
    def __init__(self, pos_data, vel_data, pos_ref, vel_ref, t_vec_ref, legends, sparsing_factor=1):

        # Add reference to data so it also moves
        pos_data = pos_ref + pos_data
        self.data = np.array(pos_data)

        self.reference = pos_ref[0]
        self.data_len = len(pos_data[0])
        self.n_lines = len(pos_data)

        self.t_vec = t_vec_ref - t_vec_ref[0]

        self.legends = legends

        vel_data = vel_ref + vel_data
        self.vel_data = np.array(vel_data)

        if sparsing_factor == 0:
            sparse_data = np.arange(0, self.data_len)
        else:
            sparse_data = np.arange(0, self.data_len, sparsing_factor)

        self.sparsed_data = self.data[:, sparse_data, :]
        self.vel_data = self.vel_data[:, sparse_data, :]
        self.t_vec = self.t_vec[sparse_data]

        self.max_buffer_size = 50
        self.colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        self.data_len = len(sparse_data)

        x_data = np.concatenate(tuple([dat[sparse_data, 0] for dat in pos_data]))
        y_data = np.concatenate(tuple([dat[sparse_data, 1] for dat in pos_data]))
        z_data = np.concatenate(tuple([dat[sparse_data, 2] for dat in pos_data]))

        self.max_x = np.max(x_data)
        self.min_x = np.min(x_data)
        self.max_y = np.max(y_data)
        self.min_y = np.min(y_data)
        self.max_z = np.max(z_data)
        self.min_z = np.min(z_data)

        # Make dimensions more similar and add a bit of padding
        range_x = self.max_x - self.min_x
        range_y = self.max_y - self.min_y
        range_z = self.max_z - self.min_z
        max_range = max(range_x, range_y, range_z)
        self.max_x = self.max_x + 0.25 * (max_range - range_x) + (self.max_x - self.min_x) * 0.2
        self.min_x = self.min_x - 0.25 * (max_range - range_x) - (self.max_x - self.min_x) * 0.2
        self.max_y = self.max_y + 0.25 * (max_range - range_y) + (self.max_y - self.min_y) * 0.2
        self.min_y = self.min_y - 0.25 * (max_range - range_y) - (self.max_y - self.min_y) * 0.2
        self.max_z = self.max_z + 0.25 * (max_range - range_z) + (self.max_z - self.min_z) * 0.2
        self.min_z = self.min_z - 0.25 * (max_range - range_z) - (self.max_z - self.min_z) * 0.2

        self.figure = None
        self.ax = None
        self.pos_err_ax = None
        self.speed_ax = None
        self.top_down_ax = None
        self.x_time_ax = None
        self.y_time_ax = None

        self.lines = None

        self.n_3d_lines = None
        self.vel_bars_0_idx = None
        self.pos_err_0_idx = None
        self.top_down_0_idx = None
        self.x_time_0_idx = None
        self.y_time_0_idx = None

    def on_launch(self):
        self.figure = plt.figure(figsize=(14, 10))

        self.ax = axes3d.Axes3D(self.figure, rect=(-0.02, 0.3, 0.65, 0.7))

        self.ax.set_zlim3d([self.min_z, self.max_z])
        self.ax.set_ylim3d([self.min_y, self.max_y])
        self.ax.set_xlim3d([self.min_x, self.max_x])
        self.ax.set_xlabel(r'$p_x\: [m]$')
        self.ax.set_ylabel(r'$p_y\: [m]$')
        self.ax.set_zlabel(r'$p_z\: [m]$')

        self.ax.plot(self.reference[:, 0], self.reference[:, 1], self.reference[:, 2], '-', alpha=0.2)

        self.lines = sum([self.ax.plot([], [], [], '-', c=self.colors[i], label=self.legends[i])
                          for i in range(self.n_lines)], [])
        projection_lines = sum([self.ax.plot([], [], [], '-', c=self.colors[i], alpha=0.2)
                                for i in range(self.n_lines)], [])
        projection_lines += sum([self.ax.plot([], [], [], '-', c=self.colors[i], alpha=0.2)
                                 for i in range(self.n_lines)], [])
        projection_lines += sum([self.ax.plot([], [], [], '-', c=self.colors[i], alpha=0.2)
                                 for i in range(self.n_lines)], [])

        self.lines += projection_lines

        pos_balls = sum([self.ax.plot([], [], [], 'o', c=self.colors[i]) for i in range(self.n_lines)], [])
        self.lines += pos_balls
        self.n_3d_lines = len(self.lines)

        self.ax.legend()
        self.ax.set_title('3D Visualization')

        self.vel_bars_0_idx = len(self.lines)
        bar_height = 0.1

        self.speed_ax = plt.axes((0.65, 0.8, 0.3, 0.05))
        vel_bar = [self.speed_ax.barh([0], [0], color=self.colors[0], height=bar_height)[0]]
        self.speed_ax.set_xlim([0, np.max(np.sqrt(np.sum(self.vel_data[0] ** 2, -1))) * 1.05])
        self.speed_ax.set_ylim([-bar_height, bar_height * 1.05])

        self.speed_ax.set_xlabel(r'$\|\mathbf{v}\|\:[m/s]$')
        self.speed_ax.set_yticks([0])
        self.speed_ax.set_yticklabels([self.legends[0]])
        self.speed_ax.grid()
        self.speed_ax.set_title('Quadrotor speed')

        self.lines += vel_bar

        self.pos_err_ax = plt.axes((0.65, 0.6, 0.3, 0.1))
        self.pos_err_ax.grid()
        pos_err = []
        p_errors = np.stack([self.data[0] - self.data[i+1] for i in range(self.n_lines - 1)])

        pos_err += [self.pos_err_ax.barh([(i - 1) * bar_height * 1.05], [i], color=self.colors[i], height=bar_height)[0]
                    for i in range(1, self.n_lines)]
        self.pos_err_ax.set_xlim([np.min(p_errors) * 1.05, np.max(p_errors) * 1.05])
        self.pos_err_ax.set_ylim([-bar_height, ((self.n_lines - 1) * 1.05) * bar_height])
        self.pos_err_ax.set_title('XY position error')
        self.pos_err_ax.set_xlabel(r'$\|\mathbf{p}^* - \mathbf{p}\|\: [m]$')
        self.pos_err_ax.set_yticks([i * bar_height * 1.05 for i in range(self.n_lines - 1)])
        self.pos_err_ax.set_yticklabels(self.legends[1:])

        self.pos_err_0_idx = len(self.lines)
        self.lines += pos_err

        self.top_down_ax = plt.axes((0.65, 0.08, 0.3, 0.4))
        top_down_lines = sum([self.top_down_ax.plot([], [], color=self.colors[i], label=self.legends[i])
                              for i in range(self.n_lines)], [])
        top_down_lines += sum([self.top_down_ax.plot([], [], 'o', color=self.colors[i])
                               for i in range(self.n_lines)], [])
        self.top_down_ax.plot(self.reference[:, 0], self.reference[:, 1], '-', alpha=0.2)
        self.top_down_0_idx = len(self.lines)
        self.top_down_ax.grid()
        self.top_down_ax.set_xlim([self.min_x, self.max_x])
        self.top_down_ax.set_ylim([self.min_y, self.max_y])
        self.top_down_ax.set_title('Top down view')
        self.top_down_ax.set_xlabel(r'$p_x\:[m]$')
        self.top_down_ax.set_ylabel(r'$p_y\:[m]$')
        self.top_down_ax.legend()
        self.lines += top_down_lines

        self.x_time_ax = plt.axes((0.05, 0.08, 0.26, 0.2))
        self.x_time_ax.axhline(y=0, linestyle='--', alpha=0.5)
        x_time_lines = sum([self.x_time_ax.plot([], [], color=self.colors[i], label=self.legends[i])
                            for i in range(1, self.n_lines)], [])
        self.x_time_0_idx = len(self.lines)
        self.lines += x_time_lines
        self.x_time_ax.set_xlim([self.t_vec[0], self.t_vec[-1]])
        self.x_time_ax.set_ylim([0, np.max(np.abs(p_errors[:, :, 0])) * 1.05])
        self.x_time_ax.grid()
        self.x_time_ax.set_title(r'$p_x$ error')
        self.x_time_ax.set_ylabel('pos [m]')
        self.x_time_ax.legend(loc='upper right')

        self.y_time_ax = plt.axes((0.34, 0.08, 0.26, 0.2))
        self.y_time_ax.axhline(y=0, linestyle='--', alpha=0.5)
        y_time_lines = sum([self.y_time_ax.plot([], [], color=self.colors[i], label=self.legends[i])
                            for i in range(1, self.n_lines)], [])
        self.y_time_0_idx = len(self.lines)
        self.lines += y_time_lines
        self.y_time_ax.set_xlim([self.t_vec[0], self.t_vec[-1]])
        self.y_time_ax.set_ylim([0, np.max(np.abs(p_errors[:, :, 1])) * 1.05])
        self.y_time_ax.set_xlabel('Time [s]')
        self.y_time_ax.set_title(r'$p_y$ error')
        self.y_time_ax.grid()
        self.y_time_ax.legend(loc='upper right')

    def on_init(self):
        for i in range(self.n_3d_lines):
            self.lines[i].set_data([], [])
            self.lines[i].set_3d_properties([])

        self.lines[self.vel_bars_0_idx].set_width(0)

        for i in range(self.n_lines - 1):
            self.lines[self.pos_err_0_idx + i].set_width(0)

        for i in range(self.n_lines * 2):
            self.lines[self.top_down_0_idx + i].set_data([], [])

        for i in range(self.n_lines - 1):
            self.lines[self.x_time_0_idx + i].set_data([], [])
            self.lines[self.y_time_0_idx + i].set_data([], [])

        return self.lines

    def animate(self, i):
        i = (2 * i) % self.data.shape[1]
        for j, (line, xi) in enumerate(zip(self.lines[:self.n_lines], self.sparsed_data)):
            x, y, z = xi[:i].T

            if len(x) > self.max_buffer_size:
                x = x[len(x) - self.max_buffer_size:]
                y = y[len(y) - self.max_buffer_size:]
                z = z[len(z) - self.max_buffer_size:]

            line.set_data(x, y)
            line.set_3d_properties(z)

            self.lines[j + self.n_lines].set_data(x, self.max_y)
            self.lines[j + self.n_lines].set_3d_properties(z)

            self.lines[j + 2 * self.n_lines].set_data(np.ones(len(y)) * self.min_x, y)
            self.lines[j + 2 * self.n_lines].set_3d_properties(z)

            self.lines[j + 3 * self.n_lines].set_data(x, y)
            self.lines[j + 3 * self.n_lines].set_3d_properties(self.min_z)

            if len(x) > 0:
                self.lines[j + 4 * self.n_lines].set_data(x[-1], y[-1])
                self.lines[j + 4 * self.n_lines].set_3d_properties(z[-1])

                self.lines[self.top_down_0_idx + j + self.n_lines].set_data(x[-1], y[-1])

            self.lines[self.top_down_0_idx + j].set_data(x, y)

        self.lines[self.vel_bars_0_idx].set_width(np.sqrt(np.sum(self.vel_data[0, i, :] ** 2)))

        for j in range(self.n_lines - 1):
            self.lines[self.pos_err_0_idx + j].set_width(
                np.sqrt(np.sum((self.data[0][i, :2] - self.data[j+1][i, :2]) ** 2)))

        for j in reversed(range(self.n_lines - 1)):
            self.lines[self.x_time_0_idx + j].set_data(self.t_vec[:i], np.abs(self.data[0, :i, 0] - self.data[j + 1, :i, 0]))
            self.lines[self.y_time_0_idx + j].set_data(self.t_vec[:i], np.abs(self.data[0, :i, 1] - self.data[j + 1, :i, 1]))

        return self.lines

    def __call__(self, save=False):
        self.on_launch()

        ani = animation.FuncAnimation(self.figure, self.animate, init_func=self.on_init, frames=self.data_len,
                                      interval=5, blit=True, repeat=False)

        if save:
            results_dir = DirectoryConfig.RESULTS_DIR
            plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
            writer = animation.FFMpegWriter(fps=30)
            ani.save(results_dir + '/animation.mp4', writer=writer)

        else:
            plt.show()
