""" Set of tunable parameters for the Simplified Simulator and model fitting.

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


class DirectoryConfig:
    """
    Class for storing directories within the package
    """

    _dir_path = os.path.dirname(os.path.realpath(__file__))
    SAVE_DIR = _dir_path + '/../results/model_fitting'
    RESULTS_DIR = _dir_path + '/../results'
    CONFIG_DIR = _dir_path + ''
    DATA_DIR = _dir_path + '/../data'


class SimpleSimConfig:
    """
    Class for storing the Simplified Simulator configurations.
    """

    # Set to True to show a real-time Matplotlib animation of the experiments for the Simplified Simulator. Execution
    # will be slower if the GUI is turned on. Note: setting to True may require some further library installation work.
    custom_sim_gui = False

    # Set to True to display a plot describing the trajectory tracking results after the execution.
    result_plots = False

    # Set to True to show the trajectory that will be executed before the execution time
    pre_run_debug_plots = False

    # Choice of disturbances modeled in our Simplified Simulator. For more details about the parameters used refer to
    # the script: src/quad_mpc/quad_3d.py.
    simulation_disturbances = {
        "noisy": True,                       # Thrust and torque gaussian noises
        "drag": True,                        # 2nd order polynomial aerodynamic drag effect
        "payload": False,                    # Payload force in the Z axis
        "motor_noise": True                  # Asymmetric voltage noise in the motors
    }


class ModelFitConfig:
    """
    Class for storing flags for the model fitting scripts.
    """

    # ## Dataset loading ## #
    ds_name = "simplified_sim_dataset"
    ds_metadata = {
        "noisy": True,
        "drag": True,
        "payload": False,
        "motor_noise": True
    }

    # ds_metadata = {
    #     "gazebo": "default",
    # }

    # ## Visualization ## #
    # Training mode
    visualize_training_result = True
    visualize_data = False

    # Visualization mode
    grid_sampling_viz = True
    x_viz = [7, 8, 9]
    u_viz = []
    y_viz = [7, 8, 9]

    # ## Data post-processing ## #
    histogram_bins = 40              # Cluster data using histogram binning
    histogram_threshold = 0.001      # Remove bins where the total ratio of data is lower than this threshold
    velocity_cap = 16                # Also remove datasets point if abs(velocity) > x_cap

    # ############# Experimental ############# #

    # ## Use fit model to generate synthetic data ## #
    use_dense_model = False
    dense_model_version = ""
    dense_model_name = ""
    dense_training_points = 200

    # ## Clustering for multidimensional models ## #
    clusters = 1
    load_clusters = False
