import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import json

from utils.GMPlotter import eval
import utils.pose_prediction as pose_prediction

def plot_3d_paths(trajectories, title):
    """
    Plots 3D paths of trajectories.

    Parameters:
    - trajectories (list): List of numpy arrays representing trajectories.
    - title (str): Title of the plot.

    Returns:
    None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot each trajectory
    for i, trajectory in enumerate(trajectories):
        m = trajectory.shape[1]
        x = trajectory[1, :]
        y = trajectory[2, :]
        z = trajectory[3, :]
        ax.plot(x, y, z, label=f'T{i + 1}')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(title)
    ax.legend()
    plt.show()

def extract_action_from_library(actionName, library):
    """
    Extracts the robot pose for a given action name from a library.

    Parameters:
    actionName (str): The name of the action to extract the robot pose for.
    library (dict): The library containing the robot poses for different actions.

    Returns:
    dict: A dictionary where the keys are the robot pose names and the values are the corresponding paths.
    """
    robot_pose = {}
    for key in library:
        for item in library[key]:
            if item["id"] == actionName:
                robot_pose[key] = item["path"]
    return robot_pose

def extract_angles_from_library(actionName, library):
    """
    Extracts the joint angles from a library of actions based on the given action name.

    Parameters:
    - actionName (str): The name of the action to extract joint angles for.
    - library (dict): The library of actions containing joint angles.

    Returns:
    - robot_pose (dict): A dictionary mapping joint names to their corresponding joint angles.
    """
    robot_pose = {}
    for key in library:
        for item in library[key]:
            if item["id"] == actionName:
                robot_pose[key] = item["joint_dependency"]
    return robot_pose

def extract_action(df, actionName):
    """
    Extracts rows from a DataFrame that correspond to a specific action.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        actionName (str): The name of the action to extract.

    Returns:
        pandas.DataFrame: The extracted DataFrame containing only rows with the specified action.
    """
    df = df.loc[df['action'] == actionName]
    users = np.arange(1, 21, 1)
    return df

def extract_time(timestamp):
    """
    Extracts the time differences between consecutive timestamps.

    Args:
        timestamp (list): A list of timestamps.

    Returns:
        numpy.ndarray: An array of time differences between consecutive timestamps.
    """
    timestamps_seconds = np.array([float(ts) for ts in timestamp]) / 1e6 # miliseconds, use 1e9 for seconds
    #  Calculate the time difference between consecutive timestamps
    time_diff_seconds = np.diff(timestamps_seconds)
    return np.insert(np.cumsum(time_diff_seconds), 0, 0)

def extract_end_effector_path(vec, time):
    """
    Extracts the end effector path from a given vector of trajectories.

    Args:
        vec (list): A list of trajectories, where each trajectory is a list of vectors.
        time (float): The time value associated with the end effector path.

    Returns:
        numpy.ndarray: The end effector path, where each row represents a vector and the first column represents the time value.
    """
    last_vectors = np.array([trajectory[-1] for trajectory in vec])
    return np.insert(last_vectors, 0, time, axis=1).T

def add_time_to_angles(vec, time):
    """
    Add time to the angles vector.

    Args:
        vec (list or numpy.ndarray): The angles vector.
        time (float or int): The time value to be added.

    Returns:
        numpy.ndarray: The modified angles vector with time added.

    """
    vec = np.array(vec)
    return np.insert(vec, 0, time, axis=1).T

def linear_angular_mapping_gen3(vec):
    """
    Maps the given vector of angles to the range [0, 2*pi].

    Args:
        vec (numpy.ndarray): The input vector of angles.

    Returns:
        numpy.ndarray: The mapped vector of angles in the range [0, 2*pi].
    """
    new_angle_vec = np.zeros_like(vec)
    for i in range(len(vec)):
        if vec[i] >= 0:
            new_angle_vec[i] = vec[i]
        else:
            new_angle_vec[i] = 2 * np.pi + vec[i]
    return new_angle_vec

def extract_vectors(data, robotName):
    """
    Extracts angle vectors from the given data.

    Parameters:
    - data: A list of dictionaries representing data points.
    - robotName: A string representing the name of the robot.

    Returns:
    - angle_vectors: A list of angle vectors extracted from the data.
    """
    angles = data[0].keys()
    # Create vectors for each angle
    angle_vectors = []
    for point in data:
        values = [point[i] for i in angles]
        if robotName == "gen3":
            new_values = linear_angular_mapping_gen3(np.deg2rad(values))
            angle_vectors.append(new_values)
        else:
            angle_vectors.append(np.deg2rad(values))
    return angle_vectors

def extract_axis(vec, axis):
    """
    Extracts the values of a specific axis from a given vector.

    Parameters:
    vec (numpy.ndarray): The input vector containing time and axis values.
    axis (str): The axis to extract values from. Can be "x", "y", or "z".

    Returns:
    numpy.ndarray: A new vector containing time and the values of the specified axis.
    """
    time = vec[0, :]
    if axis == "x":
        axis_values = vec[1, :]
    elif axis == "y":
        axis_values = vec[2, :]
    elif axis == "z":
        axis_values = vec[3, :]
    return np.vstack((time, axis_values))

def extract_angle(vec, angle):
    """
    Extracts the specified angle from a given vector.

    Args:
        vec (numpy.ndarray): The input vector.
        angle (int): The angle to extract from the vector.

    Returns:
        numpy.ndarray: The extracted angle from the vector.
    """
    return vec[[0, angle], :]

def extract_angle_vec(mat, angle):
    """
    Extracts the specified angle from each sublist in the given matrix.

    Args:
        mat (list): The matrix containing sublists.
        angle (float): The angle to extract from each sublist.

    Returns:
        list: A list of angles extracted from each sublist.
    """
    return [extract_angle(sub_list, angle) for sub_list in mat]

def plot_angles_vs_time(arrays):
    """
    Plot angles vs time for multiple arrays.

    Args:
        arrays (list): List of arrays containing time and variable values.

    Returns:
        None
    """
    for i in range(1, arrays[0].shape[1]):
        plt.figure()
        plt.title(f"Variable {i} vs Time")
        for j in range(len(arrays)):
            time_values = arrays[j][:, 0]
            variable_values = arrays[j][:, i]
            plt.plot(time_values, variable_values, label=f"Array {j+1}")

        plt.xlabel("Time")
        plt.ylabel(f"Variable {i}")
        plt.legend()
        plt.show()

def plot_axis_vs_time(vec, axis, max_length=1000, target_length=100, window_length=5, polyorder=3):
    """
    Plots the smoothed variable with respect to time for multiple arrays and calculates the average signal.

    Args:
        vec (list): List of arrays containing time and variable values.
        axis (str): The variable to plot.
        max_length (int, optional): The maximum length of the time axis. Defaults to 1000.
        target_length (int, optional): The desired length of the interpolated signal. Defaults to 100.
        window_length (int, optional): The length of the window used for smoothing. Defaults to 5.
        polyorder (int, optional): The order of the polynomial used for smoothing. Defaults to 3.

    Returns:
        tuple: A tuple containing the time values, data values, and average signal.
    """
    variable_to_plot = axis
    data_as_array = []
    time_as_array = []
    # Extract time and the chosen variable from each array
    time_values = [array[0, :] for array in vec]
    variable_values = [array[1, :] for array in vec]
    # Interpolate each instance to the specified target length
    interp_variable_values = [
        np.interp(np.linspace(0, max_length, target_length), np.linspace(0, max_length, len(time)), variable)
        for time, variable in zip(time_values, variable_values)
    ]

    # Smooth the interpolated signals using the Savitzky-Golay filter
    smoothed_variable_values = [
        savgol_filter(interp_variable, window_length=window_length, polyorder=polyorder)
        for interp_variable in interp_variable_values
    ]

    # Plot the smoothed variable with respect to time for all arrays
    for i, smoothed_variable in enumerate(smoothed_variable_values):
        plt.plot(np.linspace(0, max_length, target_length), smoothed_variable, label=f'{i + 1}')
        data_as_array.extend(smoothed_variable)
        time_as_array.extend(np.linspace(0, max_length, target_length))

    # Calculate and plot the average signal
    avg_signal = np.mean(smoothed_variable_values, axis=0)
    plt.plot(np.linspace(0, max_length, target_length), avg_signal, label='Average', linestyle='--', linewidth=2)

    # Add labels and legend
    plt.xlabel('Normalized Time')
    plt.ylabel(f'{variable_to_plot.capitalize()} Values (Smoothed)')
    plt.legend()
    plt.show()
    return np.array(time_as_array), np.array(data_as_array), avg_signal

def read_library_from_file(name, robotName, babbled_points):
    """
    Read a library of paths from a file.

    Args:
        name (str): The name of the file.
        robotName (str): The name of the robot.
        babbled_points (int): The number of babbled points.

    Returns:
        list: A list of dictionaries representing the paths read from the file.

    Raises:
        FileNotFoundError: If the specified file is not found.

    """
    try:
        with open("data/test_"+ robotName + "/paths_lib/" + name + "_" + str(babbled_points) + ".json", "r") as jsonfile:
            data = [json.loads(line) for line in jsonfile.readlines()]
            return data
    except FileNotFoundError:
        print(f"File {name}_data.json not found.")
        return []

def reshape_array(arr2):
    """
    Reshapes a 2D array into a 3D array by splitting each row into smaller chunks.

    Parameters:
    arr2 (ndarray): The input 2D array to be reshaped.

    Returns:
    ndarray: The reshaped 3D array.
    """
    array_shape = arr2.shape
    for i in range(array_shape[0]):
        v = arr2[i].reshape(int(array_shape[1]/100), 100)
        if i == 0:
            arr = v
        else:
            arr = np.dstack([arr, v])
    return arr

def plot_time_vs_signal(time, signal, xlabel='Time', ylabel='Signal', title='Time vs Signal'):
    """
    Plot the relationship between time and signal.

    Parameters:
    time (array-like): The time values.
    signal (array-like): The signal values.
    xlabel (str, optional): The label for the x-axis. Defaults to 'Time'.
    ylabel (str, optional): The label for the y-axis. Defaults to 'Signal'.
    title (str, optional): The title of the plot. Defaults to 'Time vs Signal'.
    """
    # Plot the time vs signal
    plt.scatter(time, signal)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Show the plot
    plt.show()

def reshape_array(time, arr, max_length=1000):
    """
    Reshapes a given array and time vector into a 2D array with a maximum length.

    Parameters:
    time (numpy.ndarray): The time vector.
    arr (numpy.ndarray): The array to be reshaped.
    max_length (int, optional): The maximum length of each row in the reshaped array. Default is 1000.

    Returns:
    numpy.ndarray: The reshaped array.

    """
    arr_len = len(arr)
    time = time.reshape(int(arr_len/max_length), max_length)[0]
    arr = arr.reshape(int(arr_len/max_length), max_length)
    arr = np.vstack((time, arr))
    return arr.T #[:, 0:2]

def gmm_for_limb(angles_with_time, robotName, action, limb, babbled_points, num_components=5, num_clusters=3):
    """
    Fits a Gaussian Mixture Model (GMM) to the given angles with time data for a specific limb of a robot.

    Parameters:
    - angles_with_time (numpy.ndarray): Array of angles with time data.
    - robotName (str): Name of the robot.
    - action (str): Name of the action.
    - limb (str): Name of the limb.
    - babbled_points (int): Number of babbled points.
    - num_components (int, optional): Number of components in the GMM. Default is 5.
    - num_clusters (int, optional): Number of components in the GMR. Default is 3.
    Returns:
    None
    """
    for i in range(1, angles_with_time[0].shape[0]):
        variable_to_fit = extract_angle_vec(angles_with_time, i)
        time, smoothed_angles, avg_signal = plot_axis_vs_time(variable_to_fit, str(i)) #7093, 7093
        eval(time, smoothed_angles, avg_signal, str(babbled_points), robotName, action, limb + str(i), num_components, num_clusters)

if __name__ == "__main__":
    """
    Usage example
    """

    robotName = 'gen3'
    users = np.arange(1, 21, 1)
    num_components = 7
    action = 'shallow_plate'
    babbled_points = 150
    lib_dict = {}
    t_x_left = []
    t_y_left = []
    t_z_left = []
    left_side_with_time = []
    right_side_with_time = []
    head_with_time = []
    end_effector_dict = ["jointRight_9", "jointRight_9"]
    file_path = "./robot_configuration_files/"+ robotName + ".yaml"
    pose_predictor = pose_prediction.Prediction(file_path, robotName)

    # For old actions
    df = pose_predictor.read_file("combined_actions")

    # For new actions
    # df = pose_predictor.read_file("/QT_recordings/human/arms_sides_2")
    dict_pose_vec_left = []
    dict_pose_vec_right = []
    dict_pose_vec_head = []
    angles_left_with_time = []
    angles_right_with_time = []
    angles_head_with_time = []
    for key in end_effector_dict:
        lib_dict[key] = read_library_from_file(key, robotName, babbled_points)
    for user in users:
        # For old actions
        _, _, _, timestamps = pose_predictor.read_csv_combined(df, action, user)
        # For new actions
        # _, _, _, timestamps = pose_predictor.read_recorded_action_csv(df, action, user)
        cumulative_time = extract_time(timestamps)
        dict_pose = extract_action_from_library(str(user) + action, lib_dict)
        cartesian_left_vec = dict_pose[end_effector_dict[0]]
        cartesian_right_vec = dict_pose[end_effector_dict[1]]
        if robotName != "gen3":
            cartesian_head_vec = dict_pose[end_effector_dict[2]]
        dep_dict = extract_angles_from_library(str(user) + action, lib_dict)
        jointLeft_vectors = extract_vectors(dep_dict[end_effector_dict[0]], robotName)
        jointRight_vectors = extract_vectors(dep_dict[end_effector_dict[1]], robotName)
        if robotName != "gen3":
            jointHead_vectors = extract_vectors(dep_dict[end_effector_dict[2]])
        if len(cumulative_time) > 1:
            aux_left = add_time_to_angles(cartesian_left_vec, cumulative_time)
            aux_right = add_time_to_angles(cartesian_right_vec, cumulative_time)
            if robotName != "gen3":
                aux_head = add_time_to_angles(cartesian_head_vec, cumulative_time)
            aux_left_angles = add_time_to_angles(jointLeft_vectors, cumulative_time)
            aux_right_angles = add_time_to_angles(jointRight_vectors, cumulative_time)
            if robotName != "gen3":
                aux_head_angles = add_time_to_angles(jointHead_vectors, cumulative_time)
            angles_left_with_time.append(aux_left_angles)
            angles_right_with_time.append(aux_right_angles)
            if robotName != "gen3":
                angles_head_with_time.append(aux_head_angles)
            t_x_left.append(extract_axis(aux_left, "x"))
            t_y_left.append(extract_axis(aux_left, "y"))
            t_z_left.append(extract_axis(aux_left, "z"))
            left_side_with_time.append(aux_left)
            right_side_with_time.append(aux_right)
            if robotName != "gen3":
                head_with_time.append(aux_head)
        dict_pose_vec_left.append(jointLeft_vectors)
        dict_pose_vec_right.append(jointRight_vectors)
        if robotName != "gen3":
            dict_pose_vec_head.append(jointHead_vectors)
    # print("angles_left_with_time: ", angles_left_with_time)
    # plot_angles_vs_time(angles_left_with_time)
    # plot_angles_vs_time(angles_right_with_time)
    # plot_angles_vs_time(angles_head_with_time)
    # plot_3d_paths(left_side_with_time, "Left EE")
    # plot_3d_paths(right_side_with_time, "Right EE")
    # plot_3d_paths(head_with_time, "Head EE")
    gmm_for_limb(angles_left_with_time, robotName, action + "", "left_", babbled_points, num_components)
    gmm_for_limb(angles_right_with_time, robotName, action + "", "right_", babbled_points, num_components)
    if robotName != "gen3":
        gmm_for_limb(angles_head_with_time, robotName, action + "", "head_", babbled_points, num_components)