import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from matplotlib.widgets import Slider, Button
from scipy.spatial import KDTree
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

import utils.robot_graph as robot_graph
import torch.optim.lr_scheduler as lr_scheduler
import utils.robot as robot

plt.rcParams.update({'font.size': 10})

#HUMAN PARAMETERS
number_arm_human_joints = 5
number_head_human_joints = 3
human_joints_head  = ['JOINT_LEFT_COLLAR', 'JOINT_NECK', 'JOINT_HEAD']
human_joints_left  = ['JOINT_LEFT_COLLAR', 'JOINT_LEFT_SHOULDER', 'JOINT_LEFT_ELBOW', 'JOINT_LEFT_WRIST', 'JOINT_LEFT_HAND']
human_joints_right = ['JOINT_RIGHT_COLLAR', 'JOINT_RIGHT_SHOULDER', 'JOINT_RIGHT_ELBOW', 'JOINT_RIGHT_WRIST', 'JOINT_RIGHT_HAND']

def createRobotGraphs(robot):
    """
    Create robot graphs based on the given robot object.

    Args:
        robot: The robot object containing information about the joint angles.

    Returns:
        joint_dic: A dictionary containing the robot graphs, where the keys are the joint names
                   and the values are instances of the Graph class.
    """
    joint_dic = {}
    for i in range(len(robot.leftArmAngles)):
        joint_dic["jointLeft_" + str(i)] = robot_graph.Graph(i, "jointLeft_" + str(i))

    for i in range(len(robot.rightArmAngles)):
        joint_dic["jointRight_" + str(i)] = robot_graph.Graph(i, "jointRight_" + str(i))

    for i in range(len(robot.headAngles)):
        joint_dic["jointHead_" + str(i)] = robot_graph.Graph(i, "jointHead_" + str(i))
    return joint_dic

def extract_vectors(data):
    """
    Extracts angle vectors from the given data.

    Args:
        data (list): A list of dictionaries containing angle values.

    Returns:
        list: A list of angle vectors in radians.
    """
    angles = data[0].keys()
    angle_vectors = []
    for point in data:
        values = [point[i] for i in angles]
        angle_vectors.append(np.deg2rad(values))
    return angle_vectors

def find_closest_point(new_point, kdtree):
    """
    Finds the closest point to the given new_point in the provided kdtree.

    Args:
        new_point (array-like): The coordinates of the new point.
        kdtree (scipy.spatial.cKDTree): The kdtree containing the points.

    Returns:
        array-like: The coordinates of the closest point in the kdtree.
    """
    distance, index = kdtree.query(new_point)
    return kdtree.data[index]

def path_planning(demonstration, graph):
    """
    Plans a path based on a given demonstration and a graph.

    Args:
        demonstration (iterable): The demonstration data.
        graph (Graph): The graph representing the environment.

    Returns:
        list: The planned path as a list of tuples.

    """
    kdtree = KDTree(graph.get_nodes_values())
    path = []
    demonstration = list(demonstration)
    for i in demonstration:
        node = find_closest_point(i, kdtree)
        path.append(tuple(node))
    return path

def plotPath(key, demonstration, predicted_path, path_from_library):
    """
    Plots the path of a robot in a 3D space.

    Parameters:
    - key (str): The key or name of the path.
    - demonstration (ndarray): The coordinates of the points in the user demonstration.
    - predicted_path (ndarray): The coordinates of the points in the predicted path.
    - path_from_library (ndarray): The coordinates of the points in the path from the library.

    Returns:
    None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xdem = demonstration[:, 0]
    ydem = demonstration[:, 1]
    zdem = demonstration[:, 2]
    xpath = predicted_path[:, 0]
    ypath = predicted_path[:, 1]
    zpath = predicted_path[:, 2]

    # Scatter plots for the points
    ax.scatter3D(xdem[1:-1], ydem[1:-1], zdem[1:-1], c='blue', marker='o', label='User demonstration')
    ax.scatter3D(xdem[0], ydem[0], zdem[0], c='green', marker='*')
    ax.scatter3D(xdem[-1], ydem[-1], zdem[-1], c='red', marker='*')

    ax.scatter3D(xpath[1:-1], ypath[1:-1], zpath[1:-1], c='pink', marker='o', label='Path planning')
    ax.scatter3D(xpath[0], ypath[0], zpath[0], c='green', marker='*', label='Starting point')
    ax.scatter3D(xpath[-1], ypath[-1], zpath[-1], c='red', marker='*', label='Ending point')

    if len(path_from_library):
        xdata = path_from_library[:, 0]
        ydata = path_from_library[:, 1]
        zdata = path_from_library[:, 2]
        ax.scatter3D(xdata, ydata, zdata, c='purple', label='Path from library')

    # Plot edges between the points
    for i in range(len(xdem) - 1):
        ax.plot([xdem[i], xdem[i+1]], [ydem[i], ydem[i+1]], [zdem[i], zdem[i+1]], c='grey')
    for i in range(len(xpath) - 1):
        ax.plot([xpath[i], xpath[i+1]], [ypath[i], ypath[i+1]], [zpath[i], zpath[i+1]], c='lightblue')
    for i in range(len(xdata) - 1):
        ax.plot([xdata[i], xdata[i+1]], [ydata[i], ydata[i+1]], [zdata[i], zdata[i+1]], c='green')

    ax.set_title(key)
    ax.set_xlabel("\n X [mm]", linespacing=3.2)
    ax.set_ylabel("\n Y [mm]", linespacing=3.2)
    ax.set_zlabel("\n Z [mm]", linespacing=3.2)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend()
    fig.tight_layout()
    # plt.savefig(key + "_world_path.pdf", format="pdf")
    plt.show()

def extract_action_from_library(actionName, library):
    """
    Extracts the robot pose for a given action name from the library.

    Args:
        actionName (str): The name of the action.
        library (dict): The library containing the robot poses.

    Returns:
        dict: A dictionary containing the robot pose for each key in the library.
    """
    robot_pose = {}
    for key in library:
        for item in library[key]:
            if item["id"] == actionName:
                robot_pose[key] = item["path"]
    return robot_pose

def extract_angles_from_library(actionName, library):
    """
    Extracts the angles from a library of robot poses based on the given action name.

    Parameters:
    actionName (str): The name of the action to extract angles for.
    library (dict): The library of robot poses.

    Returns:
    dict: A dictionary containing the extracted angles for each joint dependency.
    """
    robot_pose = {}
    for key in library:
        for item in library[key]:
            if item["id"] == actionName:
                robot_pose[key] = item["joint_dependency"]
    return robot_pose

def read_library_from_file(name, robotName):
    """
    Read library data from a file.

    Args:
        name (str): The name of the library.
        robotName (str): The name of the robot.

    Returns:
        list: The library data read from the file.

    Raises:
        FileNotFoundError: If the file is not found.

    """
    try:
        with open(name + "_" + robotName + "_data.json", "r") as jsonfile:
            data = [json.loads(line) for line in jsonfile.readlines()]
            return data
    except FileNotFoundError:
        print(f"File {name}_data.json not found.")
        return []

def find_end_effectors_keys(dict):
    """
    Finds the keys corresponding to the maximum values of the 'jointLeft', 'jointRight', and 'jointHead' prefixes in a dictionary.

    Args:
        dict (dict): The dictionary containing the keys with the 'jointLeft', 'jointRight', and 'jointHead' prefixes.

    Returns:
        list: A list containing the keys with the maximum values for 'jointLeft', 'jointRight', and 'jointHead'.
    """
    left_numbers = [int(key.split('_')[1]) for key in dict if key.startswith('jointLeft')]
    right_numbers = [int(key.split('_')[1]) for key in dict if key.startswith('jointRight')]
    head_numbers = [int(key.split('_')[1]) for key in dict if key.startswith('jointHead')]

    max_left = 'jointLeft_' + str(max(left_numbers, default=0))
    max_right = 'jointRight_' + str(max(right_numbers, default=0))
    max_head = 'jointHead_' + str(max(head_numbers, default=0))

    return [max_left, max_right, max_head]

class Prediction(object):
    """
    Class for performing pose prediction.

    This class provides methods for converting Cartesian coordinates to spherical coordinates,
    extracting spherical angles from a given human pose, calculating the forward kinematics of a robot
    based on joint angles, and reading and processing pose data from CSV files.

    Attributes:
        robot (Robot): An instance of the Robot class.
        base (float): The base distance of the robot.
        robotName (str): The name of the robot.
    """

    def __init__(self, file_path, name):
        """
        Initialize the PosePrediction class.

        Args:
            file_path (str): The file path to import the robot from.
            name (str): The name of the robot.

        Attributes:
            robot (Robot): An instance of the Robot class.
            base (float): The base distance of the robot.
            robotName (str): The name of the robot.
        """
        self.robot = robot.Robot(name)
        self.robot.import_robot(file_path)
        self.base = self.robot.baseDistance
        self.robotName = name

    def cartesian_to_spherical(self, cartesian_points):
        """
        Convert Cartesian coordinates to spherical coordinates.

        Parameters:
        - cartesian_points (numpy.ndarray): Array of Cartesian coordinates with shape (N, 3).

        Returns:
        - theta (numpy.ndarray): Array of inclination angles with shape (N,).
        - phi (numpy.ndarray): Array of azimuth angles with shape (N,).
        """
        r = np.linalg.norm(cartesian_points, axis=1)                    # Radial distance
        theta = np.arccos(cartesian_points[:, 2] / r)                   # Inclination angle
        phi = np.arctan2(cartesian_points[:, 1], cartesian_points[:, 0])  # Azimuth angle
        return theta, phi

    def extract_spherical_angles_from_human(self, left, right, head=[]):
        """
        Extracts spherical angles from the given human pose.

        Parameters:
        - left: numpy array representing the left pose keypoints
        - right: numpy array representing the right pose keypoints
        - head: numpy array representing the head pose keypoints (optional)

        Returns:
        - left_theta: numpy array of theta angles for the left pose keypoints
        - left_phi: numpy array of phi angles for the left pose keypoints
        - right_theta: numpy array of theta angles for the right pose keypoints
        - right_phi: numpy array of phi angles for the right pose keypoints
        - head_theta: numpy array of theta angles for the head pose keypoints
        - head_phi: numpy array of phi angles for the head pose keypoints
        """
        left_theta, left_phi = self.cartesian_to_spherical(np.diff(left, axis=0))
        right_theta, right_phi = self.cartesian_to_spherical(np.diff(right, axis=0))
        head_theta, head_phi = self.cartesian_to_spherical(np.diff(head, axis=0))
        return left_theta, left_phi, right_theta, right_phi, head_theta, head_phi

    def robot_embodiment(self, angles_left, angles_right, angles_head):
        """
        Calculates the forward kinematics of the robot based on the given joint angles.

        Args:
            angles_left (numpy.ndarray): Array of joint angles for the left side of the robot.
            angles_right (numpy.ndarray): Array of joint angles for the right side of the robot.
            angles_head (numpy.ndarray): Array of joint angles for the head of the robot.

        Returns:
            Tuple[List[List[float]], List[List[float]], List[List[float]]]: A tuple containing the forward kinematics
            matrices for the left side, right side, and head of the robot, respectively.
        """
        angles = np.concatenate((angles_left, angles_right, angles_head), axis=None)
        if self.robotName == "qt":
            l, r, h = self.robot.forward_kinematics_qt(angles)
        elif self.robotName == "nao":
            l, r, h = self.robot.forward_kinematics_nao(angles)
        elif self.robotName == "gen3":
            l, r, h = self.robot.forward_kinematics_gen3(angles)
        return self.matrix_array_to_list_list(l), self.matrix_array_to_list_list(r), self.matrix_array_to_list_list(h)

    def matrix_array_to_list_list(self, vec):
        """
        Converts a matrix array to a list of lists.

        Args:
            vec (numpy.ndarray): The matrix array to be converted.

        Returns:
            numpy.ndarray: The converted list of lists.
        """
        new_list = [list(i) for i in vec]
        return np.around(np.array(new_list), decimals=2)

    def vectorise_string(self, vec):
        """
        Converts a string representation of a vector into a list of floats.

        Args:
            vec (str): The string representation of the vector.

        Returns:
            list: A list of floats representing the vector.

        Example:
            >>> vectorise_string('[1.0, 2.0, 3.0]')
            [1.0, 2.0, 3.0]
        """
        aux_data = ''.join([i for i in vec if not (i=='[' or i==']' or i==',')])
        data = np.array(aux_data.split())
        return list(data[0:3].astype(float))

    def vectorise_spherical_data(self, vec):
        """
        Converts a spherical vector into a list of float values.

        Args:
            vec (str): The spherical vector to be converted.

        Returns:
            list: A list of float values representing the spherical vector.

        Example:
            vec = "[1.0, 2.0, 3.0]"
            result = vectorise_spherical_data(vec)
            # Output: [1.0, 2.0, 3.0]
        """
        aux_data = ''.join([i for i in vec if not (i=='[' or i==']' or i==',')])
        data = np.array(aux_data.split())
        #+1 for QT, human has 4 angles and QT 3
        return list(data[0:self.robot.dof_arm + 1].astype(float))

    def read_csv_combined(self, df, action, user):
        """
        Read and process a CSV file containing pose data for a specific action and user.

        Args:
            df (pandas.DataFrame): The DataFrame containing the pose data.
            action (str): The action to filter the data by.
            user (str): The user to filter the data by.

        Returns:
            tuple: A tuple containing four numpy arrays:
                - left_side: An array of pose data for the left side of the body.
                - right_side: An array of pose data for the right side of the body.
                - head: An array of pose data for the head.
                - time: An array of timestamps corresponding to each pose data entry.
        """
        left_side = []
        right_side = []
        head = []
        time = []
        aux_arm = [[] for _ in range(number_arm_human_joints)]
        aux_head = [[] for _ in range(number_head_human_joints)]
        try:
            filtered_df = df[(df['action'] == action) & (df['participant_id'] == user)]

            for i in range(len(filtered_df)):
                time.append(filtered_df['timestamp'].iloc[i])
                for count, left in enumerate(human_joints_left):
                    aux_arm[count] = self.vectorise_string(filtered_df[left].iloc[i])
                left_side.append(aux_arm.copy())

                for count, right in enumerate(human_joints_right):
                    aux_arm[count] = self.vectorise_string(filtered_df[right].iloc[i])
                right_side.append(aux_arm.copy())

                for count, _head in enumerate(human_joints_head):
                    aux_head[count] = self.vectorise_string(filtered_df[_head].iloc[i])
                head.append(aux_head.copy())

        except KeyError as e:
            print(f"Error: {e}")
            # Handle the exception here if needed, e.g., return default values

        return np.array(left_side), np.array(right_side), np.array(head), np.array(time)

    def read_recorded_action_csv(self, df, action, user):
        """
        Read recorded action data from a CSV file.

        Args:
            df (pandas.DataFrame): The DataFrame containing the action data.
            action (str): The action being performed.
            user (str): The user who performed the action.

        Returns:
            tuple: A tuple containing numpy arrays of the left side, right side, head, and time data.

        Raises:
            KeyError: If the required columns are not found in the DataFrame.

        """
        left_side = []
        right_side = []
        head = []
        time = []
        np_array = {'np': np, 'array': np.array}
        try:
            for i in range(len(df)):
                time.append(df['timestamp'].iloc[i])

                aux_arm = eval(df['left'].iloc[i], np_array)
                result_list = [value.tolist() for value in aux_arm.values()]
                left_side.append(result_list)

                aux_arm = eval(df['right'].iloc[i], np_array)
                result_list = [value.tolist() for value in aux_arm.values()]
                right_side.append(result_list)

                aux_head = eval(df['head'].iloc[i], np_array)
                result_list = [aux_head[key].tolist() for key in aux_head.keys() if key in human_joints_head]
                result_list.reverse()
                head.append(result_list)

        except KeyError as e:
            print(f"Error: {e}")
            # Handle the exception here if needed, e.g., return default values
        return np.array(left_side), np.array(right_side), np.array(head), np.array(time)

    def read_training_data(self, file_name):
        """
        Read training data from a file and preprocess it.

        Args:
            file_name (str): The name of the file containing the training data.

        Returns:
            tuple: A tuple containing the preprocessed training data arrays. The order of the arrays is as follows:
                - theta_left (ndarray): The vectorized spherical data for the left arm human theta angles.
                - phi_left (ndarray): The vectorized spherical data for the left arm human phi angles.
                - theta_right (ndarray): The vectorized spherical data for the right arm human theta angles.
                - phi_right (ndarray): The vectorized spherical data for the right arm human phi angles.
                - theta_head (ndarray): The vectorized spherical data for the head human theta angles.
                - phi_head (ndarray): The vectorized spherical data for the head human phi angles.
                - left_arm_robot (ndarray): The vectorized spherical data for the left arm robot angles.
                - right_arm_robot (ndarray): The vectorized spherical data for the right arm robot angles.
                - head_robot (ndarray): The vectorized spherical data for the head robot angles.
        """
        df = self.read_file(file_name)
        theta_left = np.array([self.vectorise_spherical_data(i) for i in df['left_arm_human_theta']])
        theta_right = np.array([self.vectorise_spherical_data(i) for i in df['right_arm_human_theta']])
        theta_head = np.array([self.vectorise_spherical_data(i) for i in df['head_human_theta']])
        phi_left = np.array([self.vectorise_spherical_data(i) for i in df['left_arm_human_phi']])
        phi_right = np.array([self.vectorise_spherical_data(i) for i in df['right_arm_human_phi']])
        phi_head = np.array([self.vectorise_spherical_data(i) for i in df['head_human_phi']])
        left_arm_robot = np.array([self.vectorise_spherical_data(i) for i in df['left_arm_robot']])
        right_arm_robot = np.array([self.vectorise_spherical_data(i) for i in df['right_arm_robot']])
        head_robot = np.array([self.vectorise_spherical_data(i) for i in df['head_robot']])
        return theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot

    def save_mapping_csv(self, file_name, action, user, lht, lhp, hht, rht, rhp, hhp, ra):
        """
        Saves the mapping data to a CSV file.

        Args:
            file_name (str): The name of the CSV file.
            action (str): The action associated with the mapping data.
            user (str): The participant ID associated with the mapping data.
            lht (float): The left arm human theta value.
            lhp (float): The left arm human phi value.
            hht (float): The head human theta value.
            rht (float): The right arm human theta value.
            rhp (float): The right arm human phi value.
            hhp (float): The head human phi value.
            ra (list): The robot arm values.

        Returns:
            None
        """
        df = pd.read_csv(file_name + ".csv")
        for i in range(len(df)):
            if action == df['action'][i] and user == df['participant_id'][i]:
                df['left_arm_human_theta'][i] = str(lht)
                df['right_arm_human_theta'][i] = str(rht)
                df['head_human_theta'][i] = str(hht)
                df['left_arm_human_phi'][i] = str(lhp)
                df['right_arm_human_phi'][i] = str(rhp)
                df['head_human_phi'][i] = str(hhp)
                df['left_arm_robot'][i] = str(ra[0:self.robot.dof_arm])
                df['right_arm_robot'][i] = str(ra[self.robot.dof_arm:2*self.robot.dof_arm])
                if "gen3" in file_name:
                    df['head_robot'][i] = str([0, 0])
                else:
                    df['head_robot'][i] = str(ra[2*self.robot.dof_arm:2*self.robot.dof_arm + 2])
        df.to_csv(file_name + ".csv")

    def create_3d_plot(self):
        """
        Create a 3D plot with two subplots.

        Returns:
            fig (matplotlib.figure.Figure): The figure object.
            ax1 (matplotlib.axes._subplots.Axes3DSubplot): The first subplot.
            ax2 (matplotlib.axes._subplots.Axes3DSubplot): The second subplot.
        """
        fig = plt.figure(figsize=(12, 5))
        # ax1 = fig.add_subplot(121, projection='3d')
        # ax2 = fig.add_subplot(122, projection='3d')

        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], height_ratios=[1])

        ax1 = fig.add_subplot(gs[0], projection='3d')
        ax2 = fig.add_subplot(gs[1], projection='3d')

        # Set the axis labels and limits for both subplots (adjust as needed)
        ax1.set_xlabel("\n X [mm]", linespacing=3.2)
        ax1.set_ylabel("\n Y [mm]", linespacing=3.2)
        ax1.set_zlabel("\n Z [mm]", linespacing=3.2)
        #QTROBOT
        # ax1.set_xlim([-500, 500])
        # ax1.set_ylim([-500, 500])
        # ax1.set_zlim([0, 800])
        #HUMAN
        ax1.set_xlim([1500, 3000])
        ax1.set_ylim([-900, 900])
        ax1.set_zlim([-400, 800])

        ax2.set_xlabel("\n X [mm]", linespacing=3.2)
        ax2.set_ylabel("\n Y [mm]", linespacing=3.2)
        ax2.set_zlabel("\n Z [mm]", linespacing=3.2)
        if self.robotName == "gen3":
            ax2.set_xlim([-1000, 1000])
            ax2.set_ylim([-1000, 1000])
            ax2.set_zlim([0, 1500])
        elif self.robotName == "qt":
            ax2.set_xlim([-500, 500])
            ax2.set_ylim([-500, 500])
            ax2.set_zlim([0, 800])
        elif self.robotName == "nao":
            ax2.set_xlim([-300, 300])
            ax2.set_ylim([-400, 400])
            ax2.set_zlim([-200, 300])

        return fig, ax1, ax2

    def plot_animation_3d(self, point_clouds_list):
        """
        Plot a 3D animation of point clouds.

        Args:
            point_clouds_list (list): A list of tuples, each containing 6 point clouds.

        Raises:
            ValueError: If the input data is not a list of tuples, each containing 6 point clouds.

        Returns:
            None
        """
        if not isinstance(point_clouds_list, list) or not all(isinstance(pc, tuple) and len(pc) == 6 for pc in point_clouds_list):
            raise ValueError("Invalid input data. Expecting a list of tuples, each containing 5 point clouds")

        fig, ax1, ax2 = self.create_3d_plot()

        # Line plots for connections in both subplots
        lines1 = [ax1.plot([], [], [], c='b', linewidth=2)[0] for _ in range(len(point_clouds_list[0][0]) - 1)]
        lines2 = [ax1.plot([], [], [], c='g', linewidth=2, linestyle='--')[0] for _ in range(len(point_clouds_list[0][1]) - 1)]
        lines3 = [ax1.plot([], [], [], c='r', linewidth=2)[0] for _ in range(len(point_clouds_list[0][2]) - 1)]
        lines4 = [ax2.plot([], [], [], c='b', linewidth=2)[0] for _ in range(len(point_clouds_list[0][3]) - 1)]
        lines5 = [ax2.plot([], [], [], c='g', linewidth=2, linestyle='--')[0] for _ in range(len(point_clouds_list[0][4]) - 1)]
        lines6 = [ax2.plot([], [], [], c='r', linewidth=2)[0] for _ in range(len(point_clouds_list[0][5]) - 1)]

        def update(frame):
            points1, points2, points3, points4, points5, points6 = point_clouds_list[frame]

            # Set the titles for each subplot
            ax1.set_title('Human', fontsize=20)
            ax2.set_title('Robot', fontsize=20)

            # Update line plots in both subplots
            for i in range(len(points1) - 1):
                lines1[i].set_data([points1[i, 0], points1[i + 1, 0]], [points1[i, 1], points1[i + 1, 1]])
                lines1[i].set_3d_properties([points1[i, 2], points1[i + 1, 2]])

            for i in range(len(points2) - 1):
                lines2[i].set_data([points2[i, 0], points2[i + 1, 0]], [points2[i, 1], points2[i + 1, 1]])
                lines2[i].set_3d_properties([points2[i, 2], points2[i + 1, 2]])

            for i in range(len(points3) - 1):
                lines3[i].set_data([points3[i, 0], points3[i + 1, 0]], [points3[i, 1], points3[i + 1, 1]])
                lines3[i].set_3d_properties([points3[i, 2], points3[i + 1, 2]])

            for i in range(len(points4) - 1):
                lines4[i].set_data([points4[i, 0], points4[i + 1, 0]], [points4[i, 1], points4[i + 1, 1]])
                lines4[i].set_3d_properties([points4[i, 2], points4[i + 1, 2]])

            for i in range(len(points5) - 1):
                lines5[i].set_data([points5[i, 0], points5[i + 1, 0]], [points5[i, 1], points5[i + 1, 1]])
                lines5[i].set_3d_properties([points5[i, 2], points5[i + 1, 2]])

            for i in range(len(points6) - 1):
                lines6[i].set_data([points6[i, 0], points6[i + 1, 0]], [points6[i, 1], points6[i + 1, 1]])
                lines6[i].set_3d_properties([points6[i, 2], points6[i + 1, 2]])

            return *lines1, *lines2, *lines3, *lines4, *lines5, *lines6

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(point_clouds_list), interval=200, blit=True)

        # Add a slider to control the animation
        slider_ax = plt.axes([0.1, 0.01, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(slider_ax, 'Frame', 0, len(point_clouds_list) - 1, valinit=0, valstep=1)
        ax2.scatter(self.base[0], self.base[1], self.base[2], marker="o")

        def update_animation(val):
            frame = int(slider.val)
            update(frame)
            fig.canvas.draw_idle()

        slider.on_changed(update_animation)
        # Show the animation
        plt.show()

    def read_file(self, name):
        """
        Reads a CSV file from the './data/' directory.

        Args:
            name (str): The name of the file (without the extension).

        Returns:
            pandas.DataFrame: The contents of the CSV file as a DataFrame.
        """
        return pd.read_csv('./data/' + name + '.csv')

    def train_pytorch(self, robot, theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, num_epochs=500):
        """
        Trains a PyTorch model for pose prediction.

        Args:
            robot (Robot): The robot object.
            theta_left (ndarray): The theta values for the left arm.
            phi_left (ndarray): The phi values for the left arm.
            theta_right (ndarray): The theta values for the right arm.
            phi_right (ndarray): The phi values for the right arm.
            theta_head (ndarray): The theta values for the head.
            phi_head (ndarray): The phi values for the head.
            left_arm_robot (ndarray): The left arm robot data.
            right_arm_robot (ndarray): The right arm robot data.
            head_robot (ndarray): The head robot data.
            num_epochs (int, optional): The number of training epochs. Defaults to 500.

        Returns:
            None
        """
        class MultiOutputModel(nn.Module):
            """
            A class representing a multi-output model for pose prediction.

            This model consists of a shared layer followed by separate layers for predicting
            the pose of the left arm, right arm, and head (in they exist). It takes an input tensor and produces
            output tensors for each of the pose predictions.

            Attributes:
                shared_layer1 (nn.Linear): The shared layer of the model.
                left_arm_layer (nn.Linear): The layer for predicting the pose of the left arm.
                right_arm_layer (nn.Linear): The layer for predicting the pose of the right arm.
                head_layer (nn.Linear): The layer for predicting the pose of the head.
            """

            def __init__(self):
                """
                Initializes a MultiOutputModel object.

                This function sets up the layers of the model for pose prediction.
                It creates a shared layer followed by separate layers for the left arm,
                right arm, and head predictions.

                Args:
                    None

                Returns:
                    None
                """
                super(MultiOutputModel, self).__init__()
                self.shared_layer1 = nn.Linear(20, 128)
                self.left_arm_layer = nn.Linear(128, robot.dof_arm)
                self.right_arm_layer = nn.Linear(128, robot.dof_arm)
                self.head_layer = nn.Linear(128, 2)

            def forward(self, x):
                """
                Forward pass of the pose_prediction model.

                Args:
                    x (torch.Tensor): Input tensor.

                Returns:
                    Tuple[torch.Tensor]: Tuple containing the output tensors for the left arm, right arm, and head.
                """
                x = torch.relu(self.shared_layer1(x))
                left_arm_output = self.left_arm_layer(x)
                right_arm_output = self.right_arm_layer(x)
                head_output = self.head_layer(x)
                return left_arm_output, right_arm_output, head_output

        # Instantiate the model
        model = MultiOutputModel()

        # Define the optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        criterion = nn.MSELoss()

        # Prepare training data as PyTorch tensors
        input_data = torch.Tensor(np.concatenate((theta_left, phi_left, theta_right, phi_right, theta_head, phi_head), axis=1))
        output_data_left = torch.Tensor(left_arm_robot)
        output_data_right = torch.Tensor(right_arm_robot)
        output_data_head = torch.Tensor(head_robot)

        # Training loop
        training_losses = []
        validation_losses = []
        train_input_data, val_input_data, train_output_data_left, val_output_data_left, train_output_data_right, val_output_data_right, train_output_data_head, val_output_data_head = train_test_split(
        input_data, output_data_left, output_data_right, output_data_head, test_size=0.2, random_state=50)
        start = time.perf_counter()
        for epoch in range(num_epochs):
            # Forward pass
            left_arm_pred, right_arm_pred, head_pred = model(train_input_data)

            # Calculate loss for each output branch
            loss_left = criterion(left_arm_pred, train_output_data_left)
            loss_right = criterion(right_arm_pred, train_output_data_right)
            loss_head = criterion(head_pred, train_output_data_head)

            # Total loss as a combination of individual losses
            total_loss = loss_left + loss_right + loss_head

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            training_losses.append(total_loss.item())

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()}')

         # For validation, use separate validation data
            with torch.no_grad():  # Disable gradient computation for validation
                val_left_arm_pred, val_right_arm_pred, val_head_pred = model(val_input_data)

                # Calculate validation loss for each output branch
                val_loss_left = criterion(val_left_arm_pred, val_output_data_left)
                val_loss_right = criterion(val_right_arm_pred, val_output_data_right)
                val_loss_head = criterion(val_head_pred, val_output_data_head)

                # Total validation loss as a combination of individual losses
                total_val_loss = val_loss_left + val_loss_right + val_loss_head
                validation_losses.append(total_val_loss.item())

                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {total_val_loss.item()}')

        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken: {elapsed:.6f} seconds')

        self.model = model
        # Plot the training loss
        plt.plot(range(1, num_epochs + 1), training_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), validation_losses, label='Validation Loss')
        plt.title('Performance of the model with MSE loss & 128 hidden neurons - Freddy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        # plt.savefig('nn_performace/MSELoss16-2000'+ self.robotName +'.pdf', format='pdf')
        # plt.show()

    def dicts_to_lists(self, left, right, head):
        """
        Convert dictionaries of joint positions to lists.

        Args:
            left (dict): Dictionary containing joint positions for the left side of the body.
            right (dict): Dictionary containing joint positions for the right side of the body.
            head (dict): Dictionary containing joint positions for the head.

        Returns:
            left_side (numpy.ndarray): Array containing joint positions for the left side of the body.
            right_side (numpy.ndarray): Array containing joint positions for the right side of the body.
            head_ (numpy.ndarray): Array containing joint positions for the head.
        """
        left_side = []
        right_side = []
        head_ = []
        for key in human_joints_head:
            head_.append(head[key])
        for key in human_joints_left:
            left_side.append(left[key])
        for key in human_joints_right:
            right_side.append(right[key])
        return np.array(left_side), np.array(right_side), np.array(head_)

    def predict_pytorch(self, left_input, right_input, head_input):
        """
        Predicts the pose using a PyTorch model.

        Args:
            left_input (numpy.ndarray): The input data for the left arm.
            right_input (numpy.ndarray): The input data for the right arm.
            head_input (numpy.ndarray): The input data for the head.

        Returns:
            tuple: A tuple containing the predicted poses for the left arm, right arm, and head.
        """
        theta_left, phi_left, theta_right, phi_right, theta_head, phi_head = self.extract_spherical_angles_from_human(left_input, right_input, head_input)
        theta_head = [0, 0]
        phi_head = [0, 0]
        input_data = torch.Tensor(np.concatenate((theta_left, phi_left, theta_right, phi_right, theta_head, phi_head), axis=None))

        with torch.no_grad():
            left_arm_pred, right_arm_pred, head_pred = self.model(input_data)

        # Convert the predictions to numpy arrays
        left_arm_pred = left_arm_pred.numpy()
        right_arm_pred = right_arm_pred.numpy()
        head_pred = head_pred.numpy()
        return left_arm_pred, right_arm_pred, head_pred

    def plot_angle_sequence(self, data_1, data_2 = []):
        """
        Plots the angle sequence for given data.

        Args:
            data_1 (list): List of angle data for the first set of joints.
            data_2 (list, optional): List of angle data for the second set of joints. Defaults to [].

        Returns:
            None
        """
        columns_1 = np.array(data_1).T
        columns_2 = np.array(data_2).T

        # Determine the number of subplots
        num_subplots = max(len(columns_1), len(columns_2))

        # Create subplots
        if len(data_2):
            fig, axs = plt.subplots(num_subplots, 2, figsize=(15, 3 * num_subplots))
        else:
            fig, axs = plt.subplots(num_subplots, 1, figsize=(15, 3 * num_subplots))

        if len(data_2):
            for i, column in enumerate(columns_1):
                axs[i, 0].plot(column, label=f'Joint {i + 0}', color='C{}'.format(i))
                axs[i, 0].legend()
                axs[i, 0].grid()

            for i, column in enumerate(columns_2):
                axs[i, 1].plot(column, label=f'Joint {i + 0}', color='C{}'.format(i))
                axs[i, 1].legend()
                axs[i, 1].grid()
            axs[0, 0].set_title(f'Left arm')
            axs[0, 1].set_title(f'Right arm')
            fig.text(0.77, 0.01, 'Frame', ha='center')
            fig.text(0.27, 0.01, 'Frame', ha='center')
            fig.text(0.01, 0.5, 'Angle [rad]', va='center', rotation='vertical')
        else:
            for i, column in enumerate(columns_1):
                axs[i].plot(column, label=f'Joint {i + 0}', color='C{}'.format(i))
                axs[i].legend()
                axs[i].grid()
            axs[0].set_title(f'Head')
            fig.text(0.5, 0.01, 'Frame', ha='center')
            fig.text(0.01, 0.5, 'Angle [rad]', va='center', rotation='vertical')

        plt.tight_layout()
        plt.show()

    def group_matrix(self, data):
        """
        Groups the data by columns.

        Args:
            data (list): A list of arrays representing the data.

        Returns:
            list: A list of arrays, where each array contains the elements from the corresponding column of the input data.
        """
        grouped_data = list(zip(*data))
        grouped_arrays = [np.array(group) for group in grouped_data]
        return grouped_arrays

    def mat_to_dict_per_joint(self, paths_left, paths_right=[], paths_head=[]):
        """
        Convert matrices to dictionaries per joint.

        Args:
            paths_left (list): List of matrices for the left joints.
            paths_right (list, optional): List of matrices for the right joints. Defaults to [].
            paths_head (list, optional): List of matrices for the head joints. Defaults to [].

        Returns:
            None
        """
        paths_left = self.group_matrix(paths_left)
        paths_right = self.group_matrix(paths_right)
        paths_head = self.group_matrix(paths_head)

        for count in range(len(paths_left)):
            self.robot.robotDict["jointLeft_" + str(count)] = paths_left[count]
        for count in range(len(paths_right)):
            self.robot.robotDict["jointRight_" + str(count)] = paths_right[count]
        for count in range(len(paths_head)):
            self.robot.robotDict["jointHead_" + str(count)] = paths_head[count]

    def plot_3d_paths(self, paths):
            """
            Plots 3D paths for each joint.

            Args:
                paths (ndarray): Array of shape (num_points, num_paths, 3) containing the X, Y, Z coordinates of the paths.

            Returns:
                None
            """
            num_points, num_paths, _ = paths.shape
            # Calculate the number of rows and columns based on the desired layout
            num_rows = (num_paths + 2) // 3  # Ensure there are enough rows for all paths
            num_cols = min(num_paths, 3)
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows), subplot_kw={'projection': '3d'})
            for i in range(num_paths):
                row = i // 3
                col = i % 3
                # Extract X, Y, Z coordinates
                x, y, z = paths[:, i, :].T
                axs[row, col].scatter(x, y, z, s=10) # label=f'Joint {i}'
                axs[row, col].plot(x, y, z, color='grey', linestyle='dashed') # label=f'Joint {i}'
                # Highlight
                axs[row, col].scatter(x[0], y[0], z[0], color='green', s=50, label='Start Point')
                axs[row, col].scatter(x[-1], y[-1], z[-1], color='red', s=50, label='End Point')
                axs[row, col].set_xlabel('X [mm]')
                axs[row, col].set_ylabel('Y [mm]')
                axs[row, col].set_zlabel('Z [mm]')
                axs[row, col].set_title(f'Joint {i}')
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right')
            # Remove empty subplots
            for i in range(num_paths, num_rows * num_cols):
                fig.delaxes(axs.flatten()[i])
            plt.tight_layout()
            plt.show()

    def linear_angular_mapping_gen3(self, vec):
        """
        Maps the given vector of angles to the range [0, 2*pi].

        Args:
            vec (numpy.ndarray): The input vector of angles.

        Returns:
            numpy.ndarray: The mapped vector of angles.

        """
        new_vec = []
        for l in vec:
            new_angle_vec = np.zeros_like(l)
            for i in range(len(l)):
                if l[i] >= 0:
                    new_angle_vec[i] = l[i]
                else:
                    new_angle_vec[i] = 2*np.pi + l[i]
            new_vec.append(new_angle_vec)
        return np.array(new_vec)

    def plot_animation_3d_labeling(self, point_clouds_list, initial_ra):
        """
        Plot a 3D animation of labeled point clouds with interactive sliders and a save button.

        Args:
            point_clouds_list (list): A list of tuples, each containing 6 point clouds.
            initial_ra (list): A list of initial angles for the sliders.

        Raises:
            ValueError: If the input data is not in the expected format.

        Returns:
            None
        """
        # Function code here
    def plot_animation_3d_labeling(self, point_clouds_list, initial_ra):
        limits = list(self.robot.physical_limits_left.values()) + list(self.robot.physical_limits_right.values()) + list(self.robot.physical_limits_head.values())
        if not isinstance(point_clouds_list, list) or not all(isinstance(pc, tuple) and len(pc) == 6 for pc in point_clouds_list):
            raise ValueError("Invalid input data. Expecting a list of tuples, each containing 5 point clouds")
        fig, ax1, ax2 = self.create_3d_plot()
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
        # Line plots for connections in both subplots
        lines1 = [ax1.plot([], [], [], c='b', linewidth=2)[0] for _ in range(len(point_clouds_list[0][0]) - 1)]
        lines2 = [ax1.plot([], [], [], c='g', linewidth=2, linestyle='--')[0] for _ in range(len(point_clouds_list[0][1]) - 1)]
        lines3 = [ax1.plot([], [], [], c='r', linewidth=2)[0] for _ in range(len(point_clouds_list[0][2]) - 1)]
        lines4 = [ax2.plot([], [], [], c='b', linewidth=2)[0] for _ in range(len(point_clouds_list[0][3]) - 1)]
        lines5 = [ax2.plot([], [], [], c='g', linewidth=2, linestyle='--')[0] for _ in range(len(point_clouds_list[0][4]) - 1)]
        lines6 = [ax2.plot([], [], [], c='r', linewidth=2)[0] for _ in range(len(point_clouds_list[0][5]) - 1)]

        # Create sliders for each angle in ra
        slider_axes = [plt.axes([0.2, 0.02 + 0.02 * i, 0.65, 0.03], facecolor='lightgoldenrodyellow') for i in range(len(initial_ra))]
        sliders = [Slider(slider_ax, f'Angle {i}', np.deg2rad(limits[i][0]), np.deg2rad(limits[i][1]), valinit=initial_ra[i]) for i, slider_ax in enumerate(slider_axes)]

        button_ax = plt.axes([0.8, 0.9, 0.1, 0.04])  # Adjust the position as needed
        button = Button(button_ax, 'Save CSV')

        def on_button_click(event):
            # Callback function for button click
            ra = np.array([slider.val for slider in sliders])
            self.save_mapping_csv(file_name, action, user, lht, lhp, hht, rht, rhp, hhp, ra)
            print("CSV saved!")

        # Connect the button click function
        button.on_clicked(on_button_click)

        def update_sliders(val, sliders):
            slider_values = [slider.val for slider in sliders]
            print("Slider Values:", slider_values)
            for slider, v in zip(sliders, val):
                slider.valtext.set_text(f'{slider.val:.2f}')

       # Set the update function for the sliders
        def slider_update_func(val, sliders=sliders):
            update_sliders(val, sliders)

        # Connect the slider update function to each slider
        for slider in sliders:
            slider.on_changed(lambda val, sliders=sliders: slider_update_func(val, sliders))

        def update(frame, ra):
            ra = np.array([slider.val for slider in sliders])  # Get the current values of sliders
            points1, points2, points3, points4, points5, points6 = point_clouds_list[frame]

            # Set the titles for each subplot
            ax1.set_title('Human', fontsize=20)
            ax2.set_title('Robot', fontsize=20)

            # Update line plots in both subplots
            for i in range(len(points1) - 1):
                lines1[i].set_data([points1[i, 0], points1[i + 1, 0]], [points1[i, 1], points1[i + 1, 1]])
                lines1[i].set_3d_properties([points1[i, 2], points1[i + 1, 2]])

            for i in range(len(points2) - 1):
                lines2[i].set_data([points2[i, 0], points2[i + 1, 0]], [points2[i, 1], points2[i + 1, 1]])
                lines2[i].set_3d_properties([points2[i, 2], points2[i + 1, 2]])

            for i in range(len(points3) - 1):
                lines3[i].set_data([points3[i, 0], points3[i + 1, 0]], [points3[i, 1], points3[i + 1, 1]])
                lines3[i].set_3d_properties([points3[i, 2], points3[i + 1, 2]])

            points4, points5, points6 = self.robot_embodiment(ra[0:7], ra[7:14], [0, 0])

            for i in range(len(points4) - 1):
                lines4[i].set_data([points4[i, 0], points4[i + 1, 0]], [points4[i, 1], points4[i + 1, 1]])
                lines4[i].set_3d_properties([points4[i, 2], points4[i + 1, 2]])

            for i in range(len(points5) - 1):
                lines5[i].set_data([points5[i, 0], points5[i + 1, 0]], [points5[i, 1], points5[i + 1, 1]])
                lines5[i].set_3d_properties([points5[i, 2], points5[i + 1, 2]])

            for i in range(len(points6) - 1):
                lines6[i].set_data([points6[i, 0], points6[i + 1, 0]], [points6[i, 1], points6[i + 1, 1]])
                lines6[i].set_3d_properties([points6[i, 2], points6[i + 1, 2]])
             # Update robot pose using the provided angles

            return lines1 + lines2 + lines3 + lines4 + lines5 + lines6

        # Create animation
        animation = FuncAnimation(fig, update, frames=len(point_clouds_list), fargs=(initial_ra,), interval=100, blit=True)
        # plt.tight_layout()
        plt.show()
        plt.ioff()

if __name__ == "__main__":
    robotName = "gen3"
    file_path = "./robot_configuration_files/"+ robotName + ".yaml"
    pose_predictor = Prediction(file_path, robotName)

    df = pose_predictor.read_file("combined_actions_filtered")
    actions = ['teacup', 'teapot', 'spoon', 'ladle', 'shallow_plate',
               'dinner_plate', 'knife', 'fork', 'salt_shaker',
               'sugar_bowl', 'mixer', 'pressure_cooker']

    robot_pose = []
    # action = "salt_shaker_right"
    # user = 15
    # users = np.arange(1, 21, 1)
    # left_side, right_side, head, time = pose_predictor.read_csv_combined(df, action, user)
    # left_side = left_side * 1000
    # right_side = right_side * 1000
    # head = head * 1000
    # file_name = "./data/robot_angles_" + robotName

    #Freddy azul - izquierda
    # 3th joints have diff references to rotate
    # for i in range(len(left_side)):
        # ra = np.array([
        # np.deg2rad(82.5), np.deg2rad(45), np.deg2rad(150), np.deg2rad(70), np.deg2rad(60), np.deg2rad(70), np.deg2rad(random.randint(-180, 181)),
        # np.deg2rad(-82.5), np.deg2rad(45), np.deg2rad(-150), np.deg2rad(70), np.deg2rad(-60), np.deg2rad(70), np.deg2rad(random.randint(-180, 181))])
    # ra = np.array([np.deg2rad(65), np.deg2rad(20), np.deg2rad(-100), np.deg2rad(-80), np.deg2rad(-90),
    #                np.deg2rad(100), np.deg2rad(-20), np.deg2rad(100), np.deg2rad(10), np.deg2rad(40),
    #                np.deg2rad(0), np.deg2rad(0)])
    # lht, lhp, rht, rhp, hht, hhp = pose_predictor.extract_spherical_angles_from_human(left_side[0], right_side[0], head[0])
    # points4, points5, points6 = pose_predictor.robot_embodiment(ra[0:5], ra[5:10], [10, 12])
    # robot_pose.append((left_side[0], right_side[0], head[0], points4, points5, points6))
    # pose_predictor.plot_animation_3d_labeling(robot_pose, ra)

    #NN TRAINING
    file_name = "robot_angles_or_" + robotName
    theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot = pose_predictor.read_training_data(file_name)
    if robotName == "gen3":
        right_arm_robot = pose_predictor.linear_angular_mapping_gen3(right_arm_robot)
        left_arm_robot = pose_predictor.linear_angular_mapping_gen3(left_arm_robot)
    pose_predictor.train_pytorch(pose_predictor.robot, theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, 1500)

    #NN TESTING
    df = pose_predictor.read_file("combined_actions")
    action = "spoon"
    user = 4
    robot_pose = []
    robot_pose_2 = []
    left_side, right_side, head, time = pose_predictor.read_csv_combined(df, action, user)

    left_side = left_side * 1000
    right_side = right_side * 1000
    head = head * 1000
    # df = pose_predictor.read_file("/QT_recordings/human/robot_angles_")
    # left_side, right_side, head, time = pose_predictor.read_recorded_action_csv(df, "arm_sides", 21)

    angles_left_vec = []
    angles_right_vec = []
    angles_head_vec = []
    cartesian_left_vec = []
    cartesian_right_vec = []
    cartesian_head_vec = []
    cartesian_left_vec_2 = []
    cartesian_right_vec_2 = []

    file_path = "./robot_configuration_files/" + robotName + ".yaml"
    qt = robot.Robot(robotName)
    qt.import_robot(file_path)
    robot_graphs = createRobotGraphs(qt)
    lib_dict = {}
    dep_dict = {}
    length = 15

    # end_effector_dict = find_end_effectors_keys(robot_graphs)

    # for key in end_effector_dict:
    #     lib_dict[key] = read_library_from_file(key, robotName)
    #     robot_graphs[key].read_graph_from_file(key, robotName)
        # robot_graphs[key].plot_graph(key)

    # dict_pose = extract_action_from_library(str(user) + action, lib_dict)
    # dep_dict = extract_angles_from_library(str(user) + action, lib_dict)

    # jointLeft_vectors = extract_vectors(dep_dict[end_effector_dict[0]])
    # jointRight_vectors = extract_vectors(dep_dict[end_effector_dict[1]])
    # jointHead_vectors = extract_vectors(dep_dict[end_effector_dict[2]])

    # for key in end_effector_dict:
    #     dep_dict[key] = robot_graphs[key].select_joint_dependencies(dict_pose[key])

    # print(dict_pose)
    # for key in robot_graphs:
        # robot_graphs[key].read_graph_from_file(key, robotName)
        # generated_trajectory = pose_predictor.robot.robotDict[key]
        # tra = path_planning(generated_trajectory, robot_graphs[key])
        # plotPath(action + " " + str(user) + " " + key, generated_trajectory, np.asarray(tra), np.asarray(dict_pose[key]))

    for i in range(len(left_side)):
        angles_left, angles_right, angles_head = pose_predictor.predict_pytorch(left_side[i], right_side[i], head[i])

        angles_left_vec.append(angles_left)
        angles_right_vec.append(angles_right)
        angles_head_vec.append(angles_head)

        points4, points5, points6 = pose_predictor.robot_embodiment(angles_left, angles_right, angles_head)
        cartesian_left_vec.append(points4)
        cartesian_right_vec.append(points5)
        cartesian_head_vec.append(points6)

        # points1, points2, points3 = pose_predictor.robot_embodiment(jointLeft_vectors[i], jointRight_vectors[i], jointHead_vectors[i])
        # cartesian_left_vec_2.append(points1)
        robot_pose.append((left_side[i], right_side[i], head[i], points4, points5, points6))
        # robot_pose_2.append((points4, points5, points6, points1, points2, points3))

    pose_predictor.mat_to_dict_per_joint(cartesian_left_vec, cartesian_right_vec, cartesian_head_vec)
    # pose_predictor.plot_3d_paths(np.asarray(cartesian_right_vec))
    # pose_predictor.plot_3d_paths(np.asarray(cartesian_left_vec_2))
    # pose_predictor.plot_3d_paths(np.asarray(cartesian_right_vec))
    # pose_predictor.plot_3d_paths(np.asarray(cartesian_head_vec))
    # pose_predictor.plot_angle_sequence(angles_left_vec, angles_right_vec)
    # pose_predictor.plot_angle_sequence(angles_head_vec)
    # pose_predictor.plot_animation_3d(robot_pose)
    pose_predictor.plot_animation_3d(robot_pose)
    # pose_predictor.plot_angle_sequence(angles_left_vec, jointLeft_vectors)
    # pose_predictor.plot_angle_sequence(cartesian_left_vec, points1)