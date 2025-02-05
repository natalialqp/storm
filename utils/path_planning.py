import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import json
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error
import copy

import utils.world_graph as world_graph
import utils.robot_graph as robot_graph
import utils.simulate_position as simulate_position
import utils.robot as robot

plt.rcParams.update({'font.size': 18})

class PathPlanning:
    def __init__(self, robotName, delta=15):
        """
        Initializes the GraphPlot object.
        This class is used to perform creation of the world, search, recognition and plot graphs for the robot and the world.
        Used for path planning and trajectory approximation in the robot's world.

        Args:
            delta (int): The length parameter for the graph.

        Attributes:
            robotName (str): The name of the robot.
            myRobot (Robot): The Robot object representing the robot.
            robot_graphs (list): The list of robot graphs.
            length (int): The length parameter for the graph.

        Returns:
            None
        """
        self.robotName = robotName
        file_path = "./robot_configuration_files/" + self.robotName + ".yaml"
        self.myRobot = robot.Robot(self.robotName)
        self.myRobot.import_robot(file_path)
        self.robot_graphs = self.createRobotGraphs()
        self.length = delta
        self.find_end_effectors_keys()

    def findEdgesOptimized(self, graph):
        """
        Finds the edges between nodes in a graph using an optimized approach.

        Parameters:
        - graph: The graph object containing nodes.

        Returns:
        - edges: A list of tuples representing the edges between nodes.
        """
        nodes = graph.get_nodes()
        points = np.array(nodes)
        tree = cKDTree(points)
        max_dist = 2 * self.length
        edges = []
        for i, point in tqdm(enumerate(points)):
            # Find neighbors within the maximum distance
            neighbors_indices = tree.query_ball_point(point, max_dist)
            for neighbor_index in neighbors_indices:
                if neighbor_index > i:
                    if i < len(nodes) and neighbor_index < len(nodes):
                        edges.append((points[i], points[neighbor_index]))
                    else:
                        print("Invalid indices:", i, neighbor_index)
        return edges

    def find_edges_optimized_robot(self, graph):
        """
        Find edges between nodes in a graph using an optimized algorithm for a robot.

        Parameters:
        - graph: The graph object containing nodes and their attributes.

        Returns:
        - edges: A list of tuples representing the edges between nodes.

        Description:
        This function finds the edges between nodes in a graph using an optimized algorithm for a robot.
        It uses a k-d tree data structure to efficiently search for neighboring nodes within a certain distance.
        The maximum distance for considering two nodes as neighbors is twice the length of the robot.
        The function returns a list of tuples, where each tuple represents an edge between two nodes.
        """
        nodes = graph.get_nodes()
        points = [graph.get_node_attr(node, "value") for node in nodes]
        tree = cKDTree(points)
        max_dist = 2 * self.length
        edges = []
        for i, point in enumerate(points):
            neighbors_indices = tree.query_ball_point(point, max_dist)
            for neighbor_index in neighbors_indices:
                if neighbor_index > i:
                    if i < len(nodes) and neighbor_index < len(nodes):
                        edges.append((points[i], points[neighbor_index]))
                    else:
                        print("Invalid indices:", i, neighbor_index)
        return edges

    def createWorldCubes(self, lowEdge, highEdge):
        """
        Creates a world of cubes within the specified boundaries.

        Args:
            lowEdge (list): The lower boundaries of the world in each dimension.
            highEdge (list): The upper boundaries of the world in each dimension.

        Returns:
            None
        """
        self.graph_world = world_graph.Graph()
        world = np.column_stack((lowEdge, highEdge))
        world = [np.sort(i) for i in world]
        for x in np.arange(world[0][0], world[0][1], self.length):
            for y in np.arange(world[1][0], world[1][1], self.length):
                for z in np.arange(world[2][0], world[2][1], self.length):
                    node = np.around([x, y, z], decimals=2)
                    self.graph_world.add_one_node(node)
        edges = self.findEdgesOptimized(self.graph_world)
        self.graph_world.add_edges(edges)

    def createObject(self, lowEdge, highEdge):
        """
        Creates a new object in the world graph.

        Args:
            lowEdge (list): The lower bounds of the object's dimensions.
            highEdge (list): The upper bounds of the object's dimensions.

        Returns:
            Graph: The newly created object in the world graph.
        """
        new_object = world_graph.Graph()
        world = np.column_stack((lowEdge, highEdge))
        world = [np.sort(i) for i in world]
        for x in np.arange(world[0][0], world[0][1], self.length):
            for y in np.arange(world[1][0], world[1][1], self.length):
                for z in np.arange(world[2][0], world[2][1], self.length):
                    node = np.around([x, y, z], decimals=2)
                    new_object.add_one_node(node)
        edges = self.findEdgesOptimized(new_object)
        new_object.add_edges(edges)
        return new_object

    def find_closest_point(self, new_point, kdtree):
        """
        Find the closest point from the KD-tree to the new point.

        This function takes a new 3D point represented as a NumPy array and a KD-tree built from a set of random points.
        It calculates the Euclidean distance between the new point and all points in the KD-tree, and returns the closest point.

        Parameters:
            new_point (numpy.ndarray): The new 3D point represented as a NumPy array of shape (3,).
            kdtree (scipy.spatial.KDTree): The KD-tree built from the set of random points.

        Returns:
            numpy.ndarray: The closest point from the set to the new point.
        """
        distance, index = kdtree.query(new_point)
        return kdtree.data[index]

    def findRobotWorld(self, demonstration, world):
        """
        Finds the robot world graph based on a given demonstration and world graph.

        Parameters:
        - demonstration: A list of points representing the demonstration.
        - world: The world graph.

        Returns:
        - robotWorld: The robot world graph.

        """
        kdtree = KDTree(world.get_nodes())
        robotWorld = world_graph.Graph()
        for i in demonstration:
            node = self.find_closest_point(i, kdtree)
            robotWorld.add_one_node(node)
        edges = self.findEdgesOptimized(robotWorld)
        robotWorld.add_edges(edges)
        return robotWorld

    def findJointsGraph(self, cartesian_points, joint_angles):
        """
        Finds the joints graph for a given set of Cartesian points and joint angles.

        Args:
            cartesian_points (dict): A dictionary of Cartesian points.
            joint_angles (dict): A dictionary of joint angles.

        Returns:
            dict: A dictionary of robot graphs.

        Raises:
            None
        """
        kdtree = KDTree(self.graph_world.get_nodes())
        for key in cartesian_points:
            if self.robotName == "gen3":
                if "0" in key or "1" in key or "2" in key:
                    node = cartesian_points[key]
                    dependencies = []
                else:
                    node = self.find_closest_point(cartesian_points[key], kdtree)
                    dependencies = self.slice_dict(joint_angles, key.split('_'))
            else:
                if "0" in key or "1" in key:
                    node = cartesian_points[key]
                    dependencies = []
                else:
                    node = self.find_closest_point(cartesian_points[key], kdtree)
                    dependencies = self.slice_dict(joint_angles, key.split('_'))
            self.robot_graphs[key].set_attribute(node, dependencies)
            edges = self.find_edges_optimized_robot(self.robot_graphs[key])
            self.robot_graphs[key].add_edges(edges)
        return self.robot_graphs

    def slice_dict(self, dict, details):
        """
        Slice a dictionary based on specified details.

        Args:
            dict (dict): The dictionary to be sliced.
            details (list): A list containing the details for slicing.

        Returns:
            list: A list containing the sliced elements from the dictionary.
        """
        sub_list = []
        for i in dict:
            if self.robotName == "gen3":
                if details[0] in i and len(sub_list) <= int(details[1]) - 3:
                    sub_list.append(dict[i])
            else:
                if details[0] in i and len(sub_list) <= int(details[1]) - 2:
                    sub_list.append(dict[i])
        return sub_list

    def createRobotGraphs(self):
        """
        Creates and returns a dictionary of robot graphs.

        This method iterates over the left arm angles, right arm angles, and head angles of the robot,
        and creates a graph object for each joint angle. The graph objects are stored in a dictionary
        with keys in the format "joint<Side>_<Index>", where <Side> is either "Left", "Right", or "Head",
        and <Index> is the index of the joint angle.

        Returns:
            dict: A dictionary containing the robot graphs, with keys representing the joint names and
                  values representing the corresponding graph objects.
        """
        joint_dic = {}
        for i in range(len(self.myRobot.leftArmAngles)):
            joint_dic["jointLeft_" + str(i)] = robot_graph.Graph(i, "jointLeft_" + str(i))
        for i in range(len(self.myRobot.rightArmAngles)):
            joint_dic["jointRight_" + str(i)] = robot_graph.Graph(i, "jointRight_" + str(i))
        if self.robotName != "gen3":
            for i in range(len(self.myRobot.headAngles)):
                joint_dic["jointHead_" + str(i)] = robot_graph.Graph(i, "jointHead_" + str(i))
        return joint_dic

    def error_between_points(self, point1, point2, max_distance):
        """
        Calculates a new point between two given points based on a maximum distance.

        Args:
            point1 (numpy.ndarray): The starting point.
            point2 (numpy.ndarray): The ending point.
            max_distance (float): The maximum distance between the new point and the starting point.

        Returns:
            numpy.ndarray: The new point calculated based on the maximum distance.

        """
        anomaly_direction = point2 - point1
        anomaly_magnitude = np.linalg.norm(anomaly_direction)
        # Ensure the anomaly is not too close to the previous point
        if anomaly_magnitude > max_distance:
            anomaly_direction /= anomaly_magnitude  # Normalize the direction
            new_point = point1 + max_distance * anomaly_direction
        else:
            new_point = point2
        return new_point

    def path_planning(self, demonstration, graph):
        """
        Performs path planning based on a given demonstration and a graph.

        Args:
            demonstration (list): List of points representing the demonstration.
            graph (Graph): Graph object containing nodes and edges.
            max_distance (float, optional): Maximum distance allowed between points. Defaults to 30.

        Returns:
            tuple: A tuple containing the planned path, mean squared error (MSE), and root mean squared error (RMSE).
        """
        max_distance = 2*self.length
        kdtree = KDTree(graph.get_nodes_values())
        path = []
        first_node = demonstration[0]
        prev_node = self.find_closest_point(first_node, kdtree)
        path.append(tuple(prev_node))
        for i in range(1, len(demonstration)):
            candidate_node = self.error_between_points(prev_node, demonstration[i], max_distance)
            node = self.find_closest_point(candidate_node, kdtree)
            path.append(tuple(node))
            prev_node = node
        # MSE, RMSE = self.error_calculation(demonstration, path)
        MSE, RMSE = 0, 0
        return path, MSE, RMSE

    def path_planning_online(self, demonstration, graph, max_distance = 30):
            """
            Performs online path planning based on a given demonstration and a graph.

            Args:
                demonstration (list): List of points representing the demonstration.
                graph (Graph): Graph object representing the environment.
                max_distance (float, optional): Maximum distance allowed between points. Defaults to 30.

            Returns:
                float: The last joint angle of the calculated path.
            """
            kdtree = KDTree(graph.get_nodes_values())
            path = []
            first_node = demonstration[0]
            prev_node = self.find_closest_point(first_node, kdtree)
            path.append(prev_node.tolist())
            for i in range(1, len(demonstration)):
                candidate_node = self.error_between_points(prev_node, demonstration[i], max_distance)
                node = self.find_closest_point(candidate_node, kdtree)
                path.append(node)
                prev_node = node
            path_angles = graph.select_joint_dependencies(path[0])
            return path_angles[-1]

    def plot_error(self, Dict, iter):
        """
        Plot the values of Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for each joint.

        Args:
            Dict (dict): A dictionary containing the error values for each joint.
            iter (int): The iteration number.

        Returns:
            None
        """
        # Extracting data for plotting
        joints = list(Dict.keys())
        values_MSE = [sub_dict["MSE"] for sub_dict in Dict.values()]
        values_RMSE = [sub_dict["RMSE"] for sub_dict in Dict.values()]

        # Plotting
        bar_width = 0.35
        index = range(len(joints))

        fig, ax = plt.subplots()
        bar1 = ax.bar(index, values_MSE, bar_width, label='MSE')
        bar2 = ax.bar([i + bar_width for i in index], values_RMSE, bar_width, label='RMSE')

        # Adding labels and title
        ax.set_xlabel('Joints')
        ax.set_ylabel('Error')
        ax.set_title('Values of MSE and RMSE for each Joint in the iteration ' + str(iter))
        ax.set_xticks([i + bar_width/2 for i in index])
        ax.set_xticklabels(joints,  rotation=0, ha='center')
        ax.legend()
        plt.grid()
        # Show the plot
        plt.savefig("error_values_iter_" + str(iter) + ".pdf", format="pdf")
        plt.show()

    def error_calculation(self, y_true, y_pred):
        """
        Calculates the mean squared error (MSE) and root mean squared error (RMSE) between the true values and predicted values.

        Parameters:
        - y_true: The true values.
        - y_pred: The predicted values.

        Returns:
        - MSE: The mean squared error.
        - RMSE: The root mean squared error.
        """
        RMSE = mean_squared_error(y_true, y_pred, squared=False)
        MSE = mean_squared_error(y_true, y_pred, squared=True)
        return MSE, RMSE

    def approximateTrajectory(self, demonstration, robot_joint_graph):
        """
        Approximates the trajectory by finding the closest point in the robot joint graph for each point in the demonstration.

        Args:
            demonstration (list): List of points representing the demonstration.
            robot_joint_graph (Graph): The robot joint graph.

        Returns:
            list: List of nodes representing the approximate trajectory.
        """
        kdtree = KDTree(robot_joint_graph.get_nodes_values())
        trajectory = []
        for i in demonstration:
            node = self.find_closest_point(i, kdtree)
            trajectory.append(node)
        return trajectory

    def plotPath(self, key, demonstration, path, candidates=[]):
        """
        Plot the given demonstration path and repaired path in a 3D scatter plot.

        Parameters:
        - key (str): The title of the plot.
        - demonstration (numpy.ndarray): The coordinates of the demonstration path points.
        - path (numpy.ndarray): The coordinates of the repaired path points.
        - candidates (numpy.ndarray, optional): The coordinates of the candidate points. Default is an empty list.

        Returns:
        None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        xdem = demonstration[:, 0]
        ydem = demonstration[:, 1]
        zdem = demonstration[:, 2]
        xpath = path[:, 0]
        ypath = path[:, 1]
        zpath = path[:, 2]

        # Scatter plots for the points
        ax.scatter3D(xdem[1:-1], ydem[1:-1], zdem[1:-1], c='blue', marker='o', label='Original path')
        ax.scatter3D(xdem[0], ydem[0], zdem[0], c='green', marker='*')
        ax.scatter3D(xdem[-1], ydem[-1], zdem[-1], c='red', marker='*')

        ax.scatter3D(xpath[1:-1], ypath[1:-1], zpath[1:-1], c='red', marker='o', label='Repaired path')
        ax.scatter3D(xpath[0], ypath[0], zpath[0], c='green', marker='*', label='Starting point')
        ax.scatter3D(xpath[-1], ypath[-1], zpath[-1], c='red', marker='*', label='Ending point')

        if len(candidates):
            xdata = candidates[:, 0]
            ydata = candidates[:, 1]
            zdata = candidates[:, 2]
            ax.scatter3D(xdata, ydata, zdata, c='purple', label='Candidates')

        # Plot edges between the points
        for i in range(len(xdem) - 1):
            ax.plot([xdem[i], xdem[i + 1]], [ydem[i], ydem[i + 1]], [zdem[i], zdem[i + 1]], c='grey')
        for i in range(len(xpath) - 1):
            ax.plot([xpath[i], xpath[i + 1]], [ypath[i], ypath[i + 1]], [zpath[i], zpath[i + 1]], c='lightblue')

        max_range = max(np.ptp(arr) for arr in [xdem, ydem, zdem, xpath, ypath, zpath])

        # Set equal aspect ratio for all axes
        ax.set_box_aspect([max_range, max_range, max_range])

        ax.set_title(key)
        ax.set_xlabel("\n X [mm]", linespacing=3.2)
        ax.set_ylabel("\n Y [mm]", linespacing=3.2)
        ax.set_zlabel("\n Z [mm]", linespacing=3.2)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        ax.legend()
        fig.tight_layout()
        plt.show()

    def printRobotGraphs(self, robot):
        """
        Prints the graphs of the given robot.

        Parameters:
        - robot: A dictionary containing the robot's graphs.

        Returns:
        None
        """
        for key in robot:
            print(key, ": ")
            robot[key].print_graph()

    def readTxtFile(self, file_path):
        """
        Reads a text file and returns its contents as a JSON object.

        Args:
            file_path (str): The path to the text file.

        Returns:
            dict: The contents of the text file as a JSON object.
        """
        with open(file_path) as f:
            contents = f.read()
        return json.loads(contents)

    def divideFrames(self, data):
        """
        Divides the data into sub-frames and concatenates them into a new array.

        Args:
            data (dict): A dictionary containing arrays of data.

        Returns:
            np.ndarray: A new array with the sub-frames concatenated.
        """
        sub_frame = []
        for key in data:
            frame = [np.array([el]) for el in data[key]]
            sub_frame.append(frame)
            new_data = [np.array([0]) for el in data[key]]
        new_data = np.array(new_data)
        for frame in sub_frame:
            new_data = np.hstack((new_data, np.array(frame)))
        return new_data[:,1:]

    def forward_kinematics_n_frames(self, joint_angles):
        """
        Calculates the forward kinematics for multiple frames of joint angles.

        Args:
            joint_angles (list): A list of joint angles for each frame.

        Returns:
            tuple: A tuple containing three lists: left, right, and head.
                - left (list): The positions of the left side of the robot for each frame.
                - right (list): The positions of the right side of the robot for each frame.
                - head (list): The positions of the head of the robot for each frame.
        """
        left = []
        right = []
        head = []
        for frame in joint_angles:
            frame = np.radians(np.array(frame))
            forward_kinematics = getattr(self.myRobot, "forward_kinematics_" + self.robotName)
            pos_left, pos_right, pos_head = forward_kinematics(frame)
            left.append(copy.copy(pos_left))
            right.append(copy.copy(pos_right))
            head.append(copy.copy(pos_head))
        return left, right, head

    def read_babbling(self, path_name):
        """
        Read babbling data from a text file.

        Args:
            path_name (str): The path to the text file.

        Returns:
            numpy.ndarray: An array of joint angles.

        """
        points = self.readTxtFile("./data/" + path_name)
        keys = list(points)
        joint_angles = points[keys[0]]
        for count in range(1, len(keys)):
            joint_angles = np.hstack((joint_angles, points[keys[count]]))
        return joint_angles.astype(float)

    def learn_environment(self, cartesian_points, joint_angles_dict):
        """
        Learns the environment by finding the robot graphs for each set of cartesian points and joint angles.

        Args:
            cartesian_points (list): A list of cartesian points.
            joint_angles_dict (list): A list of joint angles dictionaries.

        Returns:
            list: A list of robot graphs.

        """
        for i in tqdm(range(len(joint_angles_dict))):
            self.robot_graphs = self.findJointsGraph(cartesian_points[i], joint_angles_dict[i])
        return self.robot_graphs

    def angle_interpolation(self, joint_angles_list):
            """
            Interpolates the given joint angles list to create a smoother trajectory.

            Parameters:
            joint_angles_list (list): List of joint angles.

            Returns:
            numpy.ndarray: Array of interpolated joint angles.
            """
            new_joint_angles_list = []
            for count in range(len(joint_angles_list) - 1):
                diff = joint_angles_list[count + 1] - joint_angles_list[count]
                iterations = np.max(np.abs(diff))
                delta = diff / iterations
                actual_angle = joint_angles_list[count]
                for i in range(int(iterations)):
                    new_joint_angles_list.append(copy.copy(actual_angle))
                    actual_angle += delta
            new_joint_angles_list.append(joint_angles_list[-1])
            return np.array(new_joint_angles_list)

    def list_to_string(self, vec):
        """
        Converts a list of values to a string representation.

        Args:
            vec (list): The input list of values.

        Returns:
            str: The string representation of the input list.
        """
        modified_vector = [0.0 if value in {0, 0., -0., -0.0, -0} else value for value in vec]
        vector_str = '[' + ', '.join(map(str, modified_vector)) + ']'
        return vector_str

    def find_object_in_world(self, new_object):
        """
        Finds the closest points in the world graph to the nodes of a new object.

        Parameters:
        - new_object: The object whose nodes need to be matched with the closest points in the world graph.

        Returns:
        - object_nodes: A list of the closest points in the world graph to the nodes of the new object.
        - object_nodes_str: A list of string representations of the closest points.

        """
        kdtree = cKDTree(self.graph_world.get_nodes())
        object_nodes = []
        object_nodes_str = []
        for node in new_object.get_nodes():
            closest = self.find_closest_point(node, kdtree)
            object_nodes_str.append(self.list_to_string(closest))
            object_nodes.append(closest)
        return object_nodes, object_nodes_str

    def match_object_in_world(self, objectName, lowEdge_robot, highEdge_robot):
        """
        Matches an object in the world based on its name.

        Args:
            objectName (str): The name of the object to be matched.

        Returns:
            None
        """
        self.createWorldCubes(lowEdge_robot, highEdge_robot)
        object = world_graph.Graph()
        object2 = world_graph.Graph()
        object.read_object_from_file(self.robotName, objectName)
        object_nodes, _ = self.find_object_in_world(object)
        object2.add_nodes(object_nodes)
        object2.save_object_to_file(self.robotName, objectName + "_nodes")

    def find_trajectory_in_world(self, tra):
        """
        Finds the trajectory in the world by finding the closest point in the graph to each node in the trajectory.

        Args:
            tra (list): List of nodes representing the trajectory.

        Returns:
            list: List of nodes representing the trajectory in the world.
        """
        kdtree = cKDTree(self.graph_world.get_nodes())
        world_nodes = []
        for node in tra:
            closest_point = self.find_closest_point(node, kdtree)
            if world_nodes:
                if (world_nodes[-1] != closest_point).any():
                    world_nodes.append(closest_point)
            else:
                world_nodes.append(closest_point)
        return world_nodes

    def read_library_from_file(self, name, babbled_file):
        """
        Reads a library from a file.

        Args:
            name (str): The name of the library.
            babbled_file (str): The name of the babbled file.

        Returns:
            list: The data read from the file.

        Raises:
            FileNotFoundError: If the file is not found.
        """
        try:
            with open("data/test_" + self.robotName + "/paths_lib/" + name + "_" + babbled_file + ".json", "r") as jsonfile:
                data = [json.loads(line) for line in jsonfile.readlines()]
                return data
        except FileNotFoundError:
            print(f"File {name}.json not found.")
            return []

    def extract_action_from_library(self, actionName, library):
        """
        Extracts the robot pose for a given action name from the library.

        Args:
            actionName (str): The name of the action to extract the robot pose for.
            library (dict): The library containing the actions and their corresponding robot poses.

        Returns:
            dict: A dictionary mapping the robot pose keys to their corresponding paths.
        """
        robot_pose = {}
        for key in library:
            for item in library[key]:
                if item["id"] == actionName:
                    robot_pose[key] = item["path"]
        return robot_pose

    def extract_angles_from_library(self, actionName, library):
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

    def find_end_effectors_keys(self):
        """
        Finds the keys of the end effectors in the robot graphs.

        Returns:
            list: A list of keys representing the end effectors in the robot graphs.
        """
        left_numbers = [int(key.split('_')[1]) for key in self.robot_graphs if key.startswith('jointLeft')]
        right_numbers = [int(key.split('_')[1]) for key in self.robot_graphs if key.startswith('jointRight')]
        head_numbers = [int(key.split('_')[1]) for key in self.robot_graphs if key.startswith('jointHead')]

        max_left = 'jointLeft_' + str(max(left_numbers, default=0))
        max_right = 'jointRight_' + str(max(right_numbers, default=0))
        max_head = 'jointHead_' + str(max(head_numbers, default=0))
        self.end_effectors_keys = [max_left, max_right, max_head]

    def find_fk_from_dict(self, dict):
        """
        Finds the forward kinematics from a dictionary of joint data.

        Args:
            dict (dict): A dictionary containing joint data.

        Returns:
            None
        """
        result_list = []
        for position in range(max(len(joint_data) for joint_data in dict.values())):
            joint_angles = []
            for joint_name, joint_data in dict.items():
                if position < len(joint_data):
                    joint_angles.extend(list(joint_data[position].values()))
            result_list.append(joint_angles)
        pos_left, pos_right, pos_head = self.forward_kinematics_n_frames(result_list)
        s = simulate_position.RobotSimulation(pos_left, pos_right, pos_head)
        s.animate()

    def fill_robot_graphs(self, babbling_file):
        """
        Fills the robot graphs with data from a specified file.

        Parameters:
        - babbling_file (str): The path to the file containing the data.

        Returns:
        None
        """
        for key in self.robot_graphs:
            self.robot_graphs[key].read_graph_from_file(key, self.robotName, babbling_file)
            # self.robot_graphs[key].plot_graph(self.robotName)

    def identify_new_nodes(self, old_path, new_path):
        """
        Identifies the new nodes in the given new path that are not present in the old path.

        Args:
            old_path (list): The list of nodes in the old path.
            new_path (list): The list of nodes in the new path.

        Returns:
            list: A list of new nodes that are present in the new path but not in the old path.
        """
        new_nodes = []
        for node in new_path:
            if node not in old_path and node not in new_nodes:
                new_nodes.append(node)
        return new_nodes
