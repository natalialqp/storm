import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import ast
from scipy.spatial import KDTree
import xml.etree.ElementTree as ET
import json
import re
from scipy.optimize import minimize
from math import radians

plt.rcParams.update({"font.size": 20})

class Graph(object):
    """
    A class representing a graph used for robot manipulation planning.

    This class provides methods to manipulate and analyze the graph, such as adding nodes and edges,
    setting attributes, performing searches, and finding shortest paths.

    Attributes:
        G (nx.Graph): The graph object.
        n_dependecies (int): The number of dependencies.
        objects_in_world (dict): A dictionary of objects in the world.
        key (str): The key for the object.
        nodes_id (int): The ID of the nodes.
    """

    def __init__(self, n_dependecies, key):
        """
        Initializes a RobotGraph object.

        Args:
            n_dependecies (int): The number of dependencies.
            key (str): The key for the object.

        Attributes:
            G (nx.Graph): The graph object.
            n_dependecies (int): The number of dependencies.
            objects_in_world (dict): A dictionary of objects in the world.
            key (str): The key for the object.
            nodes_id (int): The ID of the nodes.
        """
        self.G = nx.Graph()
        self.n_dependecies = n_dependecies
        self.objects_in_world = {}
        self.key = key
        self.nodes_id = 0

    def get_all_nodes_attr(self, attr):
        """
        Get the attribute values of all nodes in the graph.

        Parameters:
        - attr (str): The name of the attribute to retrieve.

        Returns:
        - list: A list of attribute values for all nodes in the graph.
        """
        nodes = self.get_nodes()
        node_values = [self.get_node_attr(node, attr) for node in nodes]
        return node_values

    def get_node_attr(self, node, attr):
        """
        Get the value of a specific attribute for a given node.

        Args:
            node (str or list): The node name or list of node names.
            attr (str): The attribute name.

        Returns:
            The value of the specified attribute for the given node.
        """
        if not isinstance(node, str):
            node = self.list_to_string(node)
        return self.G.nodes[node][attr]

    def change_node_attr(self, node, attr, value):
        """
        Change the attribute value of a node in the graph.

        Args:
            node: The node whose attribute value needs to be changed.
            attr: The attribute name to be modified.
            value: The new value to be assigned to the attribute.

        Returns:
            None
        """
        aux_node = str(node)
        self.G.nodes[aux_node][attr] = value

    def list_to_string(self, vec):
        """
        Converts a list of numbers to a string representation.

        Args:
            vec (list): The input list of numbers.

        Returns:
            str: The string representation of the input list.
        """
        modified_vector = [0.0 if value in {0, 0., -0., -0.0, -0} else value for value in vec]
        vector_str = "[" + ", ".join(map(str, modified_vector)) + "]"
        return vector_str

    def set_attribute(self, node, angles):
        """
        Set the attribute of a node in the robot graph.

        Args:
            node (str): The name of the node.
            angles (list): A list of angles.

        Returns:
            None
        """
        if len(angles) > 0:
            keyList = np.arange(1, self.n_dependecies - 1, 1)
        else:
            keyList = []
        dependencies = {"angle_" + str(keyList[i]): round(angles[i], 2) for i in range(len(keyList))}
        self.add_one_node(node, [dependencies])

    def set_edge_attr(self, u, v, attr):
        """
        Set the 'occupied' attribute of the edge between nodes u and v in the graph.

        Parameters:
            u (node): The source node of the edge.
            v (node): The target node of the edge.
            attr (bool): The value to set for the 'occupied' attribute.

        Returns:
            None
        """
        nx.set_edge_attributes(self.G, {(u, v): {"occupied": attr}})

    def add_one_node(self, node, attr):
        """
        Adds a new node to the graph.

        Args:
            node (list): The node to be added.
            attr (list): The joint dependency attributes of the node.

        Returns:
            None
        """
        new_node = self.list_to_string(node)
        if not self.has_node(new_node):
            self.G.add_node(new_node, value=node, nodes_id=self.nodes_id, occupied=False, joint_dependency=attr)
            self.nodes_id += 1
        else:
            node_attr = self.get_node_attr(new_node, "joint_dependency")
            for a in attr:
                self.update_joint_attr_dict(new_node, a, node_attr)

    def update_joint_attr_dict(self, node, dict, dict_list):
        """
        Update the joint attribute dictionary and list.

        This function checks if the new dictionary is different from the existing dictionaries in the list.
        If it is different, the new dictionary is appended to the list and the node attribute is updated.

        Args:
            node (Node): The node to update the attribute for.
            dict (dict): The new joint attribute dictionary.
            dict_list (list): The list of existing joint attribute dictionaries.

        Returns:
            None
        """
        is_different = all(old_dict[attr] != dict[attr] for old_dict in dict_list for attr in dict) and len(dict) != 0
        if is_different:
            dict_list.append(dict)
            self.change_node_attr(node, "joint_dependency", dict_list)

    def change_edge_weight(self, u, v, value):
        """
        Change the weight of an edge in the graph.

        Args:
            u: The source node of the edge.
            v: The target node of the edge.
            value: The new weight value for the edge.

        Returns:
            None
        """
        self.G[u][v]["weight"] = value

    def add_one_edge(self, edge):
        """
        Adds a single edge to the graph.

        Parameters:
        - edge: A tuple representing the edge to be added.

        Returns:
        None
        """
        u, v = self.list_to_string(edge[0]), self.list_to_string(edge[1])  # Convert ndarrays to tuples
        self.G.add_edge(u, v, weight=1)
        self.set_edge_attr(u, v, False)

    def add_edges(self, edges):
            """
            Add multiple edges to the graph.

            Args:
                edges (list): A list of edges to be added to the graph.

            Returns:
                None
            """
            for i in edges:
                self.add_one_edge(i)

    def has_edge(self, u, v):
        """
        Check if there is an edge between two nodes in the graph.

        Parameters:
        - u: The first node.
        - v: The second node.

        Returns:
        - True if there is an edge between u and v, False otherwise.
        """
        u, v = tuple(u), tuple(v)
        return self.G.has_edge(u, v)

    def save_file(self, name):
        """
        Save the graph as a GML file.

        Parameters:
        - name (str): The name of the file to save.

        Returns:
        None
        """
        nx.write_gml(self.G, name + ".gml.gz", stringizer=list)

    def search(self, initial_node, goal_node, name):
        """
        Perform a search algorithm to find a path from the initial node to the goal node.

        Args:
            initial_node: The starting node of the search.
            goal_node: The target node to reach.
            name: The name of the search algorithm to use.

        Returns:
            A list of nodes representing the path from the initial node to the goal node.
        """
        if name == "A*":
            path = nx.astar_path(self.G, initial_node, goal_node)
        return path

    def shortest_path(self, initial_node, goal_node):
        """
        Finds the shortest path between two nodes in the graph.

        Args:
            initial_node (list): The initial node coordinates.
            goal_node (list): The goal node coordinates.

        Returns:
            list: The shortest path as a list of nodes.

        """
        initial_node = self.list_to_string(initial_node)
        goal_node = self.list_to_string(goal_node)
        return nx.shortest_path(self.G, str(initial_node), str(goal_node))

    def number_of_nodes(self):
        """
        Returns the number of nodes in the graph.

        Returns:
            int: The number of nodes in the graph.
        """
        return self.G.number_of_nodes()

    def has_path(self, initial_node, goal_node):
        """
        Check if there is a path between the initial_node and the goal_node in the graph.

        Args:
            initial_node (list): The initial node represented as a list.
            goal_node (list): The goal node represented as a list.

        Returns:
            bool: True if there is a path between the initial_node and the goal_node, False otherwise.
        """
        initial_node = self.list_to_string(initial_node)
        goal_node = self.list_to_string(goal_node)
        return nx.has_path(self.G, initial_node, goal_node)

    def has_node(self, node):
        """
        Check if the graph has a specific node.

        Args:
            node: The node to check. It can be either a string or a list.

        Returns:
            True if the graph has the node, False otherwise.
        """
        if type(node) == list:
            node = self.list_to_string(node)
        return self.G.has_node(node)

    def get_nodes(self):
        """
        Returns a list of nodes in the graph.

        Returns:
            list: A list of nodes in the graph.
        """
        return self.G.nodes()

    def find_nodes_with_angle_tolerance(self, input_values_list, tolerance = 5):
        """
        Finds nodes in the graph that have joint dependencies with angles within a specified tolerance.

        Parameters:
        - input_values_list (list): A list of dictionaries containing input values for each joint.
        - tolerance (int): The maximum difference allowed between the input values and the joint angles.

        Returns:
        - result_ids (list): A list of node IDs that meet the angle tolerance criteria.
        """
        result_ids = []
        for node in self.G.nodes(data = True):
            node_id = node[0]
            joint_dependency = node[1].get("joint_dependency")
            if joint_dependency:
                for input_values in input_values_list:
                    meets_tolerance = any(
                        all(
                            abs(angle_dict.get(key, 0) - input_values.get(key, 0)) <= tolerance
                            for key in input_values
                        )
                        for angle_dict in joint_dependency
                    )
                    if meets_tolerance:
                        result_ids.append(node_id)
                        break
        return result_ids

    def new_object_in_world(self, object_nodes, object_name, end_effector, prev_joint_angles=[]):
        """
        Adds a new object to the world graph.

        Args:
            object_nodes (list): List of nodes representing the object.
            object_name (str): Name of the object.
            end_effector (bool): Indicates whether the object is an end effector or not.
            prev_joint_angles (list, optional): List of previous joint angles. Defaults to [].

        Returns:
            list: List of future dependencies for the added object.
        """
        storage_graph = nx.Graph()
        if end_effector and prev_joint_angles:
            aux = self.find_nodes_with_angle_tolerance(prev_joint_angles)
            object_nodes.extend(aux)
        future_dependencies = []
        for node in object_nodes:
            if self.has_node(node):
                node_attr = self.G.nodes[node]
                local_edges = self.G.edges(node, data=True)
                storage_graph.add_node(node, **node_attr)
                storage_graph.add_edges_from(local_edges)
                self.G.remove_node(node)
                if end_effector == False:
                    future_dependencies.extend(node_attr["joint_dependency"])
        self.objects_in_world[object_name] = storage_graph
        return future_dependencies

    def remove_object_from_world(self, name):
        """
        Removes an object from the world graph.

        Args:
            name (str): The name of the object to be removed.

        Returns:
            None
        """
        self.find_subgraphs()
        # self.plot_graph()
        deleted_object = self.objects_in_world[name]
        self.G.add_nodes_from(deleted_object.nodes.items())
        self.G.add_edges_from(deleted_object.edges)
        del self.objects_in_world[name]
        self.find_subgraphs()
        # self.plot_graph()

    def get_nodes_values(self):
        """
        Returns a list of values associated with each node in the graph.

        Returns:
            list: A list of values associated with each node in the graph.
        """
        nodes = self.G.nodes()
        return [self.get_node_attr(node, "value") for node in nodes]

    def get_edges(self):
        """
        Returns the edges of the graph.

        Returns:
            A list of edges in the graph.
        """
        return self.G.edges()

    def print_graph(self):
        """
        Print the list of nodes and edges in the graph.

        This function iterates over the nodes and edges of the graph and prints their attributes.

        Args:
            None

        Returns:
            None
        """
        print("List of nodes:")
        for node, attributes in self.G.nodes(data=True):
            print("Node:", node, "Attributes:", attributes)
        print("List of edges:")
        for u, v, attributes in self.G.edges(data=True):
            print("Edge:", u, "-", v, "Attributes:", attributes)

    def read_last_id_path_from_file(self, robotName):
        """
        Reads the last ID from a JSON file containing robot data.

        Parameters:
        - robotName (str): The name of the robot.

        Returns:
        - int: The last ID found in the JSON file.
        """
        try:
            with open(self.key + robotName + "_data.json", "r") as jsonfile:
                lines = jsonfile.readlines()
                if lines:
                    last_line = lines[-1]
                    last_pose = json.loads(last_line)
                    last_id = last_pose.get("id", 0)
                    return last_id
                else:
                    return 0
        except FileNotFoundError:
            return 0
        except json.JSONDecodeError:
            return 0

    def save_path_in_library(self, trajectory, dependency, robotName, actionName, babbling_file):
        """
        Save a path in the library.

        Args:
            trajectory (list): The trajectory to be saved.
            dependency (list): The joint dependency of the trajectory.
            robotName (str): The name of the robot.
            actionName (str): The name of the action.
            babbling_file (str): The name of the babbling file.

        Returns:
            None
        """
        with open("./data/test_" + robotName + "/paths_lib/" + self.key + "_" + babbling_file + ".json", "a") as jsonfile:
            pose = {"id": actionName, "path": trajectory, "joint_dependency": dependency}
            jsonfile.write(json.dumps(pose))
            jsonfile.write("\n")

    def save_graph_to_file(self, name, robot, babbled_nodes):
        """
        Save the graph to a file in XML format.

        Parameters:
        - name (str): The name of the graph.
        - robot (str): The name of the robot.
        - babbled_nodes (str): The number of babbled nodes.

        Returns:
        None
        """
        output_file = "./data/test_" + robot + "/graphs/joints_graphs_lib/" + name + "_" + babbled_nodes + ".xml"
        with open(output_file, "w", encoding="utf-8") as file:
            file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            file.write("<graph>\n")

            for node, attributes in self.G.nodes(data=True):
                x, y, z = attributes["value"]
                value = attributes["value"]
                occupied = attributes["occupied"]
                nodes_id = attributes["nodes_id"]
                joint_dependency = attributes["joint_dependency"]
                file.write(f' <node id="[{x}, {y}, {z}]" value= "{value}" nodes_id = "{nodes_id}" occupied="{occupied}" joint_dependency="{joint_dependency}"/>\n')
            for u, v, attributes in self.G.edges(data = True):
                source = f"{u}"
                target = f"{v}"
                weight = attributes["weight"]
                occupied = attributes["occupied"]
                file.write(f' <edge source="{source}" target="{target}" weight="{weight}" occupied="{occupied}"/>\n')
            file.write("</graph>\n")

    def select_joint_dependencies(self, trajectory):
        """
        Selects the joint dependencies for each point in the given trajectory.

        Args:
            trajectory (list): A list of points representing the trajectory.

        Returns:
            list: A list of dictionaries representing the selected joint dependencies for each point in the trajectory.
        """
        prev_dict =  {"angle_" + str(key + 1): 0 for key in range(self.n_dependecies - 1)}#change for initial position of the robot joints
        dependencies = []
        for point in trajectory:
            if not isinstance(point, str):
                point = self.list_to_string(point)
            dep_list = self.get_node_attr(point, "joint_dependency")
            least_error_dict = self.find_least_error_dict(prev_dict, dep_list)
            dependencies.append(least_error_dict)
            prev_dict = least_error_dict
        return dependencies

    def calculate_error(self, d_1, d_2):
        """
        Calculates the error between two dictionaries.

        Parameters:
        - d_1 (dict): The first dictionary.
        - d_2 (str): The second dictionary in string format.

        Returns:
        - error (float): The error between the two dictionaries.
        """
        d_2 = self.string_to_dict(d_2)
        return sum((d_1[key] - d_2[key]) ** 2 for key in d_1 if key in d_2)

    def find_least_error_dict(self, reference_dict, dict_list):
        """
        Finds the dictionary in the given list that has the least error compared to the reference dictionary.

        Args:
            reference_dict (dict): The reference dictionary to compare against.
            dict_list (list): A list of dictionaries to search through.

        Returns:
            dict: The dictionary with the least error compared to the reference dictionary.
        """
        least_error_dict = min(dict_list, key=lambda d: self.calculate_error(reference_dict, d))
        return least_error_dict

    def parse_node(self, element):
        """
        Parses a node element and returns the node ID and its attributes.

        Args:
            element (Element): The XML element representing a node.

        Returns:
            tuple: A tuple containing the node ID and a dictionary of its attributes.
                The dictionary contains the following keys:
                - "value": The value of the node.
                - "nodes_id": The ID of the connected nodes.
                - "occupied": The occupancy status of the node.
                - "joint_dependency": The joint dependency of the node.

        """
        node_id_cart = element.get("id")
        value = self.vectorise_string(element.get("value"))
        occupied = element.get("occupied")
        node_id = element.get("nodes_id")
        joint_dependency = ast.literal_eval(element.get("joint_dependency"))
        return node_id_cart, {"value": value, "nodes_id": node_id, "occupied": occupied, "joint_dependency": joint_dependency}

    def parse_edge(self, element):
        """
        Parses an XML element representing an edge in the robot graph.

        Args:
            element (xml.etree.ElementTree.Element): The XML element representing the edge.

        Returns:
            tuple: A tuple containing the source and target nodes of the edge, and a dictionary
                   containing the weight and occupied status of the edge.
        """
        source = element.get("source")
        target = element.get("target")
        weight = element.get("weight")
        occupied = element.get("occupied")
        return (source, target), {"weight": float(weight), "occupied": occupied}

    def read_graph_from_file(self, name, robot, babbled_nodes):
        """
        Reads a graph from a file and adds the nodes and edges to the graph.

        Args:
            name (str): The name of the graph.
            robot (str): The name of the robot.
            babbled_nodes (str): The number of babbled nodes.

        Returns:
            None
        """
        file_path = "./data/test_" + robot + "/graphs/joints_graphs_lib/" + name + "_" + babbled_nodes + ".xml"
        tree = ET.parse(file_path)
        root = tree.getroot()
        self.nodes_id = 0

        for node_element in root.findall(".//node"):
            node, attributes = self.parse_node(node_element)
            self.G.add_node(node, **attributes)
            self.nodes_id += 1

        for edge_element in root.findall(".//edge"):
            edge, attributes = self.parse_edge(edge_element)
            self.G.add_edge(*edge, **attributes)

    def read_object_from_file(self, name):
        """
        Reads a graph object from a file.

        Args:
            name (str): The name of the file (without extension) to read the graph from.

        Returns:
            networkx.Graph: The graph object read from the file.
        """
        return nx.read_graphml("./data/objects/" + name + ".net")

    def vectorise_string(self, vec):
        """
        Converts a string representation of a vector into a list of floats.

        Args:
            vec (str): The string representation of the vector.

        Returns:
            list: A list of floats representing the vector.
        """
        aux_data = "".join([i for i in vec if not (i=="[" or i=="]" or i==",")])
        data = np.array(aux_data.split())
        return list(data[0:3].astype(float))

    def dict_of_dep_to_dict_of_lists(self, dict):
        """
        Converts a dictionary of dependencies to a dictionary of lists.

        Args:
            dict (dict): The input dictionary of dependencies.

        Returns:
            dict: A dictionary where the keys are the unique keys from the input dictionary,
                  and the values are lists of corresponding values from the input dictionary.
        """
        result_dict = {}
        for entry in dict:
            for key, value in entry.items():
                if key not in result_dict:
                    result_dict[key] = []
                result_dict[key].append(value)
        return result_dict

    def plot_dict_of_dependencies(self, dict_old, dict_new):
        """
        Plots the dictionary of dependencies.

        This function takes two dictionaries, `dict_old` and `dict_new`, and plots the angles contained in each dictionary.
        The angles are plotted against the frames, with the old angles represented by circles and the new angles represented by crosses.

        Parameters:
        - dict_old (dict): A dictionary containing the old angles.
        - dict_new (dict): A dictionary containing the new angles.

        Returns:
        None
        """
        for key in dict_new.keys():
            plt.plot(np.radians(np.array(dict_old[key])), label=f'Old angles - {key}', marker='o')
            plt.plot(np.radians(np.array(dict_new[key])), label=f'New angles - {key}', marker='x')

        # Adding labels and legend
        plt.grid()
        plt.xlabel('Frames')
        plt.ylabel('Angle in Radians')
        plt.legend(loc='upper right')
        plt.title('Old vs new angles after Path Repair')
        plt.show()

    def plot_graph(self, robotName, nodes='', trajectory=[], candidates=[]):
        """
        Plot a 3D graph with trajectory, candidates, nodes, and edges.

        Parameters:
        - robotName (str): The name of the robot.
        - nodes (str): Optional. The nodes to be plotted.
        - trajectory (list): Optional. The trajectory points to be plotted.
        - candidates (list): Optional. The candidate points to be plotted.

        Returns:
        None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot trajectory only if it exists
        if len(trajectory):
            xdata = trajectory[:, 0]
            ydata = trajectory[:, 1]
            zdata = trajectory[:, 2]
            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap="Greens")
                # Plot edges between the points
            for i in range(len(xdata) - 1):
                ax.plot([xdata[i], xdata[i+1]], [ydata[i], ydata[i+1]], [zdata[i], zdata[i+1]], c="blue")
        if len(candidates):
            xdata = candidates[:, 0]
            ydata = candidates[:, 1]
            zdata = candidates[:, 2]
            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap="Reds")

           # Plot the nodes
        for node, xyz in zip(self.get_nodes(), self.get_nodes()):
            xyz_rounded = tuple(round(coord, 2) for coord in self.get_node_attr(xyz, "value"))
            ax.scatter(*xyz_rounded, s=100, ec="w", cmap="Greens")

        # Plot the edges
        for edge in self.get_edges():
            first_edge = self.vectorise_string(edge[0])
            second_edge = self.vectorise_string(edge[1])
            x = [first_edge[0], second_edge[0]]
            y = [first_edge[1], second_edge[1]]
            z = [first_edge[2], second_edge[2]]
            ax.plot(x, y, z, color="grey")

        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines on
            ax.grid(True)
            # Set axes labels
            ax.set_title(self.key)
            ax.set_xlabel("\n X [mm]", linespacing=3.2)
            ax.set_ylabel("\n Y [mm]", linespacing=3.2)
            ax.set_zlabel("\n Z [mm]", linespacing=3.2)

        _format_axes(ax)
        fig.tight_layout()
        # plt.savefig("data/test_"+ robotName + "/graphs/graphs_plots/" + self.key + "_" + robotName + "_" + nodes + ".pdf", format="pdf") #_object
        plt.show()

    def find_neighbors(self, node):
        """
        Find the neighbors of a given node in the graph.

        Parameters:
            node (list): The node for which to find neighbors.

        Returns:
            list: A list of neighbors of the given node.
        """
        return self.G.neighbors(self.list_to_string(node))

    def find_closest_point(self, new_point, kdtree):
        """
        Finds the closest point to the given new_point in the provided kdtree.

        Args:
            new_point (array-like): The coordinates of the new point.
            kdtree (scipy.spatial.KDTree): The KDTree containing the points.

        Returns:
            array-like: The coordinates of the closest point in the kdtree.
        """
        distance, index = kdtree.query(new_point)
        return kdtree.data[index]

    def find_trajectory_in_graph(self, tra):
        """
        Finds the trajectory in the graph by finding the closest points to each node in the given trajectory.

        Parameters:
        tra (list): The list of nodes representing the trajectory.
        Returns:
        list: The list of closest points in the graph corresponding to each node in the trajectory.
        """
        nodes = self.get_nodes_values()
        kdtree = KDTree(np.array(nodes))
        world_nodes = []
        for node in tra:
            closest_point = self.find_closest_point(node, kdtree)
            if world_nodes:
                if (world_nodes[-1] != closest_point).any():
                    world_nodes.append(closest_point)
            else:
                world_nodes.append(closest_point)
        return world_nodes

    def find_neighbors_candidates(self, nodes):
        """
        Finds the neighbor candidates for a given list of nodes.

        Args:
            nodes (list): A list of nodes for which to find neighbor candidates.

        Returns:
            list: A list of neighbor candidates for each node in the input list.
        """
        return [self.find_neighbors(node) for node in nodes]

    def find_trajectory_shared_nodes(self, tra):
        """
        Finds the shared nodes between a given trajectory and the nodes in the kdtree.

        Parameters:
            tra (list): The trajectory to find shared nodes with.

        Returns:
            list: A list of points from the trajectory that are not present in the kdtree.
        """
        kdtree = self.get_nodes()
        return [point for point in tra if self.list_to_string(point) not in kdtree]

    def find_subgraphs(self):
        """
        Finds and returns a list of subgraphs in the graph.

        Returns:
            list: A list of subgraphs in the graph.
        """
        sub_graphs = [self.G.subgraph(c) for c in nx.connected_components(self.G)]
        print(len(sub_graphs))

    def lerp(self, a, b, t):
        """
        Linearly interpolates between two dictionaries.

        Args:
            a (dict): The starting dictionary.
            b (dict): The ending dictionary.
            t (float): The interpolation parameter. Should be between 0 and 1.

        Returns:
            dict: The interpolated dictionary.

        """
        interpolated_dict = {key: round((1 - t) * a[key] + t * b[key], 2) for key in a}
        return interpolated_dict

    def string_to_dict(self, dict):
            """
            Converts a string representation of a dictionary to a dictionary object.

            Args:
                dict (str): The string representation of the dictionary.

            Returns:
                dict: The dictionary object.

            Raises:
                ValueError: If the string cannot be converted to a dictionary.

            """
            if type(dict) == str:
                try:
                    dict = re.sub(r"\'", '"', dict)
                    dict = json.loads(dict)
                except ValueError as e:
                    print(f"Error: {e}")
            return dict

    def calculate_average_angle(self, dict_list):
        """
        Calculate the average angle from a list of dictionaries.

        Args:
            dict_list (list[dict]): A list of dictionaries containing angle values.

        Returns:
            dict: A dictionary containing the average angle for each key in the input dictionaries.
        """
        if not dict_list:
            return {}
        array_of_dicts = np.array([list(d.values()) for d in dict_list])
        avg_array = np.mean(array_of_dicts, axis=0)
        avg_dict = dict(zip(dict_list[0].keys(), avg_array))
        return avg_dict

    def interpolate_joint_angles(self, joint_angles_a, joint_angles_b, t = 0.5):
        """
        Interpolates between two sets of joint angles.

        Args:
            joint_angles_a (list): The first set of joint angles.
            joint_angles_b (list): The second set of joint angles.
            t (float, optional): The interpolation parameter. Defaults to 0.5.

        Returns:
            list: The interpolated joint angles.
        """
        joint_angles_a = self.calculate_average_angle(joint_angles_a)
        joint_angles_b = self.calculate_average_angle(joint_angles_b)
        return self.lerp(joint_angles_a, joint_angles_b, t)

    def adding_candidates_to_graph(self, robot, length, candidates):
        """
        Adds candidates to the graph by finding their neighbors and connecting them with edges.

        Parameters:
        - robot: The robot object.
        - length: The length of the robot.
        - candidates: List of candidate points to be added to the graph.
        """
        nodes = self.get_nodes_values()
        tree = KDTree(np.array(nodes))
        max_dist = 2 * length
        for i, point in enumerate(candidates):
            neighbors_indices = tree.query_ball_point(point, max_dist)
            if len(neighbors_indices) > 1:
                distances, indices = tree.query(point, k = 1)
                initial_angles = self.get_node_attr(tree.data[indices], "joint_dependency")
                angles = []
                for i_a in initial_angles:
                    angles.append(self.gauss_newton_angle_estimation(robot, i_a, point))
                self.add_one_node(point, angles)
                for index in neighbors_indices:
                    node_link = tree.data[index]
                    self.add_one_edge([point, node_link])

    def verify_path_in_graph(self, path):
        """
        Verifies if a given path exists in the graph.

        Args:
            path (list): A list of nodes representing the path.

        Returns:
            list or None: A list of missing nodes if the path is invalid, None otherwise.
        """
        missing_nodes = []
        if self.has_node(path[0]) and self.has_node(path[-1]):
            if self.has_path(path[0], path[-1]):
                for point in path:
                    if not self.has_node(point) and point not in missing_nodes:
                        missing_nodes.append(point)
            else:
                print("NO PATH")
                missing_nodes = None
        else:
            print("NO INITIAL OR FINAL NODE")
            print(path[0], path[-1])
            missing_nodes = None
        return missing_nodes

    def find_node_dependencies_in_objects(self, node):
        """
        Finds the dependencies of a given node in the objects present in the world.

        Args:
            node (str): The name of the node.

        Returns:
            list: A list of dependencies of the given node.
        """
        dependencies = []
        for key, value in self.objects_in_world.items():
            if value.has_node(node):
                dependencies.extend(value.nodes[node]["joint_dependency"])
        return dependencies

    def find_blocked_angles(self, missing_nodes):
        """
        Finds the blocked angles for the given missing nodes.

        Parameters:
        - missing_nodes (list): A list of nodes that are missing.

        Returns:
        - blocked_angles (list): A list of blocked angles for the missing nodes.
        """
        blocked_angles = []
        for node in missing_nodes:
            blocked_angles.extend(self.find_node_dependencies_in_objects(node))
        return blocked_angles

    def find_closest_path_in_graph(self, missing_nodes, original_path):
        """
        Finds the closest path in the graph by removing the missing nodes from the original path.

        Args:
            missing_nodes (list): A list of nodes that are missing from the original path.
            original_path (list): The original path in the graph.

        Returns:
            list: The closest path in the graph after removing the missing nodes.
        """
        new_path = []
        i = 0
        print(len(original_path))
        while i < len(original_path):
            node = original_path[i]
            if node not in missing_nodes:
                new_path.append(node)
                prev_node = node
                print("NODE EXISTS ", i, node)
            else:
                next_node = None
                for j in range(i + 1, len(original_path)):
                    if original_path[j] not in missing_nodes:
                        next_node = original_path[j]
                        print("NEXT NODE", j, next_node)
                        break
                    else:
                        print("MISSING NODE", j, node)
                if next_node is not None and self.has_path(prev_node, next_node):
                    sub_path = self.shortest_path(prev_node, next_node)
                    sub_path.pop(0)
                    sub_path = [self.get_node_attr(i, "value") for i in sub_path]
                    aux_path = [item.tolist() if isinstance(item, np.ndarray) else item for item in sub_path]
                    new_path.extend(aux_path)
                    prev_node = new_path[-1]
                    i = j
                else:
                    break
                print("NEW PATH ", i, aux_path)
            i += 1
        return new_path

    def find_new_path(self, missing_nodes, original_path):
        """
        Finds a new path by avoiding the specified missing nodes in the original path.

        Args:
            missing_nodes (list): A list of nodes to be avoided in the new path.
            original_path (list): The original path to be modified.

        Returns:
            list: The new path that avoids the missing nodes.
        """
        new_path = []
        nodes = self.get_nodes_values()
        kdtree = KDTree(np.array(nodes))
        i = 0
        while i < len(original_path):
            node = original_path[i]
            if node not in missing_nodes:
                new_path.append(node)
                prev_node = node
            else:
                closest_point = self.find_closest_point(node, kdtree)
                if self.has_path(prev_node, closest_point):
                    sub_path = self.shortest_path(prev_node, closest_point)
                    sub_path.pop(0)
                    sub_path = [self.get_node_attr(i, "value") for i in sub_path]
                    aux_path = [item.tolist() if isinstance(item, np.ndarray) else item for item in sub_path]
                    new_path.extend(aux_path)
                    prev_node = new_path[-1]
            i += 1
        return new_path

    def re_path_end_effector(self, missing_nodes, original_path):
        """
        Re-calculate the path for the end effector of the robot.

        Args:
            missing_nodes (list): A list of nodes that are missing in the original path.
            original_path (list): The original path for the end effector.

        Returns:
            list or None: The recalculated path if missing_nodes is not empty, the original path if missing_nodes is empty,
                         or None if missing_nodes is None.
        """
        if missing_nodes == []:
            print("Path hasn't changed")
            return original_path
        elif missing_nodes is None:
            print("Impossible to execute the path")
            return None
        else:
            return self.find_new_path(missing_nodes, original_path)

    def gauss_newton_angle_estimation(self, robot, initial_angles, cartesian_point):
        """
        Performs Gauss-Newton optimization to estimate the optimal angles for a given robot configuration
        that minimize the Euclidean distance between the predicted and observed Cartesian positions.

        Args:
            robot (Robot): The robot instance.
            initial_angles (dict): A dictionary containing the initial angles for the optimization.
            cartesian_point (numpy.ndarray): The observed Cartesian position.

        Returns:
            dict: A dictionary containing the optimized angles.
        """

        def objective(angles, *args):
            cartesian_point = args[0]
            robot = args[1]
            # Retrieve the forward kinematics function based on the robot instance
            forward_kinematics = getattr(robot, 'forward_kinematics_' + robot.robotName)
            angle_list_len = len(robot.physical_limits_left) + len(robot.physical_limits_right) + len(robot.physical_limits_head)
            angles_zeros = np.zeros((angle_list_len))
            angles = np.deg2rad(np.array(angles))
            if 'Left' in self.key:
                insert_index = 0
            elif 'Right' in self.key:
                insert_index = len(robot.physical_limits_left)
            elif 'Head' in self.key:
                insert_index = len(robot.physical_limits_left) + len(robot.physical_limits_right)
            result_array = np.concatenate([angles_zeros[:insert_index], angles, angles_zeros[insert_index + len(angles):]])
            # Calculate the Euclidean distance between the predicted and observed Cartesian positions
            left_arm, right_arm, head = forward_kinematics(result_array)
            if 'Left' in self.key:
                observed_positions = left_arm[int(self.key[-1])]
            elif 'Right' in self.key:
                observed_positions = right_arm[int(self.key[-1])]
            elif 'Head' in self.key:
                observed_positions = head[int(self.key[-1])]
            distance = np.linalg.norm(observed_positions - cartesian_point)
            return distance

        # Perform optimization to minimize the objective function
        result = minimize(objective, list(initial_angles.values()), args=(cartesian_point, robot), method='L-BFGS-B')

        # The optimized angles
        optimized_angles = {f'angle_{i+1}': round(result.x[i], 2) for i in range(len(result.x))}
        return optimized_angles
