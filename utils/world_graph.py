import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
plt.rcParams.update({'font.size': 20})

class Graph(object):
    """
    Represents a graph data structure.

    This class provides methods to add nodes and edges to the graph, perform graph operations,
    and visualize the graph in a 3D plot for the world.

    Attributes:
        G (nx.Graph): The underlying NetworkX graph object.

    Methods:
        add_nodes: Add nodes to the graph.
        add_one_node: Adds a single node to the graph.
        list_to_string: Converts a list of numbers to a string representation.
        get_nodes_as_string: Returns a list of nodes in the graph as strings.
        add_one_edge: Adds a single edge to the graph.
        add_edges: Add edges to the graph.
        has_edge: Check if there is an edge between two nodes in the graph.
        read_object_from_file: Reads an object graph from a file and adds it to the main graph.
        save_object_to_file: Save the graph object to a file.
        save_graph_to_file: Save the graph to a file in GraphML format.
        read_graph_from_file: Reads a graph from a file and adds its nodes and edges to the graph object.
        search: Search for a path between the initial node and the goal node using a specified algorithm.
        get_nodes: Returns a list of nodes in the graph.
        get_edges: Returns the edges of the graph.
        plot_graph: Plots a 3D graph of the world with nodes, edges, and an optional trajectory.
    """

    def __init__(self):
        """
        Initializes a new instance of the WorldGraph class.
        """
        self.G = nx.Graph()

    def add_nodes(self, connections):
        """
        Add nodes to the graph.

        Parameters:
        connections (list): A list of connections to be added as nodes.

        Returns:
        None
        """
        for i in connections:
            node_label = tuple(i)
            self.G.add_node(node_label)

    def add_one_node(self, node):
        """
        Adds a single node to the graph.

        Parameters:
            node (tuple): The node to be added to the graph.

        Returns:
            None
        """
        self.G.add_node(tuple(node))

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

    def get_nodes_as_string(self):
        """
        Returns a list of nodes in the graph as strings.

        Returns:
            list: A list of nodes in the graph represented as strings.
        """
        nodes = self.G.nodes()
        return [self.list_to_string(node) for node in nodes]

    def add_one_edge(self, edge):
        """
        Adds a single edge to the graph.

        Parameters:
        - edge: A tuple representing the edge to be added. It should have the format (u, v),
            where u and v are nodes in the graph.

        Returns:
        None
        """
        u, v = tuple(edge[0]), tuple(edge[1])
        self.G.add_edge(u, v)

    def add_edges(self, edges):
        """
        Add edges to the graph.

        Parameters:
        - edges (list): A list of edges to be added to the graph.

        Returns:
        None
        """
        for i in tqdm(edges):
            u, v = tuple(i[0]), tuple(i[1])
            self.G.add_edge(u, v)

    def has_edge(self, u, v):
        """
        Check if there is an edge between two nodes in the graph.

        Parameters:
        - u: The source node.
        - v: The target node.

        Returns:
        - True if there is an edge between u and v, False otherwise.
        """
        u, v = tuple(u), tuple(v)
        return self.G.has_edge(u, v)

    def read_object_from_file(self, robotName, name):
        """
        Reads an object graph from a file and adds it to the main graph.

        Args:
            robotName (str): The name of the robot.
            name (str): The name of the object.

        Returns:
            None
        """
        aux_graph = nx.Graph()
        aux_graph = nx.read_graphml("./data/test_" + robotName + "/objects/" + name + ".net")

        for node in tqdm(aux_graph.nodes()):
            self.G.add_node(ast.literal_eval(node))
        for edges in tqdm(aux_graph.edges()):
            self.G.add_edge(ast.literal_eval(edges[0]), ast.literal_eval(edges[1]))

    def save_object_to_file(self, robotName, name):
        """
        Save the graph object to a file.

        Args:
            robotName (str): The name of the robot.
            name (str): The name of the object.

        Returns:
            None
        """
        nx.write_graphml_lxml(self.G, "./data/test_" +  robotName + "/objects/" + name + ".net")

    def save_graph_to_file(self, robotName, name):
        """
        Save the graph to a file in GraphML format.

        Args:
            robotName (str): The name of the robot.
            name (str): The name of the file.

        Returns:
            None
        """
        nx.write_graphml_lxml(self.G, "./data/test_" + robotName + "/graphs/" + name + ".net")

    def read_graph_from_file(self, robotName, name):
        """
        Reads a graph from a file and adds its nodes and edges to the graph object.

        Parameters:
        - robotName (str): The name of the robot.
        - name (str): The name of the graph file.

        Returns:
        None
        """
        aux_graph = nx.Graph()
        aux_graph = nx.read_graphml("./data/test_" + robotName + "/graphs/" + name + ".net")

        for node in tqdm(aux_graph.nodes()):
            self.G.add_node(ast.literal_eval(node))
        for edges in tqdm(aux_graph.edges()):
            self.G.add_edge(ast.literal_eval(edges[0]), ast.literal_eval(edges[1]))

    def search(self, initial_node, goal_node, name):
        """
        Search for a path between the initial node and the goal node using a specified algorithm.

        Args:
            initial_node: The initial node of the search.
            goal_node: The goal node of the search.
            name: The name of the search algorithm to use.

        Returns:
            The path between the initial node and the goal node, if found.

        Raises:
            NetworkXNoPath: If no path exists between the initial node and the goal node.
        """
        if name == "A*":
            path = nx.astar_path(self.G, initial_node, goal_node)
        return path

    def get_nodes(self):
        """
        Returns a list of nodes in the graph.

        Returns:
            list: A list of nodes in the graph.
        """
        return self.G.nodes()

    def get_edges(self):
        """
        Returns the edges of the graph.

        Returns:
            A list of edges in the graph.
        """
        return self.G.edges()

    def plot_graph(self, trajectory=[]):
        """
        Plots a 3D graph of the world with nodes, edges, and an optional trajectory.

        Parameters:
            trajectory (list, optional): A list of 3D coordinates representing the trajectory. 
                                        Each coordinate should be a list or array-like object with 3 elements.
                                        Defaults to an empty list.

        Returns:
            None
        """
        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot trajectory only if it exists
        if len(trajectory):
            xdata = trajectory[:, 0]
            ydata = trajectory[:, 1]
            zdata = trajectory[:, 2]
            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

        # Plot the nodes
        for node, xyz in zip(self.get_nodes(), self.get_nodes()):
            xyz_rounded = tuple(round(coord, 2) for coord in xyz)
            ax.scatter(*xyz_rounded, s=100, ec="w")

        # Plot the edges
        for edge in self.get_edges():
            x = [edge[0][0], edge[1][0]]
            y = [edge[0][1], edge[1][1]]
            z = [edge[0][2], edge[1][2]]
            ax.plot(x, y, z, color='grey')

        def _format_axes(ax):
            """Visualization options for the 3D axes"""
            # Turn gridlines on
            ax.grid(True)
            # Set axes labels
            ax.set_xlabel("\n X [m]", linespacing=3.2)
            ax.set_ylabel("\n Y [m]", linespacing=3.2)
            ax.set_zlabel("\n Z [m]", linespacing=3.2)

        _format_axes(ax)
        fig.tight_layout()
        # plt.savefig("world_small.pdf", format="pdf")
        plt.show()
