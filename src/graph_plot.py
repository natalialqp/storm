from cmath import sqrt
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import world_graph
import robot_graph
import itertools
from scipy.spatial import KDTree
import robot
import json
import simulate_position
from tqdm import tqdm
from scipy.spatial import cKDTree

def calculateEdgeLengths(connections):
    edge_lengths = np.zeros((len(connections),))
    for i in range(len(connections)):
        edge_lengths[i] = calculateEdgeLength(connections[i])
    return edge_lengths

def calculateEdgeLength(vertices_pair):
    return np.linalg.norm(vertices_pair[0] - vertices_pair[1])

def calculateMissingNode(vertice_1, vertice_2, vertice_3, distance):
    centroid = (vertice_1 + vertice_2 + vertice_3) / 3.0
    cross_product = np.cross((vertice_1 - vertice_2), (vertice_3 - vertice_2))
    cross_product /= np.linalg.norm(cross_product)
    fourth_point_1 = cross_product * distance + centroid
    fourth_point_2 = -cross_product * distance + centroid
    return fourth_point_1, fourth_point_2

def isInsideWorld(lowEdge, highEdge, point):
    world = np.column_stack((lowEdge, highEdge))
    world = [np.sort(i) for i in world]
    isInside = [True for i in range(len(point)) if world[i][0] <= point[i] <= world[i][1]]
    if len(isInside) == 3:
        return True
    else:
        return False

def findEdges(graph, length):
    edges = []
    max_dist = np.sqrt(3) * length
    combinations = itertools.combinations(graph.get_nodes(), 2)
    for i in combinations:
        if np.linalg.norm(np.array(i[0]) - np.array(i[1])) <= max_dist:
            edges.append(i)
    return edges

def findEdgesOptimized(graph, length):
    nodes = graph.get_nodes()
    points = np.array(nodes)  # Convert nodes to a NumPy array
    max_dist = np.sqrt(3) * length
    tree = cKDTree(points)
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

def createWorldCubes(lowEdge, highEdge, length):
    graph_world = world_graph.Graph()
    world = np.column_stack((lowEdge, highEdge))
    world = [np.sort(i) for i in world]
    for x in np.arange(world[0][0], world[0][1], length):
        for y in np.arange(world[1][0], world[1][1], length):
            for z in np.arange(world[2][0], world[2][1], length):
                node = np.around([x, y, z], decimals = 3)  # previously without list
                graph_world.add_one_node(node)
    edges = findEdgesOptimized(graph_world, length)
    graph_world.add_edges(edges)
    return graph_world

def find_closest_point(new_point, kdtree):
    """
    Find the closest point from the KD-tree to the new point.

    Parameters:
        new_point (numpy.ndarray): The new 3D point represented as a NumPy array of shape (3,).
        kdtree (scipy.spatial.KDTree): The KD-tree built from the set of random points.

    Returns:
        numpy.ndarray: The closest point from the set to the new point.
    """
    distance, index = kdtree.query(new_point)
    return kdtree.data[index]

def findRobotWorld(demonstration, length, world):
    kdtree = KDTree(world.get_nodes())
    robotWorld = world_graph.Graph()
    for i in demonstration:
        node = find_closest_point(i, kdtree)
        robotWorld.add_one_node(node)
    edges = findEdgesOptimized(robotWorld, length)
    robotWorld.add_edges(edges)
    return robotWorld

def findJointsGraph(points, length, world, joint_dic, joint_angles):
    kdtree = KDTree(world.get_nodes())
    for key in points:
        node = find_closest_point(points[key], kdtree)
        dependencies = slice_dict(joint_angles, key.split('_'))
        att = joint_dic[key].set_attribute(node, dependencies)
        joint_dic[key].add_one_node(node, att)
        edges = findEdges(joint_dic[key], length)
        joint_dic[key].add_edges(edges)
    return joint_dic

def slice_dict(dict, details):
    sub_list = []
    for i in dict:
        if details[0] in i and len(sub_list) <= int(details[1]):
            sub_list.append(dict[i])
    return sub_list

def createRobotGraphs(robot):
    joint_dic = {}
    for i in range(len(robot.leftArmAngles)):
        joint_dic["jointLeft_" + str(i)] = robot_graph.Graph(i)

    for i in range(len(robot.rightArmAngles)):
        joint_dic["jointRight_" + str(i)] = robot_graph.Graph(i)

    for i in range(len(robot.headAngles)):
        joint_dic["jointHead_" + str(i)] = robot_graph.Graph(i)
    return joint_dic

def pathPlanning(demonstration, robotWorld):
    kdtree = KDTree(robotWorld.get_nodes())
    path = []
    demonstration = list(demonstration)
    prev_node = tuple(find_closest_point(demonstration.pop(0), kdtree))
    path.append(prev_node)
    for i in demonstration:
        node = tuple(find_closest_point(i, kdtree))
        sub_path = robotWorld.search(prev_node, node, "A*")
        sub_path.pop(0)
        path.extend(sub_path)
        prev_node = node
    return path

def createWorldTetrahedrons(lowEdge, highEdge, length, height):
    graph_world = world_graph.Graph()
    vertices, connections = calculateTetrahedron(lowEdge, length)
    graph_world.add_nodes(vertices)
    graph_world.add_edges(connections)
    combinations = vertices
    i = 0
    while i < 10:
        combinations.append(list(itertools.combinations(graph_world.nodes, 3)))
    graph_world.plot_graph()

def calculateTetrahedron(point, length):
    # Calculate the remaining three vertices based on the given point and length
    vertices = np.zeros((4, 3))
    connections = np.zeros((6, 2, 3))
    vertices[0] = point
    vertices[1] = point + np.array([length, 0, 0])
    vertices[2] = point + np.array([length/2, length * np.sqrt(3) / 2, 0])
    vertices[3] = point + np.array([length/2, length * np.sqrt(3) / 6, length * np.sqrt(6) / 3])
    connections[0] = (vertices[0], vertices[1])
    connections[1] = (vertices[1], vertices[2])
    connections[2] = (vertices[2], vertices[0])
    connections[3] = (vertices[0], vertices[3])
    connections[4] = (vertices[1], vertices[3])
    connections[5] = (vertices[2], vertices[3])
    return vertices, connections

def simulateMapping():
    zdata = 5 * np.random.random(100)
    xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
    ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
    return np.column_stack((xdata, ydata, zdata))

def simulateTrajectory():
    zdata = 5 * np.random.random(10)
    xdata = np.sin(zdata) + 0.1 * np.random.randn(10)
    ydata = np.cos(zdata) + 0.1 * np.random.randn(10)
    return np.column_stack((xdata, ydata, zdata))

def learnTrajectory(demonstration, robotWorld):
    kdtree = KDTree(robotWorld.get_nodes())
    trajectory = []
    for i in demonstration:
        node = find_closest_point(i, kdtree)
        trajectory.append(node)
    return trajectory

def plotPath(demonstration, path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xdem = demonstration[:, 0]
    ydem = demonstration[:, 1]
    zdem = demonstration[:, 2]
    xpath = path[:, 0]
    ypath = path[:, 1]
    zpath = path[:, 2]
    ax.scatter3D(xdem, ydem, zdem, c='green')
    ax.scatter3D(xpath, ypath, zpath, c='red')
    plt.show()

def printRobotGraphs(robot):
    for key in robot:
        print(key, ": ")
        robot[key].print_graph()
        print()

def readTxtFile(file_path):
    with open(file_path) as f:
        contents = f.read()
    return json.loads(contents)

def divideFrames(data):
    sub_frame = []
    for key in data:
        frame = [np.array([el]) for el in data[key]]
        sub_frame.append(frame)
        new_data = [np.array([0]) for el in data[key]]
    new_data = np.array(new_data)
    for frame in sub_frame:
        new_data = np.hstack((new_data, np.array(frame)))
    return new_data[:,1:]

def forward_kinematics_n_frames(robot, joint_angles):
    left = []
    right = []
    head = []
    for frame in joint_angles:
        frame = np.array(frame)
        pos_left, pos_right, pos_head = robot.forward_kinematics(frame)
        left.append(pos_left)
        right.append(pos_right)
        head.append(pos_head)
    return left, right, head

if __name__ == "__main__":

    # Define parameter of the world and create one with cubic structure
    length = 0.02
    height = np.sqrt(2/3) * length
    lowEdge = np.array([-1, -1, 0])
    highEdge = np.array([1, 1, 1])
    # graph_world = createWorldCubes(lowEdge, highEdge, length)
    # graph_world.save_graph_to_file("test")
    graph_world = world_graph.Graph()
    graph_world.read_graph_from_file("test")
    # graph_world.plot_graph()

    #Read robot configuration from the .yaml filerobot_graphs
    file_path = "./robot_configuration_files/qt.yaml"
    qt = robot.Robot()
    qt.import_robot(file_path)
    robot_graphs = createRobotGraphs(qt)
    frames = readTxtFile("./data/angles.txt")
    joint_angles = divideFrames(frames)

    pos_left, pos_right, pos_head = forward_kinematics_n_frames(qt, joint_angles)
    s = simulate_position.RobotSimulation(pos_left, pos_right, pos_head)
    s.animate()

    points = qt.pos_mat_to_robot_mat_dict(pos_left, pos_right, pos_head)
    joint_angles_dict = qt.angular_mat_to_mat_dict(joint_angles)
    for i in range(len(joint_angles_dict)):
        joint_dict = findJointsGraph(points[i], length, graph_world, robot_graphs, joint_angles_dict[i])

    printRobotGraphs(robot_graphs)

    #Simulate trajectory and store nodes in robot graph
    # robot_world.plot_graph(np.asarray(path))
    # robot_world.plot_graph(generated_trajectory)
    # plotPath(generated_trajectory, np.asarray(path))