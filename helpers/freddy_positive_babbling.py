import numpy as np
import json

def read_babbling(path_name):
    points = readTxtFile("../data/" + path_name)
    joint_angles = {"left" : [], "right" : []}
    keys = list(points)
    for key in keys:
        for i in points[key]:
            joint_angles[key].append(linear_angular_mapping_gen3(i))
    return joint_angles

def readTxtFile(file_path):
    with open(file_path) as f:
        contents = f.read()
    return json.loads(contents)

def linear_angular_mapping_gen3(vec):
    new_angle_vec = np.zeros_like(vec)
    for i in range(len(vec)):
        if vec[i] >=  0:
            new_angle_vec[i] = vec[i]
        else:
            new_angle_vec[i] = 360 + vec[i]
    return list(new_angle_vec)

if __name__ == "__main__":
    print("This is a dummy file")
    amount_of_points = 150
    babbing_path = "test_gen3/self_exploration/self_exploration_gen3_" + str(amount_of_points) + "_old" + ".txt"

    list_of_angles = read_babbling(babbing_path)
    # file_path = "../data/test_gen3/self_exploration/self_exploration_gen3_" + str(amount_of_points) + ".txt"
    print(list_of_angles)
    # with open(file_path, 'w') as file:
    #     json.dump(list_of_angles, file)


