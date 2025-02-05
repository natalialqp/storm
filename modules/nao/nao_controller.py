# -*- encoding: UTF-8 -*-

import sys
import time
from naoqi import ALProxy
import almath
import numpy as np
import json
import random
import robot
import motion
import pandas as pd
import yaml
from math import degrees

def read_yaml_file(file_path):
    """
    Reads a YAML file and returns the data as a dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The data read from the YAML file.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

class NaoManager(object):
    def __init__(self, robot_name, robotIp):
        """
        Initializes the NaoController object.

        Args:
            robot_name (str): The name of the robot.

        Attributes:
            motor_babbling_recording (dict): A dictionary to store motor babbling recordings for each limb.
            current_ang_pos_Larm (numpy.ndarray): An array to store the current angular positions of the left arm.
            current_ang_pos_Rarm (numpy.ndarray): An array to store the current angular positions of the right arm.
            current_ang_pos_head (numpy.ndarray): An array to store the current angular positions of the head.
            key (str): A key for some functionality.
            robot_name (str): The name of the robot.
            left_columns (list): A list of column names for the left arm angles.
            right_columns (list): A list of column names for the right arm angles.
            head_columns (list): A list of column names for the head angles.
            left_angle_sequence (list): A sequence of angles for the left arm.
            right_angle_sequence (list): A sequence of angles for the right arm.
            head_angle_sequence (list): A sequence of angles for the head.
        """
        self.motor_babbling_recording = {'left': [], 'right': [],'head': []}
        self.current_ang_pos_Larm = np.zeros((5))
        self.current_ang_pos_Rarm = np.zeros((5))
        self.current_ang_pos_head = np.zeros((2))
        self.key = ''
        self.robot_name = robot_name
        self.left_columns = ['left_1', 'left_2', 'left_3', 'left_4', 'left_5']
        self.right_columns = ['right_1', 'right_2', 'right_3', 'right_4', 'right_5']
        self.head_columns = ['head_1', 'head_2']
        self.left_angle_sequence = []
        self.right_angle_sequence = []
        self.head_angle_sequence = []
        self.motionProxy = ALProxy("ALMotion", robotIp, 9559)

    def import_robot(self, file_path):
        """
        Imports a robot from the specified file path.

        Args:
            file_path (str): The path to the file containing the robot data.

        Returns:
            None
        """
        self.rob = robot.Robot(self.robot_name)
        self.rob.import_robot(file_path)

    def random_angles(self, delta_angle):
        """
        Generate random angles within the specified delta angle range.

        Args:
            delta_angle (int): The range of angle variation.

        Returns:
            numpy.ndarray: An array of randomly generated angles.
        """
        zeros = []
        if "head" in self.key:
            limits = self.rob.physical_limits_head
            for key in limits:
                zeros.append(random.randrange(limits[key][0], limits[key][1] + delta_angle, delta_angle))

        elif "left" in self.key:
            limits = self.rob.physical_limits_left
            for key in limits:
                zeros.append(random.randrange(limits[key][0], limits[key][1] + delta_angle, delta_angle))

        elif "right" in self.key:
            limits = self.rob.physical_limits_right
            for key in limits:
                zeros.append(random.randrange(limits[key][0], limits[key][1] + delta_angle, delta_angle))
        return np.array(zeros)

    def motor_babbling(self, robotIP, delta_angle, sequence_len):
            """
            Perform motor babbling on the NAO robot.

            Args:
                robotIP (str): The IP address of the NAO robot.
                delta_angle (float): The maximum change in angle for random movements.
                sequence_len (int): The desired length of the motor babbling sequence.

            Returns:
                None
            """
            try:
                self.motionProxy = ALProxy("ALMotion", self.robotIP, self.PORT)
            except Exception as e:
                print("Could not create proxy to ALMotion")
                print ("Error was: ", e)
                sys.exit(1)
            for key in self.motor_babbling_recording:
                self.key = key
                start = time.time()
                while len(self.motor_babbling_recording[key]) < sequence_len:
                    print(len(self.motor_babbling_recording[key]))
                    random_move = self.random_angles(delta_angle)
                    if "head" in key:
                        new_ang_pos_head = random_move
                        added = self.add_config(new_ang_pos_head)
                        if added:
                            self.current_ang_pos_head = new_ang_pos_head.copy()
                    elif "left" in key:
                        new_ang_pos_arm = random_move
                        added = self.add_config(new_ang_pos_arm)
                        if added:
                            self.current_ang_pos_Larm = new_ang_pos_arm.copy()
                    elif "right" in key:
                        new_ang_pos_arm = random_move
                        added = self.add_config(new_ang_pos_arm)
                        if added:
                            self.current_ang_pos_Rarm = new_ang_pos_arm.copy()
                    self.joint_angle_publisher()
                end = time.time()
                elapsed = (end - start) / 60
                print("Time taken ", self.key, ": ", elapsed, "minutes")

    def add_config(self, candidate_pos):
        """
        Adds a candidate position to the motor babbling recording if it is within the limits and not already present.

        Args:
            candidate_pos (list): The candidate position to be added.

        Returns:
            bool: True if the candidate position was successfully added, False otherwise.
        """
        added = False
        candidate_pos = list(np.round(candidate_pos, decimals=2))
        inside_limits = self.check_limits(candidate_pos)
        if inside_limits:
            if candidate_pos not in self.motor_babbling_recording[self.key]:
                self.motor_babbling_recording[self.key].append(candidate_pos)
                added = True
        return added

    def check_limits(self, candidate_pos):
        """
        Check if the candidate position is within the physical limits of the robot.

        Args:
            candidate_pos (list): The candidate position to be checked.

        Returns:
            bool: True if the candidate position is within the limits, False otherwise.
        """
        inside_limits = True
        if self.key == "head":
            limits = self.rob.physical_limits_head
        elif self.key == "left":
            limits = self.rob.physical_limits_left
        elif self.key == "right":
            limits = self.rob.physical_limits_right
        for count, i in enumerate(limits):
            if candidate_pos[count] < limits[i][0] or candidate_pos[count] > limits[i][1]:
                inside_limits = False
        return inside_limits

    def joint_angle_publisher(self):
        """
        Publishes joint angles to control the NAO robot's movements.

        This function sets the joint angles for the NAO robot's head, arms, and legs,
        and then uses the motionProxy to move the robot accordingly.

        Args:
            None

        Returns:
            None
        """
        LeftLeg  = [0, 0, -25, 40, -20, 0]
        RightLeg = [0, 0, -25, 40, -20, 0]
        Head = list(self.current_ang_pos_head)
        LeftArm = list(self.current_ang_pos_Larm)
        LeftArm.append(0)
        RightArm = list(self.current_ang_pos_Rarm)
        RightArm.append(0)
        name = "Body"
        self.motionProxy.setStiffnesses(name, 1.0)
        TargetAngles = Head + LeftArm + LeftLeg + RightLeg + RightArm
        TargetAngles = [ x * motion.TO_RAD for x in TargetAngles]
        MaxSpeedFraction = 0.2
        self.motionProxy.angleInterpolationWithSpeed(name, TargetAngles, MaxSpeedFraction)
        time.sleep(0.3)

    def read_action_from_file(self, file_path):
        """
        Reads action data from a CSV file and appends the angle sequences to the respective lists.

        Args:
            file_path (str): The path to the CSV file containing the action data.

        Returns:
            None
        """
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            left_values = [degrees(row[column]) for column in self.left_columns]
            right_values = [degrees(row[column]) for column in self.right_columns]
            head_values = [degrees(row[column]) for column in self.head_columns]
            self.left_angle_sequence.append(left_values)
            self.right_angle_sequence.append(right_values)
            self.head_angle_sequence.append(head_values)

if __name__ == "__main__":
    config = read_yaml_file("config.yaml")
    robotName = config["robot-name"]
    babblingPoints = config["babbling-points"]
    delta = config["minimum-distance"]
    robotIp = config["nao-ip"]
    functionName = config["function-name"]
    action = config["action-name"]

    if len(sys.argv) <= 1:
        print ("Usage python almotion_angleinterpolationreactif.py robotIP (optional default: 127.0.0.1)")
    else:
        robotIp = sys.argv[1]

    file_path = robotName + ".yaml"
    if functionName == "reproduce-action":
        dir = "./data/test_" + robotName + "/GMM_learned_actions/for_execution/"
        file_name = dir + "GMR_" + str(babblingPoints) + "_" + action + ".csv"
        action_reproduction = NaoManager(robotName, robotIp)
        action_reproduction.read_action_from_file(file_name)
        for i in range(len(action_reproduction.left_angle_sequence)):
            action_reproduction.current_ang_pos_Larm = action_reproduction.left_angle_sequence[i]
            action_reproduction.current_ang_pos_Rarm = action_reproduction.right_angle_sequence[i]
            action_reproduction.current_ang_pos_head = action_reproduction.head_angle_sequence[i]
            action_reproduction.joint_angle_publisher()

    elif functionName == "motor-babbling":
        self_explorator = NaoManager(robotName, robotIp)
        self_explorator.import_robot(file_path)
        self_explorator.motor_babbling(robotIp, delta, babblingPoints)
        file_path = "self_exploration_" + robotName + "_" + str(babblingPoints) + ".txt"
        with open(file_path, 'w') as file:
            json.dump(str(self_explorator.motor_babbling_recording), file)
