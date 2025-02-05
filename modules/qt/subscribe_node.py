#!/usr/bin/env python
import sys
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import json
import os
import random
import time

import utils.robot as robot
from main import read_yaml_file

class SelfExploration(object):
    """
    Class for performing self-exploration of a robot.
    This class provides methods for importing a robot, performing motor babbling, and checking torque values of the robot's joints.

    Attributes:
        motor_babbling_recording (dict): A dictionary to store motor babbling recordings for each arm and head.
        right_pub (rospy.Publisher): A publisher for the right arm position command.
        left_pub (rospy.Publisher): A publisher for the left arm position command.
        head_pub (rospy.Publisher): A publisher for the head position command.
        current_ang_pos_arm (numpy.ndarray): An array to store the current angular positions of the arm.
        current_ang_pos_head (numpy.ndarray): An array to store the current angular positions of the head.
        current_tor_head (numpy.ndarray): An array to store the current torques of the head.
        current_tor_arm (numpy.ndarray): An array to store the current torques of the arm.
        key (str): A key for some functionality.
        robot_name (str): The name of the robot.
    """

    def __init__(self, robot_name):
        """
        Initializes the SubscribeNode class.

        Args:
            robot_name (str): The name of the robot.

        Attributes:
            motor_babbling_recording (dict): A dictionary to store motor babbling recordings for each arm and head.
            right_pub (rospy.Publisher): A publisher for the right arm position command.
            left_pub (rospy.Publisher): A publisher for the left arm position command.
            head_pub (rospy.Publisher): A publisher for the head position command.
            current_ang_pos_arm (numpy.ndarray): An array to store the current angular positions of the arm.
            current_ang_pos_head (numpy.ndarray): An array to store the current angular positions of the head.
            current_tor_head (numpy.ndarray): An array to store the current torques of the head.
            current_tor_arm (numpy.ndarray): An array to store the current torques of the arm.
            key (str): A key for some functionality.
            robot_name (str): The name of the robot.
        """
        self.motor_babbling_recording = {'left': [], 'right': [], 'head': []}
        self.right_pub = rospy.Publisher('/qt_robot/right_arm_position/command', Float64MultiArray, queue_size=10)
        self.left_pub = rospy.Publisher('/qt_robot/left_arm_position/command', Float64MultiArray, queue_size=10)
        self.head_pub = rospy.Publisher('/qt_robot/head_position/command', Float64MultiArray, queue_size=10)
        self.current_ang_pos_arm = np.zeros((3))
        self.current_ang_pos_head = np.zeros((2))
        self.current_tor_head = np.zeros((2))
        self.current_tor_arm = np.zeros((3))
        self.key = ''
        self.robot_name = robot_name

    def import_robot(self, file_path):
        """
        Imports a robot from a file.

        Args:
            file_path (str): The path to the file containing the robot data.

        Returns:
            None
        """
        self.rob = robot.Robot(self.robot_name)
        self.rob.import_robot(file_path)

    def motor_babbling(self, delta_angle, sequence_len):
        """
        Perform motor babbling by randomly moving the robot's joints to collect joint angle data.

        Parameters:
        delta_angle (float): The maximum angle by which each joint can be randomly moved.
        sequence_len (int): The desired length of the joint angle sequence to be collected.

        Returns:
        None
        """
        self.joint_angle_publisher()
        waiting = 0
        for key in self.motor_babbling_recording:
            self.key = key
            start = time.perf_counter()
            while len(self.motor_babbling_recording[key]) < sequence_len:
                print(len(self.motor_babbling_recording[key]))
                self.joint_state_sub = rospy.Subscriber('/qt_robot/joints/state', JointState, self.qt_joint_state_cb)
                while self.current_ang_pos_arm[2] == 0:
                    waiting += 1
                # random_move = self.random_distribution(delta_angle)
                random_move = self.random_angles(delta_angle)
                if "head" in key:
                    # new_ang_pos_head = np.array(self.current_ang_pos_head) + random_move
                    new_ang_pos_head = random_move
                    added = self.add_config(new_ang_pos_head)
                    if added:
                        self.current_ang_pos_head = new_ang_pos_head.copy()
                else:
                    # new_ang_pos_arm = np.array(self.current_ang_pos_arm) + random_move
                    new_ang_pos_arm = random_move
                    added = self.add_config(new_ang_pos_arm)
                    if added:
                        self.current_ang_pos_arm = new_ang_pos_arm.copy()
                self.joint_angle_publisher()
            end = time.perf_counter()
            elapsed = (end - start) / 60
            print(f'Time taken {key}: {elapsed:.4f} minutes')

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
            bool: True if the candidate position is within the physical limits, False otherwise.
        """
        inside_limits = True
        if self.key == "head":
            limits = self.rob.physical_limits_head
        else:
            limits = self.rob.physical_limits_left
        for count, i in enumerate(limits):
            if candidate_pos[count] < limits[i][0] or candidate_pos[count] > limits[i][1]:
                inside_limits = False
        return inside_limits

    def qt_joint_state_cb(self, joint_state_msg):
        '''
        This function saves the recorded_positions taken from the robot's sensors.

        Args:
            joint_state_msg: The joint state message containing the position and effort values.
        '''
        if 'head' in self.key:
            self.current_ang_pos_head = [joint_state_msg.position[0], joint_state_msg.position[1]]
            self.current_tor_head = [joint_state_msg.effort[0], joint_state_msg.effort[1]]
            # rospy.loginfo(self.current_ang_pos_head)
        elif 'left' in self.key:
            self.current_ang_pos_arm = [joint_state_msg.position[3], joint_state_msg.position[4], joint_state_msg.position[2]]
            self.current_tor_arm = [joint_state_msg.effort[3], joint_state_msg.effort[4], joint_state_msg.effort[2]]
            # rospy.loginfo(self.current_ang_pos_arm)
        elif 'right' in self.key:
            self.current_ang_pos_arm = [joint_state_msg.position[6], joint_state_msg.position[7], joint_state_msg.position[5]]
            self.current_tor_arm = [joint_state_msg.effort[6], joint_state_msg.effort[7], joint_state_msg.effort[5]]
            # rospy.loginfo(self.current_ang_pos_arm)

    def check_torque(self):
        """
        Checks the torque values for different joints based on the current key.

        Returns:
            collision (bool): True if there is a collision, False otherwise.
            collision_joints (list): List of joints that are in collision.
        """
        #torque for HeadPitch (+) torque after force applied
        #torque for HeadYaw (+/-) in different directions -> stable is 0
        collision = False
        collision_joints = []
        if 'head' in self.key:
            if self.current_tor_head[0] > 0:
                collision = True
                collision_joints.append("HeadPitch")

            if self.current_tor_head[1] > 30 or self.current_tor_head[1] < -30:
                collision = True
                collision_joints.append("HeadYaw")
        # LeftShoulderPitch (+) torque after force applied to front, (-) torque after force applied to back
        # LeftShoulderRoll (-) torque after force applied
        # LeftElbowRoll (-) torque after force applied
        elif 'left' in self.key:
            if self.current_tor_arm[0] < -30:
                collision = True
                collision_joints.append("LeftShoulderPitch+")

            elif self.current_tor_arm[0] > 30:
                collision = True
                collision_joints.append("LeftShoulderPitch-") #crashes with belly, need to reduce angle

            if self.current_tor_arm[1] < -20:
                collision = True
                collision_joints.append("LeftShoulderRoll")

            if self.current_tor_arm[2] < -20:
                collision = True
                collision_joints.append("LeftElbowRoll")
        # RightShoulderPitch (-) torque after force applied to front, (+) torque after force applied to back
        # RightShoulderRoll (-) torque after force applied
        # RightElbowRoll (-) torque after force applied
        elif 'right' in self.key:
            if self.current_tor_arm[0] > 30:
                collision = True
                collision_joints.append("RightShoulderPitch-")

            elif self.current_tor_arm[0] < -30:
                collision = True
                collision_joints.append("RightShoulderPitch+") #crashes with belly, need to increase angle

            if self.current_tor_arm[1] < -20:
                collision = True
                collision_joints.append("RightShoulderRoll")

            if self.current_tor_arm[2] < -20:
                collision = True
                collision_joints.append("RightElbowRoll")
        return collision, collision_joints

    def random_distribution(self, delta_angle):
        """
        Generates a random distribution of angles based on the current key.

        Args:
            delta_angle (float): The angle increment.

        Returns:
            numpy.ndarray: An array of angles representing the random distribution.

        """
        collision, joint_list = self.check_torque()
        print("COLLISION: ", collision, joint_list)
        sign = random.choice([-1, 1])

        if "head" in self.key:
            rand = np.random.randint(0, 2)
            zeros = [0] * 2
            zeros[rand] = sign * delta_angle
            if collision and "Pitch" in joint_list:
                zeros = -np.abs(zeros)

        elif "left" in self.key or "right" in self.key:
            zeros = [0] * 3
            rand = np.random.randint(0, 3)
            zeros[rand] = sign * delta_angle
            if collision:
                if "Roll" in joint_list:
                    rand = np.random.randint(1, 3)
                    zeros[rand] = sign * delta_angle
                    zeros = np.abs(zeros)
                elif "Pitch+" in joint_list:
                    zeros = np.abs(zeros)
                elif "Pitch-" in joint_list:
                    zeros = -np.abs(zeros)
        return np.array(zeros)

    def random_angles(self, delta_angle):
        """
        Generates random angles within specified limits for the robot joints.

        Args:
            delta_angle (int): The increment value for generating random angles.

        Returns:
            numpy.ndarray: An array of random angles for the robot joints.
        """
        collision, joint_list = self.check_torque()
        zeros = []
        if "head" in self.key:
            limits = self.rob.physical_limits_head
            for key in limits:
                zeros.append(random.randrange(limits[key][0], limits[key][1] + delta_angle, delta_angle))
            if collision and "Pitch" in joint_list:
                zeros = -np.abs(zeros)

        elif "left" in self.key or "right" in self.key:
            limits = self.rob.physical_limits_left
            for key in limits:
                zeros.append(random.randrange(limits[key][0], limits[key][1] + delta_angle, delta_angle))
            if collision:
                if "Roll" in joint_list:
                    zeros = np.abs(zeros)
                elif "Pitch+" in joint_list:
                    zeros = np.abs(zeros)
                elif "Pitch-" in joint_list:
                    zeros = -np.abs(zeros)
        return np.array(zeros)

    def joint_angle_publisher(self):
        """
        Publishes motor commands based on the current angle positions of the joints.

        This function waits for subscriber connections and then publishes motor commands to the appropriate topic
        based on the value of the 'key' attribute. If 'key' contains 'head', the motor command is published to the
        'head_pub' topic. If 'key' contains 'right', the motor command is published to the 'right_pub' topic. If
        'key' contains 'left', the motor command is published to the 'left_pub' topic.

        The motor command is constructed using the current angle positions of the joints stored in the
        'current_ang_pos_head' and 'current_ang_pos_arm' attributes.

        This function also includes a timeout mechanism to prevent waiting indefinitely for subscriber connections.

        Raises:
            rospy.ROSInterruptException: If there is an error while publishing the motor command.

        """
        wtime_begin = rospy.get_time()
        while (self_explorator.right_pub.get_num_connections() == 0):
            rospy.loginfo("waiting for subscriber connections...")
            if rospy.get_time() - wtime_begin > 10.0:
                rospy.logerr("Timeout while waiting for subscribers connection!")
                sys.exit()
            rospy.sleep(1)
        rospy.loginfo("publishing motor command...")
        try:
            if 'head' in self.key:
                ref_head = Float64MultiArray()
                ref_head.data = [self.current_ang_pos_head[0], self.current_ang_pos_head[1]]
                self_explorator.head_pub.publish(ref_head)
            elif 'right' in self.key:
                ref_right = Float64MultiArray()
                ref_right.data = [self.current_ang_pos_arm[0], self.current_ang_pos_arm[1], self.current_ang_pos_arm[2]]
                self_explorator.right_pub.publish(ref_right)
            elif 'left' in self.key:
                ref_left = Float64MultiArray()
                ref_left.data = [self.current_ang_pos_arm[0], self.current_ang_pos_arm[1], self.current_ang_pos_arm[2]]
                self_explorator.left_pub.publish(ref_left)
            rospy.sleep(2)
        except rospy.ROSInterruptException:
            rospy.logerr("could not publish motor command!")
        rospy.loginfo("motor command published")

    def execute_online(self, delta_angle, amount_of_points):
        """
        Executes the online motor command by waiting for subscriber connections,
        publishing the motor command, and handling any exceptions.

        Args:
            delta_angle (float): The delta angle for the motor command.
            amount_of_points (int): The number of points for the motor command.

        Returns:
            None
        """
        wtime_begin = rospy.get_time()
        while (self.right_pub.get_num_connections() == 0):
            rospy.loginfo("waiting for subscriber connections...")
            if rospy.get_time() - wtime_begin > 10.0:
                rospy.logerr("Timeout while waiting for subscribers connection!")
                sys.exit()
            rospy.sleep(1)
        rospy.loginfo("publishing motor command...")
        try:
            self.motor_babbling(delta_angle, amount_of_points)
        except rospy.ROSInterruptException:
            rospy.logerr("could not publish motor command!")
        rospy.loginfo("motor command published")

if __name__ == '__main__':
    config = read_yaml_file("config.yaml")
    robotName = config["robot-name"]
    babblingPoints = config["babbling-points"]
    #delta recommended 5
    delta = config["minimum-distance"]

    rospy.init_node('self_exploration')
    rospy.loginfo("started!")

    directory = os.getcwd()
    file_path = directory  + "/robot_configuration_files/" + robotName + ".yaml"

    self_explorator = SelfExploration(robotName)
    self_explorator.import_robot(file_path)
    self_explorator.execute_online(delta, babblingPoints)

    file_path = directory  + "/data/test_" + robotName + "/self_exploration/self_exploration_qt_" + str(babblingPoints) + ".txt"
    with open(file_path, 'w') as file:
        json.dump(str(self_explorator.motor_babbling_recording), file)