#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray
import pandas as pd
from math import degrees
from main import read_yaml_file

left_columns = ['left_1', 'left_2', 'left_3']
right_columns = ['right_1', 'right_2', 'right_3']
head_columns = ['head_1', 'head_2']

left_angle_sequence = []
right_angle_sequence = []
head_angle_sequence = []

def read_action_from_file(file_path):
    """
    Read action data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing three lists: left_angle_sequence, right_angle_sequence, and head_angle_sequence.
               Each list contains a sequence of angles corresponding to the left, right, and head actions, respectively.
    """
    df = pd.read_csv(file_path)
    time_column = df['time']
    signal_columns = df.drop(columns=['time'])
    for index, row in df.iterrows():
        left_values = [degrees(row[column]) for column in left_columns]
        right_values = [degrees(row[column]) for column in right_columns]
        head_values = [degrees(row[column]) for column in head_columns]
        left_angle_sequence.append(left_values)
        right_angle_sequence.append(right_values)
        head_angle_sequence.append(head_values)
    return left_angle_sequence, right_angle_sequence, head_angle_sequence

def joint_angle_publisher(left_arm_ang_pos, right_arm_ang_pos, head_ang_pos):
    """
    Publishes the joint angles to the robot.

    Args:
        left_arm_ang_pos (list): List of joint angles for the left arm.
        right_arm_ang_pos (list): List of joint angles for the right arm.
        head_ang_pos (list): List of joint angles for the head.

    Returns:
        None
    """
    # Publish angles to robot
    rospy.loginfo("publishing motor command...")
    try:
        ref_head = Float64MultiArray()
        ref_right = Float64MultiArray()
        ref_left = Float64MultiArray()
        ref_head.data = head_ang_pos
        ref_right.data = right_arm_ang_pos
        ref_left.data = left_arm_ang_pos
        head_pub.publish(ref_head)
        right_pub.publish(ref_right)
        left_pub.publish(ref_left)
        rospy.sleep(0.43)
    except rospy.ROSInterruptException:
        rospy.logerr("could not publish motor command!")
    rospy.loginfo("motor command published")

if __name__ == '__main__':
    config = read_yaml_file("config.yaml")
    robotName = config["robot-name"]
    babblingPoints = config["babbling-points"]
    action = config["action-name"]

    rospy.init_node('qt_motor_command')
    dir = './data/test_' + robotName + '/GMM_learned_actions/for_execution/'
    file_name = dir + "GMR_" + babblingPoints + "_" + action + ".csv"
    # file_name = dir + "qt_sugar_bowl_for_execution.csv"
    # create a publisher
    left_pub = rospy.Publisher('/qt_robot/left_arm_position/command', Float64MultiArray, queue_size=1)
    right_pub = rospy.Publisher('/qt_robot/right_arm_position/command', Float64MultiArray, queue_size=1)
    head_pub = rospy.Publisher('/qt_robot/head_arm_position/command', Float64MultiArray, queue_size=1)
    left_angle_sequence, right_angle_sequence, head_angle_sequence = read_action_from_file(file_name)

    # wait for publisher/subscriber connections
    for i in range(len(left_angle_sequence)):
        joint_angle_publisher(left_angle_sequence[i], right_angle_sequence[i], head_angle_sequence[i])