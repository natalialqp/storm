#!/usr/bin/env python
import threading
import rospy
import numpy as np
import pandas as pd

from qt_nuitrack_app.msg import Skeletons
from migrave_skeleton_tools.tf_utils import TFUtils
from migrave_skeleton_tools_ros.skeleton_utils import JointUtils
from qt_robot_interface.srv import speech_say, speech_recognize
from std_msgs.msg import Float64MultiArray

from utils.path_planning import PathPlanning
import utils.pose_prediction as pose_prediction
from main import read_yaml_file

head_pub = rospy.Publisher('/qt_robot/head_position/command', Float64MultiArray, queue_size=10)
right_pub = rospy.Publisher('/qt_robot/right_arm_position/command', Float64MultiArray, queue_size = 10)
left_pub = rospy.Publisher('/qt_robot/left_arm_position/command', Float64MultiArray, queue_size = 10)

human_skeletons_left = []
human_skeletons_right = []
human_skeletons_head = []
timestamp = []
robot_left_rec = []
robot_right_rec = []
robot_head_rec = []

rospy.sleep(1.0)

class SkeletonMarkers(object):
    """
    Class not used in the current implementation.
    """

    def __init__(self, id, skeleton_frame_id):
        """
        Initializes an instance of the RealTimePosePrediction class.

        Args:
            id (int): The ID of the instance.
            skeleton_frame_id (str): The ID of the skeleton frame.

        Attributes:
            id (int): The ID of the instance.
            skeleton_frame_id (str): The ID of the skeleton frame.
            upper_body_marker: The marker for the upper body.
            left_hand_marker: The marker for the left hand.
            right_hand_marker: The marker for the right hand.
            left_leg_marker: The marker for the left leg.
            right_leg_marker: The marker for the right leg.
        """
        self.id = id
        self.skeleton_frame_id = skeleton_frame_id
        id_pad = self.id * 100
        self.upper_body_marker = self.get_marker(id_pad + 0)
        self.left_hand_marker = self.get_marker(id_pad + 1)
        self.right_hand_marker = self.get_marker(id_pad + 2)
        self.left_leg_marker = self.get_marker(id_pad + 3)
        self.right_leg_marker = self.get_marker(id_pad + 4)

class SpeechManager(object):
    """
    A class that manages speech recognition and synthesis.

    This class provides methods to start speech recognition, recognize speech input,
    and respond accordingly based on the recognized speech.
    """

    def __init__(self):
        """
        Initializes the SpeechManager class.

        This function sets up the necessary service proxies for speech recognition and speech synthesis.
        It also initializes the options for speech recognition and sets the initial speech message.
        The function also initializes variables for storing the speech response, the last command, and the speech thread.
        """
        self.recognize_speech_proxy = rospy.ServiceProxy('/qt_robot/speech/recognize', speech_recognize)
        self.speechSay = rospy.ServiceProxy('/qt_robot/speech/say', speech_say)
        self.options = ["record", "stop"]
        self.speechSay("INIT SPEECH RECOGNITION")

        self.resp = None
        self.last_cmd = []
        self.speech_thread = None
        self.is_alive = False

    def start_recognition(self):
        """
        Starts the speech recognition process in a separate thread.

        This method sets the `is_alive` flag to True and starts a new thread to run the `recognize_speech` method.
        The thread is set as a daemon thread to allow the program to exit even if the thread is still running.

        """
        self.is_alive = True
        self.speech_thread = threading.Thread(target=self.recognize_speech)
        self.speech_thread.daemon = True
        self.speech_thread.start()

    def recognize_speech(self):
        """
        Recognizes speech using a speech recognition proxy.

        This function continuously listens for speech input using a speech recognition proxy.
        It waits for a response and assigns it to the `resp` attribute of the object.
        The last recognized command is stored in the `last_cmd` attribute.
        After receiving a response, it calls the `robot_say_response` function to process the response.
        """
        while self.is_alive:
            print("QT is listening... Watch out")
            self.resp = None
            while self.resp is None:
                self.resp = self.recognize_speech_proxy("en_US", self.options, 3)
                self.last_cmd = self.resp.transcript
            self.robot_say_response()

    def robot_say_response(self):
        """
        Performs speech recognition and responds accordingly.

        If the transcript contains the word "record", it starts recording and says "Recording started".
        If the transcript contains the word "stop", it stops recording and says "Recording stopped".
        If the transcript doesn't contain any valid option, it says "No valid option".

        Returns:
            None
        """
        rospy.loginfo("Speech recognition started")
        if "record" in self.resp.transcript:
            rospy.loginfo("Recording started")
            self.speechSay("Recording started")
        elif "stop" in self.resp.transcript:
            rospy.loginfo("Recording stopped")
            self.speechSay("Recording stopped")
            self.is_alive = False
        else:
            rospy.loginfo("No valid option")
            self.speechSay("No valid option")
        return

class HumanSkeleton(object):
    """
    Represents a human skeleton and provides methods for real-time pose prediction.

    Attributes:
        nuitrack_skeleton_topic (str): The topic for receiving skeleton data from Nuitrack.
        cam_base_link_translation (list): The translation values for the camera base link.
        skeleton_frame_id (str): The frame ID for the skeleton data.
        cam_base_link_rot (list): The rotation values for the camera base link.
        pose_predictor: The pose predictor object.
        planner: The planner object.
        delta (int): The delta value.
        cam_base_link_tf (numpy.ndarray): The homogeneous transform matrix for the camera base link.
        skeleton_sub: The subscriber object for receiving skeleton data.
    """

    def __init__(self, pose_predictor, planner):
        """
        Initializes the HumanSkeleton class.

        Args:
            pose_predictor: The pose predictor object.
            planner: The planner object.
        """
        self.nuitrack_skeleton_topic = rospy.get_param('~nuitrack_skeleton_topic',
                                                       '/qt_nuitrack_app/skeletons')
        self.cam_base_link_translation = rospy.get_param('~cam_base_link_translation', [0., 0., 0.])
        self.skeleton_frame_id = rospy.get_param('~skeleton_frame_id', 'base_link')
        self.cam_base_link_rot = rospy.get_param('cam_base_link_rot', [np.pi/2, 0., np.pi/2])
        self.pose_predictor = pose_predictor
        self.planner = planner
        self.delta = 0
        self.cam_base_link_tf = TFUtils.get_homogeneous_transform(self.cam_base_link_rot,
                                                                  self.cam_base_link_translation)
        self.skeleton_sub = None

    def start(self):
        """
        Starts the real-time pose prediction.

        This function subscribes to the Nuitrack skeleton topic and captures the skeletons
        for further processing.
        """
        self.skeleton_sub = rospy.Subscriber(self.nuitrack_skeleton_topic,
                                             Skeletons,
                                             self.capture_skeletons)
        rospy.sleep(0.5)

    def stop(self):
        """
        Stops the skeleton subscription.

        Unregisters the skeleton subscriber if it is not None.
        """
        if self.skeleton_sub is not None:
            self.skeleton_sub.unregister()

    def save_action(self, action_name, action):
        """
        Save the action in a CSV file.

        Parameters:
        - action_name (str): The name of the action.
        - action (list): The action data to be saved.
        """
        action_df = pd.DataFrame(action)
        action_df.to_csv("data/QT_recordings/" + action_name + ".csv", index=False)

    def capture_skeletons(self, skeleton_collection_msg):
        """
        Captures and processes skeleton data from a skeleton collection message.

        Args:
            skeleton_collection_msg (SkeletonCollectionMsg): The skeleton collection message containing the skeleton data.
        """
        cartesian_left_vec = []
        cartesian_right_vec = []
        cartesian_head_vec = []

        for skeleton_msg in skeleton_collection_msg.skeletons:
            right_arm_points = {}
            left_arm_points = {}
            head_points = {}
            for joint in skeleton_msg.joints:
                # the joint msg contains the position in mm
                position = np.array(joint.real)  # / 1000.
                position_hom = np.array([[position[0]], [position[1]], [position[2]], [1.]])
                position_base_link = self.cam_base_link_tf.dot(position_hom)
                position_base_link = position_base_link.flatten()[0:3]
                position_base_link[1] = -position_base_link[1]
                joint_name = JointUtils.JOINTS[joint.type]
                if joint_name in JointUtils.JOINTS_TO_IGNORE:
                    continue
                if joint_name in JointUtils.BODY_JOINT_NAMES:
                    head_points[joint_name] = position_base_link
                elif joint_name in JointUtils.LEFT_ARM_JOINT_NAMES:
                    left_arm_points[joint_name] = position_base_link
                elif joint_name in JointUtils.RIGHT_ARM_JOINT_NAMES:
                    right_arm_points[joint_name] = position_base_link
            # Define the common first point for the head as the middle point between the shoulders
            head_points['JOINT_LEFT_COLLAR'] = left_arm_points['JOINT_LEFT_COLLAR']
            current_time = rospy.Time.now().to_sec()
            human_skeletons_left.append(left_arm_points)
            human_skeletons_right.append(right_arm_points)
            human_skeletons_head.append(head_points)
            timestamp.append(current_time)
            left_side, right_side, head = self.pose_predictor.dicts_to_lists(left_arm_points, right_arm_points, head_points)
            left_arm_pred, right_arm_pred, head_pred = self.pose_predictor.predict_pytorch(left_side, right_side, head)
            left_cart, right_cart, head_cart = self.pose_predictor.robot_embodiment(left_arm_pred, right_arm_pred, head_pred)
            # rospy.loginfo("predicted...")
            # rospy.loginfo(self.delta)
            cartesian_left_vec.append(left_cart)
            cartesian_right_vec.append(right_cart)
            cartesian_head_vec.append(head_cart)
            self.delta = self.delta + 1
            if self.delta >= 250:
                rospy.loginfo(self.delta)
                self.delta = 0
                predicted_angles_left, predicted_angles_right, predicted_angles_head = self.find_end_effector_angles(cartesian_left_vec, cartesian_right_vec, cartesian_head_vec)
                joint_angle_publisher(predicted_angles_left, predicted_angles_right, predicted_angles_head)
                cartesian_left_vec = []
                cartesian_right_vec = []
                cartesian_head_vec = []
                robot_left_rec.append(predicted_angles_left)
                robot_right_rec.append(predicted_angles_right)
                robot_head_rec.append(predicted_angles_head)
        return

    def find_end_effector_angles(self, left_arm_pred, right_arm_pred, head_pred):
        """
        Finds the angles of the end effectors for the left arm, right arm, and head based on the given predictions.

        Parameters:
        - left_arm_pred (dict): Predicted values for the left arm.
        - right_arm_pred (dict): Predicted values for the right arm.
        - head_pred (dict): Predicted values for the head.

        Returns:
        - angles_left (list): List of angles for the left arm end effectors.
        - angles_right (list): List of angles for the right arm end effectors.
        - angles_head (list): List of angles for the head end effectors.
        """
        angles_left = []
        angles_right = []
        angles_head = []
        for key in self.planner.end_effectors_keys:
            if "Left" in key:
                angles_left = self.planner.path_planning_online(left_arm_pred, self.planner.robot_graphs[key])
            elif "Right" in key:
                angles_right = self.planner.path_planning_online(right_arm_pred, self.planner.robot_graphs[key])
            elif "Head" in key:
                angles_head = self.planner.path_planning_online(head_pred, self.planner.robot_graphs[key])
        return list(angles_left.values()), list(angles_right.values()), list(angles_head.values())

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
        right_pub.publish(ref_right)
        left_pub.publish(ref_left)
        # head_pub.publish(ref_head)
        # rospy.sleep(0.05)
    except rospy.ROSInterruptException:
        rospy.logerr("could not publish motor command!")
    rospy.loginfo("motor command published")

def initialise(robotName, babblingPoints, epochs):
    """
    Initializes the pose predictor and planner for real-time pose prediction.

    Returns:
        pose_predictor (Prediction): The pose predictor object.
        planner (PathPlanning): The planner object.
    """
    file_path = "./robot_configuration_files/" + robotName + ".yaml"
    planner = PathPlanning(file_path)
    pose_predictor = pose_prediction.Prediction(file_path, robotName)
    planner.fill_robot_graphs(babblingPoints)

    #NN TRAINING
    file_name = "robot_angles_" + robotName
    theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot = pose_predictor.read_training_data(file_name)
    pose_predictor.train_pytorch(pose_predictor.robot, theta_left, phi_left, theta_right, phi_right, theta_head, phi_head, left_arm_robot, right_arm_robot, head_robot, epochs)
    return pose_predictor, planner

if __name__ == '__main__':
    config = read_yaml_file("config.yaml")
    robotName = config["robot-name"]
    babblingPoints = config["babbling-points"]
    epochs = config["epochs"]

    rospy.init_node('real_time_pose_estimation_node')
    rospy.loginfo("real_time_pose_estimation_node started!")
    # speech = SpeechManager()
    # speech.start_recognition()

    pose_predictor, planner = initialise(robotName, babblingPoints, epochs)
    skeleton = HumanSkeleton(pose_predictor, planner)
    while not rospy.is_shutdown():
        skeleton.start()

    # try:
    #     while not rospy.is_shutdown():
    #         skeleton.start()
        # while not rospy.is_shutdown() and speech.is_alive:
        #     if "record" in speech.last_cmd:
        #         skeleton.start()
        #     elif "stop" in speech.last_cmd:
        #         skeleton.stop()
        #         new_action_name = "dummy"
        #         skeleton.save_action("robot/" + new_action_name, np.hstack((robot_left_rec, robot_right_rec, robot_head_rec)))
        #         skeleton.save_action("human/" + new_action_name, np.hstack((timestamp, human_skeletons_left, human_skeletons_right, human_skeletons_head)))
    # except KeyboardInterrupt:
    #     pass
    rospy.loginfo("finished!")

